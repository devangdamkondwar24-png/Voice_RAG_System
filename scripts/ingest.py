"""
scripts/ingest.py
─────────────────
CLI script to ingest MSMARCO-XI dataset into Qdrant.

Full pipeline:
  HuggingFace dataset stream
  → DatasetPreprocessor (Unicode / dedup / lang-validation / digit-standardize)
  → HierarchicalChunker
  → BGEM3Embedder
  → QdrantIndexer
"""

from __future__ import annotations

import logging
import time
from typing import List

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from config.settings import get_settings
from ingestion.chunking import HierarchicalChunker
from ingestion.dataset_loader import load_language_dataset
from ingestion.embedder import BGEM3Embedder
from ingestion.indexer import QdrantIndexer
from ingestion.preprocessor import DatasetPreprocessor, PassageDeduplicator

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("ingest")
console = Console()
app = typer.Typer(help="Ingest MSMARCO-XI dataset into Qdrant.")


@app.command()
def run(
    languages: str = typer.Option(
        "",
        help="Comma-separated language codes (e.g., 'hi,ta,te'). If empty, uses config.",
    ),
    limit: int = typer.Option(
        0, help="Max queries per language to ingest. 0 uses config limit."
    ),
    recreate: bool = typer.Option(
        False, "--recreate", help="Drop and recreate existing Qdrant collection."
    ),
    skip_dedup: bool = typer.Option(
        False, "--skip-dedup", help="Disable cross-query deduplication (faster, larger index)."
    ),
):
    """Run the ingestion pipeline with preprocessing."""
    settings = get_settings()

    target_langs = (
        [lang.strip() for lang in languages.split(",") if lang.strip()]
        if languages
        else settings.ingestion.languages
    )
    target_limit = limit if limit > 0 else settings.ingestion.ingest_limit

    console.print("[bold blue]Starting MSMARCO-XI Ingestion Pipeline[/bold blue]")
    console.print(f"Languages:           {target_langs}")
    console.print(f"Limit per language:  {target_limit or 'All'}")
    console.print(f"Deduplication:       {'OFF' if skip_dedup else 'ON'}")

    # ── Initialize Components ───────────────────────────────────────────────
    with console.status("Initializing embedder and Qdrant indexer..."):
        embedder = BGEM3Embedder(
            model_name=settings.embedding.model_name,
            device=settings.embedding.device,
            batch_size=settings.embedding.batch_size,
            return_sparse=settings.embedding.return_sparse,
        )
        # Pre-load model so we can pass it to the chunker for semantic splitting
        embedder._load_model()

        chunker = HierarchicalChunker(
            child_min_tokens=settings.chunking.child_min_tokens,
            child_max_tokens=settings.chunking.child_max_tokens,
            overlap_percent=settings.chunking.overlap_percent,
            semantic_threshold=settings.chunking.semantic_split_threshold,
            embedder=embedder._model,
        )

        indexer = QdrantIndexer(
            url=settings.qdrant.url,
            api_key=settings.qdrant.api_key,
            collection_name=settings.qdrant.collection_name,
            hnsw_m=settings.qdrant.hnsw_m,
            hnsw_ef_construct=settings.qdrant.hnsw_ef_construct,
            dense_dim=settings.qdrant.dense_dim,
            use_scalar_quantization=settings.qdrant.use_scalar_quantization,
        )

        indexer.setup_collection(recreate=recreate)

    # ── Shared deduplicator (spans all languages so cross-lang dupes are caught too)
    pp_cfg = settings.preprocessing
    deduplicator = PassageDeduplicator()

    preprocessor = DatasetPreprocessor(
        enable_unicode_nfc=pp_cfg.enable_unicode_nfc,
        enable_text_cleaning=pp_cfg.enable_text_cleaning,
        enable_language_validation=pp_cfg.enable_language_validation,
        language_validation_threshold=pp_cfg.language_validation_threshold,
        enable_deduplication=(not skip_dedup) and pp_cfg.enable_deduplication,
        enable_numeric_standardization=pp_cfg.enable_numeric_standardization,
        min_passage_length=pp_cfg.min_passage_length,
        preserve_raw_text=pp_cfg.preserve_raw_text,
        deduplicator=deduplicator,
    )

    # ── Processing Loop ────────────────────────────────────────────────────
    total_chunks = 0
    start_time = time.time()

    for lang in target_langs:
        console.print(f"\n[bold green]Processing Language: {lang.upper()}[/bold green]")

        ds_stream = load_language_dataset(
            language=lang,
            dataset_name=settings.ingestion.dataset_name,
            split=settings.ingestion.split,
            limit=target_limit,
        )

        batch_chunks = []
        flush_size = settings.embedding.batch_size * 10  # ~640 chunks per flush

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:

            task = progress.add_task(f"Ingesting {lang}...", total=None)
            queries_processed = 0

            for raw_entry in ds_stream:
                # ── Preprocessing stage ──────────────────────────────────
                entry = preprocessor.preprocess_entry(raw_entry)
                if entry is None:
                    continue  # Structural failure — skip whole entry

                # ── Chunking stage ───────────────────────────────────────
                # Build a lightweight adapter so HierarchicalChunker is satisfied
                # (it expects .passages with .passage_text / .is_selected / .passage_rank)
                parents, children = _chunk_preprocessed_entry(entry, chunker)
                batch_chunks.extend(parents)
                batch_chunks.extend(children)
                queries_processed += 1

                if len(batch_chunks) >= flush_size:
                    _flush_batch(batch_chunks, embedder, indexer)
                    total_chunks += len(batch_chunks)
                    batch_chunks = []
                    progress.update(task, advance=queries_processed)
                    queries_processed = 0

            # Flush remaining
            if batch_chunks:
                _flush_batch(batch_chunks, embedder, indexer, wait_final=True)
                total_chunks += len(batch_chunks)
                progress.update(task, advance=queries_processed)

    elapsed = time.time() - start_time
    console.print("\n[bold blue]Ingestion Complete![/bold blue]")
    console.print(f"Total chunks indexed: {total_chunks}")
    console.print(f"Time elapsed: {elapsed:.2f} seconds")
    # Print full preprocessing statistics
    preprocessor.log_stats()


def _chunk_preprocessed_entry(entry, chunker: HierarchicalChunker):
    """
    Adapter that converts a PreprocessedEntry into a chunker-compatible object
    and calls chunk_entry().  We use a simple namespace-style object to avoid
    creating a hard dependency between preprocessor.py and chunking.py.
    """
    from types import SimpleNamespace

    # Rebuild a minimal DatasetEntry-like object from the preprocessed entry
    passages = [
        SimpleNamespace(
            passage_text=p.text,
            english_text=p.english_text,
            is_selected=p.is_selected,
            passage_rank=p.passage_rank,
        )
        for p in entry.passages
    ]
    adapter = SimpleNamespace(
        query_id=entry.query_id,
        language=entry.language,
        query_type=entry.query_type,
        passages=passages,
    )
    return chunker.chunk_entry(adapter)


def _flush_batch(
    chunks: List,
    embedder: BGEM3Embedder,
    indexer: QdrantIndexer,
    wait_final: bool = False,
) -> None:
    """Helper to embed and upsert a batch of chunks."""
    if not chunks:
        return

    # Only embed child chunks; parent chunks don't need vectors (payload-only)
    child_chunks = [c for c in chunks if c.chunk_type == "child"]
    parent_chunks = [c for c in chunks if c.chunk_type == "parent"]

    if child_chunks:
        texts = [c.text for c in child_chunks]
        dense_vecs, sparse_vecs = embedder.embed_chunks_batch(texts, show_progress=False)
        indexer.upsert_chunks(child_chunks, dense_vecs, sparse_vecs, wait=wait_final)

    if parent_chunks:
        import numpy as np
        # Upsert parents as payload-only (empty vectors)
        indexer.upsert_chunks(
            parent_chunks,
            np.empty((0, 1024), dtype=np.float32),
            [],
            wait=wait_final,
        )


if __name__ == "__main__":
    app()
