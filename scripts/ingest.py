"""
scripts/ingest.py
─────────────────
CLI script to ingest MSMARCO-XI dataset into Qdrant.

Executes the pipeline:
  HuggingFace dataset stream → HierarchicalChunker → BGEM3Embedder → QdrantIndexer
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
):
    """Run the ingestion pipeline."""
    settings = get_settings()

    target_langs = (
        [lang.strip() for lang in languages.split(",") if lang.strip()]
        if languages
        else settings.ingestion.languages
    )
    target_limit = limit if limit > 0 else settings.ingestion.ingest_limit

    console.print("[bold blue]Starting MSMARCO-XI Ingestion Pipeline[/bold blue]")
    console.print(f"Languages: {target_langs}")
    console.print(f"Limit per language: {target_limit or 'All'}")

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

    # ── Processing Loop ───────────────────────────────────────────────────
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

        # We accumulate chunks to batch embed and index
        batch_chunks = []
        flush_size = settings.embedding.batch_size * 10  # process ~640 chunks at a time

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            
            # Note: We can't know total queries upfront with streaming datasets
            task = progress.add_task(f"Ingesting {lang}...", total=None)
            queries_processed = 0

            for entry in ds_stream:
                parents, children = chunker.chunk_entry(entry)
                # We index BOTH parents and children.
                # Parents are stored but not retrieved (query_type="parent").
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
                _flush_batch(batch_chunks, embedder, indexer)
                total_chunks += len(batch_chunks)
                progress.update(task, advance=queries_processed)

    elapsed = time.time() - start_time
    console.print("\n[bold blue]Ingestion Complete![/bold blue]")
    console.print(f"Total chunks indexed: {total_chunks}")
    console.print(f"Time elapsed: {elapsed:.2f} seconds")


def _flush_batch(chunks: List, embedder: BGEM3Embedder, indexer: QdrantIndexer) -> None:
    """Helper to embed and upsert a batch of chunks."""
    if not chunks:
        return

    texts = [chunk.text for chunk in chunks]
    # Embed (returns dense array and list of sparse dicts)
    dense_vecs, sparse_vecs = embedder.embed_chunks_batch(texts, show_progress=False)
    # Upsert to Qdrant
    indexer.upsert_chunks(chunks, dense_vecs, sparse_vecs)


if __name__ == "__main__":
    app()
