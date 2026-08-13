"""
ingestion/embedder.py
──────────────────────
BGE-M3 batch embedder for both ingestion (bulk) and query-time (single).

Design rationale:
- BGE-M3 is unique in supporting dense + sparse (lexical-weight) vectors
  in a SINGLE forward pass via FlagEmbedding. This is critical: we get
  both retrieval signals without doubling inference time.
- Dense vector (1024-dim):   Semantic similarity (cosine). Note: dense vectors
  from FlagEmbedding are strictly L2-normalized, requiring Distance.COSINE in Qdrant.
- Sparse vector (vocab-dim): BM25-like lexical matching
- At ingest time: batch_size=64 balances GPU utilization vs. OOM risk.
- At query time: batch_size=1 (single query) ← latency critical path.

GPU memory: BGE-M3 ≈ 2.3 GB in FP16. Safe to co-load with reranker (2.3 GB)
            and DeBERTa-v3 (0.7 GB) on a 24 GB card.
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)


class BGEM3Embedder:
    """
    Wrapper around FlagEmbedding's BGEM3FlagModel.

    Supports:
    - Dense embedding (1024-dim, float32, L2-normalized)
    - Sparse embedding (dict of token_id → weight, for BM25-like retrieval)
    - Batched ingestion + single-query inference
    - Thread-safe lazy loading of the model

    Args:
        model_name:   HuggingFace model ID (default: BAAI/bge-m3)
        device:       "cuda" | "cpu" | "mps"
        batch_size:   Batch size for bulk encoding (query uses batch_size=1)
        use_fp16:     FP16 inference — halves GPU memory, <0.1% accuracy loss
        return_sparse: Also compute sparse (BM25-like) vectors
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cuda",
        batch_size: int = 64,
        use_fp16: bool = True,
        return_sparse: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.return_sparse = return_sparse
        self._model = None  # lazy-loaded
        self._lock = threading.Lock()
        
        # Issue 3: Validate FP16 constraint for CPU
        if self.device != "cuda" and use_fp16:
            logger.warning(
                f"use_fp16=True is not supported on device '{self.device}'. "
                f"Falling back to full precision (FP32)."
            )
            self.use_fp16 = False
        else:
            self.use_fp16 = use_fp16

    def _load_model(self):
        """Lazy-load BGE-M3 model on first use (saves RAM during import)."""
        if self._model is not None:
            return

        with self._lock:
            # Double-checked locking
            if self._model is not None:
                return
                
            try:
                from FlagEmbedding import BGEM3FlagModel  # type: ignore

                logger.info(f"Loading BGE-M3 from '{self.model_name}' on {self.device}…")
                self._model = BGEM3FlagModel(
                    self.model_name,
                    use_fp16=self.use_fp16,
                    device=self.device,
                )
                logger.info("BGE-M3 loaded successfully.")
            except ImportError:
                raise ImportError(
                    "FlagEmbedding not installed. Run: pip install FlagEmbedding"
                )

    def embed_query(self, text: str) -> Tuple[np.ndarray, Optional[Dict[int, float]]]:
        """
        Embed a single query string.

        Returns:
            (dense_vector, sparse_vector)
            dense_vector:  np.ndarray of shape (1024,)
            sparse_vector: dict {token_id: weight} or None if return_sparse=False

        Latency note: single query takes ~20ms on A10 GPU in FP16.
        """
        self._load_model()

        try:
            output = self._model.encode(
                [text],
                batch_size=1,
                max_length=512,      # Queries are short; cap at 512 for speed
                return_dense=True,
                return_sparse=self.return_sparse,
                return_colbert_vecs=False,  # ColBERT disabled — adds latency
            )
        except Exception as exc:
            logger.error(f"FlagEmbedding encode failed for single query: {exc}")
            raise RuntimeError(f"Embedding failed: {exc}") from exc

        dense = np.array(output["dense_vecs"][0], dtype=np.float32)
        sparse = None
        if self.return_sparse and "lexical_weights" in output:
            # FlagEmbedding returns lexical_weights as list of dicts
            lw = output["lexical_weights"]
            sparse = dict(lw[0]) if lw else {}

        return dense, sparse

    def embed_chunks_batch(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> Tuple[np.ndarray, List[Optional[Dict[Any, Any]]]]:
        """
        Embed a large list of text chunks in batches.

        Used during ingestion (bulk mode). Returns:
            dense_vecs:   np.ndarray of shape (N, 1024)
            sparse_vecs:  list of N dicts {token_id: weight}

        Args:
            texts:         List of text strings to embed
            show_progress: Show tqdm progress bar

        Note: Long passages (parent chunks) are truncated to 8192 tokens by
              BGE-M3's max context. MSMARCO passages are well within this limit.
        """
        self._load_model()

        if not texts:
            return np.empty((0, 1024), dtype=np.float32), []

        try:
            output = self._model.encode(
                texts,
                batch_size=self.batch_size,
                max_length=8192,        # BGE-M3 supports up to 8192 tokens
                return_dense=True,
                return_sparse=self.return_sparse,
                return_colbert_vecs=False,
                show_progress_bar=show_progress,
            )
        except Exception as exc:
            logger.error(f"FlagEmbedding bulk encode failed: {exc}")
            raise RuntimeError(f"Bulk embedding failed: {exc}") from exc

        dense_vecs = np.array(output["dense_vecs"], dtype=np.float32)

        sparse_vecs: List[Optional[Dict[Any, Any]]] = []
        if self.return_sparse and "lexical_weights" in output:
            for lw in output["lexical_weights"]:
                sparse_vecs.append(dict(lw) if lw else {})
        else:
            sparse_vecs = [None] * len(texts)

        return dense_vecs, sparse_vecs

    def sparse_to_qdrant(self, sparse: Dict[Any, Any]) -> dict:
        """
        Convert BGE-M3 sparse dict to Qdrant SparseVector format.

        Qdrant expects: {"indices": [int, ...], "values": [float, ...]}
        """
        if not sparse:
            return {"indices": [], "values": []}
            
        # FlagEmbedding might return string keys instead of ints; cast to be safe
        indices = [int(k) for k in sparse.keys()]
        values = [float(sparse[k]) for k in sparse.keys()]
        
        return {"indices": indices, "values": values}
