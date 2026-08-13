"""
retrieval/reranker.py
──────────────────────
Cross-encoder reranking using BGE-Reranker-v2-m3.

Design rationale:
- Hybrid search (dense+sparse) retrieves top 50 candidates, but Bi-Encoders
  miss subtle semantic interactions between query and passage.
- Cross-Encoder processes (query, passage) jointly. It is highly accurate but
  computationally expensive.
- To meet our <25ms budget, we:
  1. Only rerank top 30-50 candidates.
  2. Use FP16 inference.
  3. Expand child chunks to parent chunks AFTER reranking (prevents scoring
     redundant children from the same parent).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Reranks retrieved child chunks using BAAI/bge-reranker-v2-m3.

    Args:
        model_name: HuggingFace model ID
        device: "cuda" | "cpu" | "mps"
        use_fp16: Use half-precision for speed/memory (recommended)
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cuda",
        use_fp16: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.use_fp16 = use_fp16
        self._model = None  # lazy-loaded

    def _load_model(self):
        """Lazy load the reranker model using FlagEmbedding."""
        if self._model is not None:
            return

        try:
            from FlagEmbedding import FlagReranker  # type: ignore

            logger.info(f"Loading Reranker from '{self.model_name}' on {self.device}…")
            self._model = FlagReranker(
                self.model_name,
                use_fp16=self.use_fp16,
                device=self.device,
            )
            logger.info("Reranker loaded successfully.")
        except ImportError:
            raise ImportError(
                "FlagEmbedding not installed. Run: pip install FlagEmbedding"
            )

    def rerank(
        self,
        query: str,
        retrieved_chunks: List[Dict],
        top_k: int = 10,
    ) -> List[Dict]:
        """
        Rerank a list of retrieved chunks (payloads) against the query.

        Args:
            query: User's question
            retrieved_chunks: List of Qdrant payloads from hybrid search
            top_k: Number of top reranked chunks to return

        Returns:
            List of reranked payloads, sorted by relevance score descending.
            Each payload has a new field: `_rerank_score`.
        """
        if not retrieved_chunks:
            return []

        self._load_model()

        # Build pairs for cross-encoder: (query, passage_text)
        pairs = [(query, chunk.get("text", "")) for chunk in retrieved_chunks]

        # Score pairs (FlagReranker natively handles batching internally)
        # Note: BGE-Reranker-v2-m3 outputs unbounded logits, not probabilities
        scores = self._model.compute_score(pairs, normalize=True)

        # If only 1 chunk was passed, FlagReranker might return a float instead of list
        if isinstance(scores, float):
            scores = [scores]

        # Attach scores and sort
        for i, chunk in enumerate(retrieved_chunks):
            chunk["_rerank_score"] = float(scores[i])

        # Sort descending by rerank score
        reranked = sorted(retrieved_chunks, key=lambda x: x["_rerank_score"], reverse=True)

        return reranked[:top_k]

    def expand_to_parents(
        self,
        reranked_children: List[Dict],
        qdrant_indexer,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Expand reranked child chunks into their parent chunks for LLM context.
        Deduplicates so a parent is only included once.

        Args:
            reranked_children: List of child chunk payloads sorted by relevance
            qdrant_indexer: Instance of QdrantIndexer to fetch parent payloads
            top_k: Max number of unique parent passages to return

        Returns:
            List of unique parent chunk payloads, ordered by their highest-scoring child.
        """
        expanded = []
        seen_parents = set()

        for child in reranked_children:
            parent_id = child.get("parent_id")
            if not parent_id or parent_id in seen_parents:
                continue

            # Fetch parent payload from Qdrant
            parent_payload = qdrant_indexer.get_point_by_chunk_id(parent_id)
            if parent_payload:
                # Inherit the child's high score for observability/guardrails
                parent_payload["_inherited_score"] = child.get("_rerank_score", 0.0)
                expanded.append(parent_payload)
                seen_parents.add(parent_id)

            if len(expanded) >= top_k:
                break

        return expanded
