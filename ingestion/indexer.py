"""
ingestion/indexer.py
─────────────────────
Qdrant collection setup and chunk upsert.

Design rationale:
- Payload indexes on `language`, `chunk_type`, `is_selected` are created
  BEFORE data ingestion. Qdrant uses these to build HNSW graph edges that
  respect filter constraints — enabling sub-millisecond filtered ANN search.
- Named vectors: "dense" (1024-dim float32) + "sparse" (variable, uint32 indices).
  This enables Qdrant's prefetch API to run both retrievals server-side in one call.
- Scalar quantization (INT8): reduces index RAM from ~4 GB to ~1 GB for 1M vectors
  with <0.3% recall degradation. The full precision vectors are stored on disk 
  (on_disk=True) to actually realize the RAM savings, while quantized vectors stay in RAM.
- Batch upsert (512 points/batch): avoids Qdrant gRPC message size limits
  while maximizing throughput.
"""

from __future__ import annotations

import logging
import uuid
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Qdrant collection constants
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

# Payload field names (must match chunking.py to_payload() keys)
FIELD_LANGUAGE = "language"
FIELD_CHUNK_TYPE = "chunk_type"
FIELD_IS_SELECTED = "is_selected"
FIELD_QUERY_ID = "query_id"
FIELD_PARENT_ID = "parent_id"
FIELD_CHUNK_ID = "chunk_id"


def _build_qdrant_point(
    chunk,
    dense_vec: Optional[np.ndarray],
    sparse_vec: Optional[Dict],
) -> dict:
    """
    Build a Qdrant PointStruct-compatible dict from a Chunk + embeddings.

    Uses chunk_id as the Qdrant point UUID (deterministic, reproducible).
    """
    from qdrant_client.models import PointStruct, SparseVector  # type: ignore

    # Issue 7: Use NAMESPACE_OID instead of NAMESPACE_DNS
    point_id = str(uuid.uuid5(uuid.NAMESPACE_OID, chunk.chunk_id))

    vectors: dict = {}

    # Issue 3: Parent chunks are for payload lookup only, don't index them in HNSW
    if chunk.chunk_type != "parent":
        if dense_vec is not None:
            vectors[DENSE_VECTOR_NAME] = dense_vec.tolist()

        if sparse_vec:
            # Issue 1: Cast keys to int explicitly
            indices = [int(k) for k in sparse_vec.keys()]
            values = [float(sparse_vec[k]) for k in sparse_vec.keys()]
            vectors[SPARSE_VECTOR_NAME] = SparseVector(indices=indices, values=values)

    return PointStruct(
        id=point_id,
        vector=vectors,
        payload=chunk.to_payload(),
    )


class QdrantIndexer:
    """
    Manages Qdrant collection lifecycle and bulk upsert of embedded chunks.

    Usage:
        indexer = QdrantIndexer(url="http://localhost:6333")
        indexer.setup_collection()
        indexer.upsert_chunks(chunks, dense_vecs, sparse_vecs)
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: str = "",
        collection_name: str = "msmarco_xi_rag",
        hnsw_m: int = 16,
        hnsw_ef_construct: int = 100,
        dense_dim: int = 1024,
        use_scalar_quantization: bool = True,
        upsert_batch_size: int = 512,
    ) -> None:
        self.url = url
        self.api_key = api_key or None
        self.collection_name = collection_name
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construct = hnsw_ef_construct
        self.dense_dim = dense_dim
        self.use_scalar_quantization = use_scalar_quantization
        self.upsert_batch_size = upsert_batch_size
        self._client = None

    def _get_client(self):
        """Lazy-init Qdrant client (prefers gRPC for batch upsert throughput)."""
        if self._client is not None:
            return self._client

        from qdrant_client import QdrantClient  # type: ignore

        kwargs: dict = {"url": self.url, "prefer_grpc": True}
        if self.api_key:
            kwargs["api_key"] = self.api_key

        self._client = QdrantClient(**kwargs)
        return self._client

    def collection_exists(self) -> bool:
        client = self._get_client()
        existing = [c.name for c in client.get_collections().collections]
        return self.collection_name in existing

    def setup_collection(self, recreate: bool = False) -> None:
        """
        Create the Qdrant collection with dual-vector config and payload indexes.

        Args:
            recreate: If True, drop and recreate existing collection.
                      WARNING: This deletes all indexed data.
        """
        from qdrant_client.models import (  # type: ignore
            Distance,
            HnswConfigDiff,
            ScalarQuantization,
            ScalarQuantizationConfig,
            ScalarType,
            SparseVectorParams,
            VectorParams,
            PayloadSchemaType,
        )

        client = self._get_client()

        if self.collection_exists():
            if recreate:
                logger.warning(
                    f"Dropping existing collection '{self.collection_name}'…"
                )
                client.delete_collection(self.collection_name)
            else:
                logger.info(
                    f"Collection '{self.collection_name}' already exists. Skipping creation."
                )
                return

        logger.info(f"Creating collection '{self.collection_name}'…")

        # ── HNSW + quantization config ─────────────────────────────────────
        hnsw_config = HnswConfigDiff(
            m=self.hnsw_m,
            ef_construct=self.hnsw_ef_construct,
            # on_disk=False keeps the HNSW graph in RAM for <5ms navigation
            on_disk=False,
        )

        quantization_config = None
        # Issue 2: Actually realize RAM savings by offloading full-precision vectors to disk
        vectors_on_disk = False
        if self.use_scalar_quantization:
            vectors_on_disk = True
            quantization_config = ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8,
                    # always_ram: keep quantized vectors in RAM even on large datasets
                    always_ram=True,
                    quantile=0.99,  # Clip top 1% outliers to improve quantization
                )
            )

        client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(
                    size=self.dense_dim,
                    distance=Distance.COSINE,
                    hnsw_config=hnsw_config,
                    quantization_config=quantization_config,
                    on_disk=vectors_on_disk,
                ),
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: SparseVectorParams(),
            },
        )

        # ── Payload indexes (CRITICAL: create before ingestion) ────────────
        # These allow Qdrant's query planner to push filters into HNSW traversal.
        logger.info("Creating payload indexes…")

        index_fields = [
            (FIELD_LANGUAGE, PayloadSchemaType.KEYWORD),
            (FIELD_CHUNK_TYPE, PayloadSchemaType.KEYWORD),
            (FIELD_IS_SELECTED, PayloadSchemaType.INTEGER),
            (FIELD_QUERY_ID, PayloadSchemaType.INTEGER),
            (FIELD_PARENT_ID, PayloadSchemaType.KEYWORD),
        ]

        for field_name, schema_type in index_fields:
            client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=schema_type,
            )
            logger.debug(f"  Index created: {field_name} ({schema_type})")

        logger.info(
            f"Collection '{self.collection_name}' ready with "
            f"HNSW m={self.hnsw_m}, ef_construct={self.hnsw_ef_construct}."
        )

    def upsert_chunks(
        self,
        chunks: list,
        dense_vecs: np.ndarray,
        sparse_vecs: List[Optional[Dict]],
        batch_size: Optional[int] = None,
        wait: bool = False,
    ) -> int:
        """
        Upsert embedded chunks into Qdrant in batches.

        Args:
            chunks:      List of Chunk objects (from chunking.py)
            dense_vecs:  np.ndarray of shape (N, 1024)
            sparse_vecs: List of N sparse dicts (or None entries)
            batch_size:  Override default batch size
            wait:        If True, wait for changes to actually be indexed

        Returns:
            Number of successfully upserted points.
        """
        client = self._get_client()
        bs = batch_size or self.upsert_batch_size
        n = len(chunks)
        total_upserted = 0

        for start in range(0, n, bs):
            end = min(start + bs, n)
            batch_chunks = chunks[start:end]
            
            # Note: We safely allow dense_vecs / sparse_vecs to be empty lists
            # or None if the caller is only upserting payload-only parent chunks.
            points = []
            for i, chunk in enumerate(batch_chunks):
                d_vec = dense_vecs[start + i] if dense_vecs is not None and len(dense_vecs) > start + i else None
                s_vec = sparse_vecs[start + i] if sparse_vecs is not None and len(sparse_vecs) > start + i else None
                points.append(_build_qdrant_point(chunk, d_vec, s_vec))

            try:
                client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=wait,  # async write: faster, safe (WAL backed). callers can force wait=True
                )
                total_upserted += len(points)
                logger.debug(f"Upserted batch {start}–{end} ({len(points)} points)")
            except Exception as exc:
                logger.error(f"Upsert failed for batch {start}–{end}: {exc}")
                raise

        logger.info(f"Upserted {total_upserted}/{n} chunks into '{self.collection_name}'.")
        return total_upserted

    def get_points_by_chunk_ids(self, chunk_ids: List[str]) -> List[dict]:
        """
        Retrieve stored points' payloads by a list of chunk_id strings.
        Used for batched parent-chunk context expansion after retrieval.
        """
        import uuid as _uuid  # type: ignore

        client = self._get_client()
        
        # Issue 4: Batched retrieval for context expansion
        point_ids = [str(_uuid.uuid5(_uuid.NAMESPACE_OID, cid)) for cid in chunk_ids]

        try:
            results = client.retrieve(
                collection_name=self.collection_name,
                ids=point_ids,
                with_payload=True,
                with_vectors=False,
            )
            return [res.payload for res in results if res.payload]
        except Exception as exc:
            logger.warning(f"Failed to retrieve chunk_ids={chunk_ids}: {exc}")
        return []

    def collection_info(self) -> dict:
        """Return collection stats (vector count, status, etc.)."""
        client = self._get_client()
        info = client.get_collection(self.collection_name)
        return {
            "status": str(info.status),
            "vectors_count": info.vectors_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "points_count": info.points_count,
        }
