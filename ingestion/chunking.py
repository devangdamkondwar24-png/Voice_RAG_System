"""
ingestion/chunking.py
──────────────────────
Hierarchical parent-child chunking for MSMARCO-XI passages.

═══════════════════════════════════════════════════════════════
CHUNKING STRATEGY — Design Rationale
═══════════════════════════════════════════════════════════════

The MSMARCO-XI dataset's passages are already semantically coherent
units (Microsoft pre-segmented them from web documents). We build
*on top* of this structure with a 3-layer approach:

LAYER 1 — Parent Chunks (512–1024 tokens, per passage)
  ┌─────────────────────────────────────────────────────┐
  │ Each MSMARCO-XI translated_passage → 1 parent chunk │
  │ Stored in Qdrant but NOT used as retrieval target   │
  │ Purpose: rich LLM context (enough surrounding info) │
  └─────────────────────────────────────────────────────┘
  Why? MSMARCO passages are coherent blocks. Splitting them further
  for the parent level would lose cohesion. The LLM needs enough
  context around a fact to answer faithfully.

LAYER 2 — Child Chunks (128–256 tokens, per passage)
  ┌─────────────────────────────────────────────────────┐
  │ Each parent → 1–4 child chunks via semantic split   │
  │ Stored in Qdrant as the PRIMARY retrieval target    │
  │ Each child carries a parent_id pointer              │
  └─────────────────────────────────────────────────────┘
  Why? Dense retrieval (BGE-M3) performs best on focused 150–250
  token chunks. Short chunks → high cosine similarity with the
  specific sentence the user is asking about.

  Semantic splitting:
  - Step 1: Script-aware sentence tokenization (IndicNLP for
    Devanagari/Tamil/Telugu etc.; NLTK for fallback)
  - Step 2: Embed each sentence with BGE-M3
  - Step 3: Compute cosine similarity between adjacent pairs
  - Step 4: Insert split where similarity < threshold (0.65)
    This prevents cutting mid-topic; splits happen at natural
    topic changes.
  - Step 5: Apply 12% overlap (last 1-2 sentences of chunk N
    prepended to chunk N+1) to prevent context-cliff failures
    where the answer spans a split boundary.

LAYER 3 — Metadata Enrichment
  Every chunk carries: language, query_id, is_selected,
  passage_rank, query_type, chunk_type (parent/child), parent_id.
  This enables:
  - Language-filtered retrieval in Qdrant (12× space reduction)
  - Guardrail Layer 2: check is_selected in retrieved chunks
  - Evaluation: know ground-truth relevance for MRR/Recall
═══════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Script-specific end-of-sentence markers ───────────────────────────────
# Used as fallback when IndicNLP is unavailable for a script.
SCRIPT_EOS = {
    "hi": ["।", "?", "!", "."],
    "mr": ["।", "?", "!", "."],
    "bn": ["।", "?", "!", "."],
    "or": ["।", "?", "!", "."],
    "as": ["।", "?", "!", "."],
    "pa": ["।", "?", "!", "."],
    "gu": ["।", "?", "!", "."],
    "ta": [".", "?", "!"],
    "te": [".", "?", "!"],
    "kn": [".", "?", "!"],
    "ml": [".", "?", "!"],
    "ur": ["۔", "?", "!"],
}

# Approximate token:char ratios for each language (used for token estimation)
# Indic scripts are denser than Latin — fewer tokens per character.
CHAR_TO_TOKEN = {
    "hi": 0.4,  "mr": 0.4,  "bn": 0.4,  "or": 0.4,
    "as": 0.4,  "pa": 0.4,  "gu": 0.4,  "ur": 0.4,
    "ta": 0.35, "te": 0.35, "kn": 0.35, "ml": 0.35,
    "en": 0.25,
}


def _estimate_tokens(text: str, language: str) -> int:
    """Estimate token count without a full tokenizer (fast heuristic)."""
    ratio = CHAR_TO_TOKEN.get(language, 0.35)
    return max(1, int(len(text) * ratio))


def _split_sentences_regex(text: str, language: str) -> List[str]:
    """
    Regex-based sentence splitter using script-specific EOS markers.
    Used as fallback when IndicNLP is not available.
    """
    eos_chars = SCRIPT_EOS.get(language, [".", "?", "!"])
    # Build pattern: split after EOS marker followed by space/newline/end
    pattern = "([" + re.escape("".join(eos_chars)) + r"])\s+"
    parts = re.split(pattern, text)

    sentences: List[str] = []
    i = 0
    current = ""
    while i < len(parts):
        current += parts[i]
        if i + 1 < len(parts) and parts[i + 1] in eos_chars:
            current += parts[i + 1]
            i += 2
            if current.strip():
                sentences.append(current.strip())
            current = ""
        else:
            i += 1
    if current.strip():
        sentences.append(current.strip())

    return sentences if sentences else [text]


def _split_sentences(text: str, language: str) -> List[str]:
    """
    Split text into sentences with script-aware tokenization.

    Tries IndicNLP first (better for Indic scripts), falls back to regex.
    """
    try:
        from indicnlp.tokenize import sentence_tokenize  # type: ignore
        sentences = sentence_tokenize.sentence_split(text, lang=language)
        if sentences:
            return [s.strip() for s in sentences if s.strip()]
    except Exception:
        pass

    return _split_sentences_regex(text, language)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _make_chunk_id(language: str, query_id: int, passage_rank: int, child_idx: int) -> str:
    """Generate a deterministic, human-readable chunk ID."""
    return f"{language}_{query_id}_p{passage_rank}_c{child_idx}"


def _make_parent_id(language: str, query_id: int, passage_rank: int) -> str:
    """Generate a deterministic parent chunk ID."""
    return f"{language}_{query_id}_p{passage_rank}_parent"


@dataclass
class Chunk:
    """
    A single chunk ready for embedding and indexing in Qdrant.

    chunk_type == "parent":
        - Contains full passage text (512–1024 tokens)
        - NOT indexed as a retrieval target (stored for context expansion)
        - parent_id points to itself

    chunk_type == "child":
        - Contains a semantic subsection of the passage (128–256 tokens)
        - IS the retrieval target (indexed in Qdrant)
        - parent_id points to the parent chunk
    """

    chunk_id: str
    parent_id: str
    text: str
    chunk_type: str           # "parent" | "child"
    language: str
    query_id: int
    is_selected: int          # 1 = ground-truth relevant for original query
    passage_rank: int         # Index within passage list
    query_type: str           # ENTITY | DESCRIPTION | NUMERIC | LOCATION | PERSON
    token_count: int
    # Optional: dense/sparse embeddings populated later by embedder.py
    dense_embedding: Optional[List[float]] = field(default=None, repr=False)
    sparse_embedding: Optional[dict] = field(default=None, repr=False)

    def to_payload(self) -> dict:
        """Serialize metadata for Qdrant payload (excludes heavy embeddings)."""
        return {
            "chunk_id": self.chunk_id,
            "parent_id": self.parent_id,
            "text": self.text,
            "chunk_type": self.chunk_type,
            "language": self.language,
            "query_id": self.query_id,
            "is_selected": self.is_selected,
            "passage_rank": self.passage_rank,
            "query_type": self.query_type,
            "token_count": self.token_count,
        }


class HierarchicalChunker:
    """
    Produces parent-child chunk pairs from MSMARCO-XI DatasetEntry objects.

    Constructor params:
        child_min_tokens:      Lower bound for child chunk size (default 128)
        child_max_tokens:      Upper bound for child chunk size (default 256)
        overlap_percent:       Fractional overlap between adjacent children (default 0.12)
        semantic_threshold:    Cosine similarity below which a sentence split is forced (default 0.65)
        embedder:              Optional sentence embedder for semantic splitting.
                               If None, falls back to regex-only splitting.
    """

    def __init__(
        self,
        child_min_tokens: int = 128,
        child_max_tokens: int = 256,
        overlap_percent: float = 0.12,
        semantic_threshold: float = 0.65,
        embedder=None,  # SentenceTransformer or FlagModel; injected at runtime
    ) -> None:
        self.child_min_tokens = child_min_tokens
        self.child_max_tokens = child_max_tokens
        self.overlap_percent = overlap_percent
        self.semantic_threshold = semantic_threshold
        self._embedder = embedder

    def _embed_sentences(self, sentences: List[str]) -> Optional[np.ndarray]:
        """
        Embed sentences for semantic similarity computation.
        Returns (n_sentences, dim) array or None if embedder unavailable.
        """
        if self._embedder is None or not sentences:
            return None
        try:
            # FlagModel / SentenceTransformer compatible interface
            if hasattr(self._embedder, "encode"):
                vecs = self._embedder.encode(sentences, batch_size=32, show_progress_bar=False)
                return np.array(vecs, dtype=np.float32)
        except Exception as exc:
            logger.debug(f"Sentence embedding failed: {exc}; falling back to token-based split")
        return None

    def _group_sentences_into_children(
        self,
        sentences: List[str],
        embeddings: Optional[np.ndarray],
        language: str,
    ) -> List[str]:
        """
        Group sentences into child chunks using semantic similarity and token counts.

        Algorithm:
        1. Group respecting child_max_tokens and semantic_threshold.
        2. Merge chunks that fall below child_min_tokens.
        3. Apply token-based overlap while strictly enforcing child_max_tokens.
        """
        if not sentences:
            return []

        # ── Step 1: Initial grouping respecting max_tokens and semantics ──
        raw_chunks: List[List[str]] = []
        current: List[str] = []
        current_tokens = 0

        for i, sent in enumerate(sentences):
            sent_tokens = _estimate_tokens(sent, language)

            should_split = False
            if current_tokens + sent_tokens > self.child_max_tokens and current:
                should_split = True
            elif (
                embeddings is not None
                and not should_split
                and current
                and i > 0
            ):
                sim = _cosine_similarity(embeddings[i - 1], embeddings[i])
                if sim < self.semantic_threshold:
                    should_split = True

            if should_split:
                raw_chunks.append(current)
                current = [sent]
                current_tokens = sent_tokens
            else:
                current.append(sent)
                current_tokens += sent_tokens

        if current:
            raw_chunks.append(current)

        # ── Step 2: Merge undersized chunks (Issue 1) ──────────────────────
        merged_chunks: List[List[str]] = []
        for chunk in raw_chunks:
            if not merged_chunks:
                merged_chunks.append(chunk)
                continue
                
            chunk_tokens = sum(_estimate_tokens(s, language) for s in chunk)
            prev_tokens = sum(_estimate_tokens(s, language) for s in merged_chunks[-1])
            
            # Merge if either is undersized, provided we don't wildly exceed max_tokens
            if (chunk_tokens < self.child_min_tokens or prev_tokens < self.child_min_tokens) and (chunk_tokens + prev_tokens <= self.child_max_tokens + 50):
                merged_chunks[-1].extend(chunk)
            else:
                merged_chunks.append(chunk)

        # ── Step 3 & 4: Apply overlap by tokens and enforce bounds (Issues 3,4,5)
        child_texts: List[str] = []
        for idx, chunk in enumerate(merged_chunks):
            if idx == 0:
                child_texts.append(" ".join(chunk).strip())
                continue
                
            prev_chunk = merged_chunks[idx - 1]
            prev_tokens = sum(_estimate_tokens(s, language) for s in prev_chunk)
            target_overlap_tokens = max(1, int(prev_tokens * self.overlap_percent))
            
            chunk_tokens = sum(_estimate_tokens(s, language) for s in chunk)
            available_room = self.child_max_tokens - chunk_tokens
            
            overlap_sents = []
            if available_room > 0:
                overlap_tokens = 0
                for s in reversed(prev_chunk):
                    s_toks = _estimate_tokens(s, language)
                    if overlap_tokens + s_toks > target_overlap_tokens or overlap_tokens + s_toks > available_room:
                        # Ensure we grab at least 1 sentence if there's room and we haven't grabbed any
                        if not overlap_sents and s_toks <= available_room:
                            overlap_sents.insert(0, s)
                        break
                    overlap_sents.insert(0, s)
                    overlap_tokens += s_toks
                    
            combined = overlap_sents + chunk
            child_texts.append(" ".join(combined).strip())

        return child_texts

    def _chunk_passage_with_sentences(
        self,
        passage_text: str,
        sentences: List[str],
        embeddings: Optional[np.ndarray],
        language: str,
        query_id: int,
        is_selected: int,
        passage_rank: int,
        query_type: str,
    ) -> Tuple[Chunk, List[Chunk]]:
        """Core logic to construct parent and children given pre-computed embeddings."""
        parent_id = _make_parent_id(language, query_id, passage_rank)
        parent_tokens = _estimate_tokens(passage_text, language)

        parent_chunk = Chunk(
            chunk_id=parent_id,
            parent_id=parent_id,
            text=passage_text,
            chunk_type="parent",
            language=language,
            query_id=query_id,
            is_selected=is_selected,
            passage_rank=passage_rank,
            query_type=query_type,
            token_count=parent_tokens,
        )

        if len(sentences) <= 1 and parent_tokens <= self.child_max_tokens:
            child = Chunk(
                chunk_id=_make_chunk_id(language, query_id, passage_rank, 0),
                parent_id=parent_id,
                text=passage_text,
                chunk_type="child",
                language=language,
                query_id=query_id,
                is_selected=is_selected,
                passage_rank=passage_rank,
                query_type=query_type,
                token_count=parent_tokens,
            )
            return parent_chunk, [child]

        child_texts = self._group_sentences_into_children(sentences, embeddings, language)

        child_chunks: List[Chunk] = []
        for i, ctext in enumerate(child_texts):
            child_tokens = _estimate_tokens(ctext, language)
            child_chunks.append(
                Chunk(
                    chunk_id=_make_chunk_id(language, query_id, passage_rank, i),
                    parent_id=parent_id,
                    text=ctext,
                    chunk_type="child",
                    language=language,
                    query_id=query_id,
                    is_selected=is_selected,
                    passage_rank=passage_rank,
                    query_type=query_type,
                    token_count=child_tokens,
                )
            )

        return parent_chunk, child_chunks

    def chunk_passage(
        self,
        passage_text: str,
        language: str,
        query_id: int,
        is_selected: int,
        passage_rank: int,
        query_type: str,
    ) -> Tuple[Chunk, List[Chunk]]:
        """
        Chunk a single passage into one parent + N child chunks.
        Used standalone. For bulk processing, use chunk_entry instead for better GPU utilization.
        """
        sentences = _split_sentences(passage_text, language)
        parent_tokens = _estimate_tokens(passage_text, language)
        
        # Issue 2: Handle oversized single sentences by forcefully splitting them
        if len(sentences) <= 1 and parent_tokens > self.child_max_tokens:
            ratio = CHAR_TO_TOKEN.get(language, 0.35)
            chars_per_chunk = max(1, int(self.child_max_tokens / ratio))
            sentences = [passage_text[i:i+chars_per_chunk] for i in range(0, len(passage_text), chars_per_chunk)]
            
        embeddings = self._embed_sentences(sentences)
        
        return self._chunk_passage_with_sentences(
            passage_text=passage_text,
            sentences=sentences,
            embeddings=embeddings,
            language=language,
            query_id=query_id,
            is_selected=is_selected,
            passage_rank=passage_rank,
            query_type=query_type,
        )

    def chunk_entry(self, entry) -> Tuple[List[Chunk], List[Chunk]]:
        """
        Chunk all passages in a DatasetEntry. Batches embeddings for speed.

        Args:
            entry: DatasetEntry from dataset_loader.py

        Returns:
            (all_parents, all_children) — two separate lists.
        """
        all_parents: List[Chunk] = []
        all_children: List[Chunk] = []
        
        passage_sentences = []
        flat_sentences = []
        
        for passage in entry.passages:
            if not passage.passage_text.strip():
                passage_sentences.append([])
                continue
                
            sentences = _split_sentences(passage.passage_text, entry.language)
            parent_tokens = _estimate_tokens(passage.passage_text, entry.language)
            
            # Issue 2: Handle oversized single sentences
            if len(sentences) <= 1 and parent_tokens > self.child_max_tokens:
                ratio = CHAR_TO_TOKEN.get(entry.language, 0.35)
                chars_per_chunk = max(1, int(self.child_max_tokens / ratio))
                text = passage.passage_text
                sentences = [text[i:i+chars_per_chunk] for i in range(0, len(text), chars_per_chunk)]
                
            passage_sentences.append(sentences)
            flat_sentences.extend(sentences)
            
        # Issue 6: Batch embed all sentences across all passages
        flat_embeddings = self._embed_sentences(flat_sentences)
        
        emb_idx = 0
        for p_idx, passage in enumerate(entry.passages):
            if not passage.passage_text.strip():
                continue
                
            sentences = passage_sentences[p_idx]
            passage_embs = None
            if flat_embeddings is not None and sentences:
                passage_embs = flat_embeddings[emb_idx:emb_idx + len(sentences)]
                emb_idx += len(sentences)
                
            parent, children = self._chunk_passage_with_sentences(
                passage_text=passage.passage_text,
                sentences=sentences,
                embeddings=passage_embs,
                language=entry.language,
                query_id=entry.query_id,
                is_selected=passage.is_selected,
                passage_rank=passage.passage_rank,
                query_type=entry.query_type,
            )
            all_parents.append(parent)
            all_children.extend(children)
            
        return all_parents, all_children
