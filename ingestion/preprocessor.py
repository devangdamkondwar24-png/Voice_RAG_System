"""
ingestion/preprocessor.py
──────────────────────────
6-stage preprocessing pipeline for MSMARCO-XI passages and queries.

Pipeline position:
    dataset_loader.py → preprocessor.py → chunking.py → embedder.py → indexer.py

Why preprocessing matters here (and why naive fixes don't work):
- Unicode variants: क़ (U+0958) vs क + ◌़ (U+0915+U+093C) look identical but
  produce different embeddings and BM25 term hashes. NFC solves byte-level mismatch;
  IndicNLP solves Indic-script-specific normalization beyond NFC.
- Deduplication: MSMARCO reuses the same passage across 5-20 queries per language.
  Without dedup, 40-60% of Qdrant vectors are identical, wasting GPU and storage.
  Duplicates are *marked* (is_duplicate=True) rather than dropped so evaluation code
  can still observe them; the indexer skips upserting them to Qdrant.
- Language validation: MT failures produce Marathi text labeled as Hindi, Bengali as
  Assamese. A language-filtered Qdrant query then returns zero or irrelevant results.
- Indic digits: BM25 cannot match १२३ against 123. Standardizing to Arabic digits
  is the correct fix; raw text is preserved in metadata for display.
"""

from __future__ import annotations

import hashlib
import html
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Tuple, runtime_checkable

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

SUPPORTED_LANGUAGES = [
    "hi", "ta", "te", "bn", "mr", "gu",
    "kn", "ml", "pa", "or", "as", "ur",
]

# Unicode block ranges for each script (start inclusive, end inclusive).
# Used for script-detection language validation.
SCRIPT_RANGES: Dict[str, Tuple[str, str]] = {
    "hi": ("\u0900", "\u097F"),   # Devanagari
    "mr": ("\u0900", "\u097F"),   # Devanagari (shared with Hindi)
    "ta": ("\u0B80", "\u0BFF"),   # Tamil
    "te": ("\u0C00", "\u0C7F"),   # Telugu
    "bn": ("\u0980", "\u09FF"),   # Bengali
    "as": ("\u0980", "\u09FF"),   # Bengali script (shared with Assamese)
    "gu": ("\u0A80", "\u0AFF"),   # Gujarati
    "kn": ("\u0C80", "\u0CFF"),   # Kannada
    "ml": ("\u0D00", "\u0D7F"),   # Malayalam
    "pa": ("\u0A00", "\u0A7F"),   # Gurmukhi
    "or": ("\u0B00", "\u0B7F"),   # Odia
    "ur": ("\u0600", "\u06FF"),   # Arabic script (used for Urdu)
}

# Full Indic-to-Arabic digit replacement table.
INDIC_DIGIT_MAP: Dict[str, str] = {
    # Devanagari (hi, mr)
    "०": "0", "१": "1", "२": "2", "३": "3", "४": "4",
    "५": "5", "६": "6", "७": "7", "८": "8", "९": "9",
    # Bengali / Assamese (bn, as)
    "০": "0", "১": "1", "২": "2", "৩": "3", "৪": "4",
    "৫": "5", "৬": "6", "৭": "7", "৮": "8", "৯": "9",
    # Tamil (ta)
    "௦": "0", "௧": "1", "௨": "2", "௩": "3", "௪": "4",
    "௫": "5", "௬": "6", "௭": "7", "௮": "8", "௯": "9",
    # Telugu (te)
    "౦": "0", "౧": "1", "౨": "2", "౩": "3", "౪": "4",
    "౫": "5", "౬": "6", "౭": "7", "౮": "8", "౯": "9",
    # Gujarati (gu)
    "૦": "0", "૧": "1", "૨": "2", "૩": "3", "૪": "4",
    "૫": "5", "૬": "6", "૭": "7", "૮": "8", "૯": "9",
    # Kannada (kn)
    "೦": "0", "೧": "1", "೨": "2", "೩": "3", "೪": "4",
    "೫": "5", "೬": "6", "೭": "7", "೮": "8", "೯": "9",
    # Malayalam (ml)
    "൦": "0", "൧": "1", "൨": "2", "൩": "3", "൪": "4",
    "൫": "5", "൬": "6", "൭": "7", "൮": "8", "൯": "9",
    # Odia (or)
    "୦": "0", "୧": "1", "୨": "2", "୩": "3", "୪": "4",
    "୫": "5", "୬": "6", "୭": "7", "୮": "8", "୯": "9",
    # Gurmukhi / Punjabi (pa)
    "੦": "0", "੧": "1", "੨": "2", "੩": "3", "੪": "4",
    "੫": "5", "੬": "6", "੭": "7", "੮": "8", "੯": "9",
    # Arabic-Indic / Eastern Arabic (ur and mixed)
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
}

# Change 1: Build a single translation table at module load — O(n) vs O(n*k) loop
_DIGIT_TRANS_TABLE = str.maketrans(INDIC_DIGIT_MAP)

# Languages where ZWNJ (U+200C) is linguistically meaningful and must be preserved.
# In Malayalam and Kannada, ZWNJ controls whether consonants form conjuncts.
ZWNJ_PRESERVE_LANGS = {"ml", "kn"}

# Compiled HTML/URL cleanup patterns (module-level to avoid re-compilation per call)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


# ── Change 6: Structural Protocol types ──────────────────────────────────────

@runtime_checkable
class PassageEntryProtocol(Protocol):
    passage_text: str
    is_selected: int
    passage_rank: int
    english_text: str


@runtime_checkable
class DatasetEntryProtocol(Protocol):
    query_id: int
    query: str
    query_type: str
    gold_answer: str
    language: str
    source_lang: str
    target_lang: str
    passages: List[PassageEntryProtocol]


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class PreprocessedPassage:
    """A single passage after preprocessing."""
    text: str               # Normalized, cleaned text (fed to chunker)
    raw_text: str           # Original text before any modification (for display)
    is_selected: int
    passage_rank: int
    passage_hash: str       # SHA-256[:16] of normalized text — dedup key
    is_duplicate: bool      # True if this passage was seen before in this run
    english_text: str = ""
    # Change 5: Enriched dedup metadata (appears_in_queries, selection_ratio)
    # Note: counts reflect state at processing time; they accumulate as more
    # entries are processed. Do a second pass for fully-accurate final counts.
    dedup_metadata: Dict = field(default_factory=dict)


@dataclass
class PreprocessedEntry:
    """A dataset entry after full preprocessing — ready for chunking.py."""
    query_id: int
    query: str              # Preprocessed query
    raw_query: str          # Original query before normalization
    query_type: str
    gold_answer: str
    language: str
    source_lang: str
    target_lang: str
    passages: List[PreprocessedPassage]
    # Change 8: Split into separate counters for clarity
    skipped_passage_count: int = 0      # None/empty/too-short passages dropped
    skipped_language_count: int = 0     # Passages rejected by language validation

    @property
    def has_relevant_passage(self) -> bool:
        return any(p.is_selected == 1 for p in self.passages)


@dataclass
class PreprocessorStats:
    """Accumulated statistics across a full preprocessing run."""
    total_entries: int = 0
    structural_failures: int = 0
    language_validation_failures: int = 0
    passages_before_dedup: int = 0
    passages_after_dedup: int = 0
    text_cleaned_count: int = 0
    numeric_standardized_count: int = 0
    final_clean_passages: int = 0
    skipped_none_passages: int = 0
    skipped_short_passages: int = 0

    def report(self) -> str:
        dedup_ratio = (
            1.0 - self.passages_after_dedup / max(1, self.passages_before_dedup)
        ) * 100
        return (
            "\n============ Preprocessing Report ============\n"
            f"Total entries processed:    {self.total_entries:>10,}\n"
            f"Structural failures:        {self.structural_failures:>10,}\n"
            f"Language validation fails:  {self.language_validation_failures:>10,}\n"
            f"None/short passages skipped:{self.skipped_none_passages + self.skipped_short_passages:>10,}\n"
            f"Passages before dedup:      {self.passages_before_dedup:>10,}\n"
            f"Passages after dedup:       {self.passages_after_dedup:>10,}\n"
            f"Dedup ratio:                {dedup_ratio:>9.1f}%\n"
            f"Text cleaning applied to:   {self.text_cleaned_count:>10,}\n"
            f"Numeric standardized:       {self.numeric_standardized_count:>10,}\n"
            f"Final clean passages:       {self.final_clean_passages:>10,}\n"
            "============================================\n"
        )


# ── Per-language IndicNLP normalizer cache ────────────────────────────────────

_INDICNLP_NORMALIZER_CACHE: Dict[str, object] = {}

def _get_indicnlp_normalizer(language: str):
    """Lazy-load and cache an IndicNLP normalizer per language."""
    if language in _INDICNLP_NORMALIZER_CACHE:
        return _INDICNLP_NORMALIZER_CACHE[language]
    try:
        from indicnlp.normalize.indic_normalize import IndicNormalizerFactory  # type: ignore
        factory = IndicNormalizerFactory()
        normalizer = factory.get_normalizer(language)
        _INDICNLP_NORMALIZER_CACHE[language] = normalizer
        return normalizer
    except Exception as exc:
        logger.debug(f"IndicNLP normalizer unavailable for '{language}': {exc}")
        _INDICNLP_NORMALIZER_CACHE[language] = None
        return None


# ── Stage functions ──────────────────────────────────────────────────────────

def normalize_unicode(text: str, language: str) -> str:
    """
    Stage 2: Two-phase Unicode normalization.

    Phase 2a: NFC (Canonical Composition) — resolves decomposed vs. composed
              variants at the Unicode codepoint level. Mandatory for all scripts.
    Phase 2b: IndicNLP script-specific normalization — handles Nukta variants,
              chandrabindu, Hamza placement, Chillu characters, etc.
    """
    # Phase 2a: NFC
    text = unicodedata.normalize("NFC", text)

    # Phase 2b: IndicNLP (best-effort; falls back gracefully if not installed)
    normalizer = _get_indicnlp_normalizer(language)
    if normalizer is not None:
        try:
            text = normalizer.normalize(text)
        except Exception as exc:
            logger.debug(f"IndicNLP normalize failed for lang={language}: {exc}")

    return text


def clean_text(text: str, language: str) -> str:
    """
    Stage 3: Remove HTML artifacts, URLs, and control characters from web-scraped text.

    ZWNJ (U+200C) is preserved for Malayalam and Kannada because it controls
    conjunct consonant formation (linguistically meaningful). For all other scripts
    it is noise from HTML renderers and removed.
    """
    # 1. Decode HTML entities (double-pass for nested encoding: &amp;amp; → &amp; → &)
    text = html.unescape(html.unescape(text))

    # 2. Strip HTML tags
    text = _HTML_TAG_RE.sub(" ", text)

    # 3. Strip URLs
    text = _URL_RE.sub("", text)

    # 4. Normalize whitespace
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)

    # 5. Remove zero-width noise characters (except ZWNJ in ml/kn — see docstring)
    if language not in ZWNJ_PRESERVE_LANGS:
        text = text.replace("\u200c", "")   # ZWNJ — noise in non-conjunct scripts
    text = text.replace("\u200b", "")       # Zero-width space — always noise
    text = text.replace("\u200d", "")       # ZWJ — noise outside emoji usage
    text = text.replace("\ufeff", "")       # BOM — always noise

    return text.strip()


def validate_language(text: str, expected_lang: str, threshold: float = 0.3) -> Tuple[bool, float]:
    """
    Stage 4: Script-detection language validation.

    Counts the fraction of alphabetic characters that fall within the expected
    Unicode block for this language. MT failures produce passages in the wrong
    script; this catches them before they pollute the Qdrant index.

    Change 2: Pure-numeric or pure-symbol passages (no alphabetic chars) are
    allowed through — they are valid content (e.g. "42", "3.14") that would
    have been incorrectly rejected by the original False, 0.0 return.

    Returns:
        (is_valid, script_ratio)
        - is_valid: True if script_ratio >= threshold, or no alpha chars exist
        - script_ratio: fraction of alpha chars in the expected script
    """
    if expected_lang not in SCRIPT_RANGES:
        # Language not in our validation map; skip check
        return True, 1.0

    lo, hi = SCRIPT_RANGES[expected_lang]
    alpha_chars = [c for c in text if c.isalpha()]

    # Change 2: allow passages with no alphabetic characters through
    if not alpha_chars:
        return True, 1.0

    script_count = sum(1 for c in alpha_chars if lo <= c <= hi)
    ratio = script_count / len(alpha_chars)
    return ratio >= threshold, ratio


def standardize_numerics(text: str) -> str:
    """
    Stage 6: Convert Indic-script digits to ASCII digits.

    Change 1: Uses str.translate with a pre-built translation table for O(n)
    single-pass replacement — ~10× faster than the previous for-loop approach
    that performed k separate str.replace scans for k digit codepoints.

    Design: we replace in the processed text only; raw_text is preserved
    for display so users always see the original form.
    """
    return text.translate(_DIGIT_TRANS_TABLE)


def _passage_hash(text: str) -> str:
    """Compute a stable 16-char SHA-256 hex digest of whitespace-collapsed text."""
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


# ── Deduplicator (stateful across entries) ────────────────────────────────────

class PassageDeduplicator:
    """
    Stage 5: Cross-query passage deduplication tracker.

    Tracks all passage text hashes seen so far in a single ingestion run.
    MSMARCO reuses passages across 5-20 queries; without tracking, the Qdrant
    index contains 40-60% identical vectors.

    Metadata enrichment: passages carry aggregated relevance signals
    (query_ids they appear in, total times they were selected) which can be
    used to strengthen is_selected confidence signals.
    """

    def __init__(self) -> None:
        self._seen: Dict[str, bool] = {}
        self._query_map: Dict[str, List[int]] = {}
        self._selection_count: Dict[str, int] = {}

    def process(
        self, passage_text: str, query_id: int, is_selected: int
    ) -> Tuple[bool, str]:
        """
        Returns (is_new, passage_hash).
        If is_new=False, the passage hash was seen before; caller should mark
        the passage as is_duplicate=True but NOT drop it from the entry.
        """
        h = _passage_hash(passage_text)

        if h in self._seen:
            self._query_map[h].append(query_id)
            self._selection_count[h] += is_selected
            return False, h

        self._seen[h] = True
        self._query_map[h] = [query_id]
        self._selection_count[h] = is_selected
        return True, h

    def get_metadata(self, passage_hash: str) -> Dict:
        """Return enriched metadata for a given passage hash."""
        query_ids = self._query_map.get(passage_hash, [])
        sel_count = self._selection_count.get(passage_hash, 0)
        return {
            "passage_hash": passage_hash,
            "appears_in_queries": query_ids,
            "times_selected": sel_count,
            "selection_ratio": sel_count / max(1, len(query_ids)),
        }

    @property
    def total_seen(self) -> int:
        return len(self._seen)


# ── Main preprocessor ─────────────────────────────────────────────────────────

class DatasetPreprocessor:
    """
    Applies 6 sequential stages to a DatasetEntry before chunking.

    Stages:
        1. Structural validation — reject malformed entries early
        2. Unicode normalization — NFC + IndicNLP script normalizer
        3. Text cleaning — HTML, URLs, zero-width chars
        4. Language validation — script-detection check (skips pure-numeric passages)
        5. Deduplication — passage-level hash dedup; marks, doesn't drop
        6. Numeric standardization — Indic digits → Arabic digits (str.translate)

    The same preprocess_query() helper applies stages 2, 3, 6 at query time
    to guarantee embedding-space consistency with indexed passages.

    Args:
        enable_unicode_nfc:          Apply NFC + IndicNLP normalization
        enable_text_cleaning:        Strip HTML/URL/ZW chars
        enable_language_validation:  Reject wrong-script passages
        language_validation_threshold: Min fraction of text in expected script
        enable_deduplication:        Cross-query passage dedup (mark, not drop)
        enable_numeric_standardization: Indic → Arabic digits
        min_passage_length:          Minimum characters for a valid passage
        preserve_raw_text:           Store original text in PreprocessedPassage.raw_text
        deduplicator:                Shared deduplicator instance (pass the same
                                     object to process multiple languages in sequence)
    """

    def __init__(
        self,
        enable_unicode_nfc: bool = True,
        enable_text_cleaning: bool = True,
        enable_language_validation: bool = True,
        language_validation_threshold: float = 0.3,
        enable_deduplication: bool = True,
        enable_numeric_standardization: bool = True,
        min_passage_length: int = 10,
        preserve_raw_text: bool = True,
        deduplicator: Optional[PassageDeduplicator] = None,
    ) -> None:
        self.enable_unicode_nfc = enable_unicode_nfc
        self.enable_text_cleaning = enable_text_cleaning
        self.enable_language_validation = enable_language_validation
        self.language_validation_threshold = language_validation_threshold
        self.enable_deduplication = enable_deduplication
        self.enable_numeric_standardization = enable_numeric_standardization
        self.min_passage_length = min_passage_length
        self.preserve_raw_text = preserve_raw_text
        self.deduplicator = deduplicator or PassageDeduplicator()
        self.stats = PreprocessorStats()

    # Change 3: Split _process_text into _normalize and _standardize_digits
    # so each stage can be counted independently in stats.

    def _normalize(self, text: str, language: str) -> str:
        """Stages 2+3: Unicode normalize + text clean."""
        if self.enable_unicode_nfc:
            text = normalize_unicode(text, language)
        if self.enable_text_cleaning:
            text = clean_text(text, language)
        return text

    def _standardize_digits(self, text: str) -> Tuple[str, bool]:
        """Stage 6: Numeric standardization. Returns (text, was_changed)."""
        if not self.enable_numeric_standardization:
            return text, False
        new_text = standardize_numerics(text)
        return new_text, new_text != text

    def preprocess_entry(self, entry: DatasetEntryProtocol) -> Optional[PreprocessedEntry]:
        """
        Apply the full 6-stage pipeline to a DatasetEntry.

        Args:
            entry: DatasetEntry from dataset_loader.py (or any DatasetEntryProtocol)

        Returns:
            PreprocessedEntry if valid, None if the entry fails structural validation.
        """
        self.stats.total_entries += 1
        lang = entry.language

        # ── Stage 1: Structural validation ───────────────────────────────────
        if not entry.passages:
            self.stats.structural_failures += 1
            logger.debug(f"[{lang}] query_id={entry.query_id}: no passages, skipping.")
            return None

        if not (entry.query or "").strip():
            self.stats.structural_failures += 1
            logger.debug(f"[{lang}] query_id={entry.query_id}: empty query, skipping.")
            return None

        # ── Preprocess query (Change 3: use split helpers) ───────────────────
        raw_query = entry.query
        processed_query = self._normalize(raw_query, lang)
        processed_query, _ = self._standardize_digits(processed_query)

        # ── Process passages ──────────────────────────────────────────────────
        processed_passages: List[PreprocessedPassage] = []
        # Change 8: separate counters for structural vs. language drops
        skipped_structural = 0
        skipped_language = 0

        for p in entry.passages:
            raw_text = p.passage_text

            # Stage 1: Filter None / too-short passages
            if not raw_text or not raw_text.strip():
                self.stats.skipped_none_passages += 1
                skipped_structural += 1
                continue

            if len(raw_text.strip()) < self.min_passage_length:
                self.stats.skipped_short_passages += 1
                skipped_structural += 1
                continue

            # Change 7: Move passages_before_dedup counter to here — before
            # language validation — so it truly counts all structurally-valid
            # passages that entered the dedup decision point.
            self.stats.passages_before_dedup += 1

            # Change 3: Stage 2+3 and Stage 6 with per-stage stat tracking
            processed_text = self._normalize(raw_text, lang)
            if processed_text != raw_text:
                self.stats.text_cleaned_count += 1

            processed_text, digits_changed = self._standardize_digits(processed_text)
            if digits_changed:
                self.stats.numeric_standardized_count += 1

            # Stage 4: Language validation
            if self.enable_language_validation and lang in SCRIPT_RANGES:
                is_valid, ratio = validate_language(
                    processed_text, lang, self.language_validation_threshold
                )
                if not is_valid:
                    self.stats.language_validation_failures += 1
                    logger.debug(
                        f"[{lang}] query_id={entry.query_id} passage_rank={p.passage_rank}: "
                        f"script ratio={ratio:.2f} < {self.language_validation_threshold}, skipping."
                    )
                    skipped_language += 1
                    continue

            # Stage 5: Deduplication — Change 4: mark, don't drop
            is_new, h = self.deduplicator.process(processed_text, entry.query_id, p.is_selected)
            if is_new:
                self.stats.passages_after_dedup += 1
            self.stats.final_clean_passages += 1

            # Change 5: Attach dedup metadata to every passage
            passage = PreprocessedPassage(
                text=processed_text,
                raw_text=raw_text if self.preserve_raw_text else "",
                is_selected=p.is_selected,
                passage_rank=p.passage_rank,
                passage_hash=h,
                is_duplicate=not is_new,
                english_text=p.english_text,
            )
            passage.dedup_metadata = self.deduplicator.get_metadata(h)
            processed_passages.append(passage)

        if not processed_passages:
            # All passages were filtered — entry is unusable
            self.stats.structural_failures += 1
            logger.debug(
                f"[{lang}] query_id={entry.query_id}: all {len(entry.passages)} "
                f"passages filtered, skipping entry."
            )
            return None

        return PreprocessedEntry(
            query_id=entry.query_id,
            query=processed_query,
            raw_query=raw_query,
            query_type=entry.query_type,
            gold_answer=self._normalize(entry.gold_answer, lang) if entry.gold_answer else "",
            language=lang,
            source_lang=entry.source_lang,
            target_lang=entry.target_lang,
            passages=processed_passages,
            skipped_passage_count=skipped_structural,
            skipped_language_count=skipped_language,
        )

    def log_stats(self) -> None:
        """Log the current preprocessing statistics at INFO level."""
        logger.info(self.stats.report())


# ── Query-time preprocessing ──────────────────────────────────────────────────

def preprocess_query(text: str, language: str) -> str:
    """
    Apply the same normalization pipeline to a user query at inference time.

    This is called in the LangGraph orchestration graph between STT output
    and the embedding node to guarantee the query vector lands in the same
    region of embedding space as the indexed passages.

    Stages applied:  NFC → IndicNLP normalize → text clean → numeric standardize
    Stages skipped:  Structural validation, deduplication (not applicable to queries)

    Args:
        text:     Raw query text from STT
        language: ISO 639-1 code (e.g. "hi")

    Returns:
        Normalized, clean query string ready for embedding.
    """
    text = normalize_unicode(text, language)
    text = clean_text(text, language)
    text = standardize_numerics(text)
    return text
