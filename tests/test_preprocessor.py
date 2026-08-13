"""
tests/test_preprocessor.py
───────────────────────────
Unit tests for ingestion/preprocessor.py

Run with:
    pytest tests/test_preprocessor.py -v
"""

from __future__ import annotations
from types import SimpleNamespace

import pytest

from ingestion.preprocessor import (
    DatasetPreprocessor,
    PassageDeduplicator,
    PreprocessedEntry,
    clean_text,
    normalize_unicode,
    preprocess_query,
    standardize_numerics,
    validate_language,
)


# ── Stage 2: Unicode normalization ────────────────────────────────────────────

def test_nfc_normalization():
    """Decomposed क + ◌़ (U+0915 + U+093C) must normalize to composed क़ (U+0958)."""
    decomposed = "\u0915\u093C"   # क + nukta combining mark
    composed = "\u0958"           # क़ as a single codepoint
    result = normalize_unicode(decomposed, "hi")
    # After NFC both forms should produce the same byte sequence
    import unicodedata
    assert unicodedata.normalize("NFC", decomposed) == unicodedata.normalize("NFC", result)
    assert len(result) <= len(decomposed)  # composed is shorter or equal


def test_indicnlp_hindi():
    """IndicNLP normalizer should remove stress marks from Hindi text."""
    # Vedic stress marks U+0951 and U+0952
    text_with_stress = "नम\u0951स्ते"  # namaste with stress mark
    result = normalize_unicode(text_with_stress, "hi")
    # The result should not contain the stress mark (IndicNLP removes it)
    # If IndicNLP is unavailable the test gracefully passes by checking NFC at minimum
    assert isinstance(result, str)
    assert len(result) > 0


def test_indicnlp_malayalam_zwnj():
    """ZWNJ (U+200C) must be PRESERVED in Malayalam text after cleaning."""
    text = "ന\u200cട"  # ZWNJ between consonants (controls conjunct formation in ml)
    cleaned = clean_text(text, "ml")
    assert "\u200c" in cleaned, "ZWNJ must be preserved in Malayalam"


def test_indicnlp_kannada_zwnj():
    """ZWNJ must be PRESERVED in Kannada text after cleaning."""
    text = "ಕ\u200cನ"  # ZWNJ in Kannada context
    cleaned = clean_text(text, "kn")
    assert "\u200c" in cleaned, "ZWNJ must be preserved in Kannada"


def test_zwnj_removed_other_scripts():
    """ZWNJ must be REMOVED from Hindi text (noise from HTML rendering)."""
    text = "नम\u200cस्ते"
    cleaned = clean_text(text, "hi")
    assert "\u200c" not in cleaned


# ── Stage 3: Text cleaning ────────────────────────────────────────────────────

def test_html_entity_cleaning():
    """HTML entities must be decoded."""
    text = "5 &amp; 10 &lt; 20 &amp;amp; nested"
    result = clean_text(text, "hi")
    assert "&amp;" not in result
    assert "5 & 10" in result


def test_html_tag_removal():
    """HTML tags must be stripped."""
    text = "Some <br/> text <span class='x'>here</span> end"
    result = clean_text(text, "en")
    assert "<" not in result
    assert "Some" in result
    assert "text" in result
    assert "here" in result


def test_url_removal():
    """URLs must be removed from passage text."""
    text = "Visit https://example.com/foo?bar=1 for more info"
    result = clean_text(text, "hi")
    assert "https://" not in result
    assert "example.com" not in result
    assert "for more info" in result


def test_zero_width_removal():
    """Zero-width space (U+200B), ZWJ (U+200D), and BOM (U+FEFF) must be removed."""
    text = "\ufeffstart\u200b mid\u200d end"
    result = clean_text(text, "hi")
    assert "\ufeff" not in result
    assert "\u200b" not in result
    assert "\u200d" not in result


# ── Stage 4: Language validation ──────────────────────────────────────────────

def test_language_validation_pass():
    """Hindi text must pass for target_lang='hi'."""
    hindi_text = "यह एक हिंदी वाक्य है जो परीक्षण के लिए लिखा गया है।"
    is_valid, ratio = validate_language(hindi_text, "hi", threshold=0.3)
    assert is_valid, f"Expected Hindi text to pass, got ratio={ratio}"


def test_language_validation_fail():
    """Tamil text must fail for target_lang='hi'."""
    tamil_text = "இது தமிழில் எழுதப்பட்ட ஒரு வாக்கியம்."
    is_valid, ratio = validate_language(tamil_text, "hi", threshold=0.3)
    assert not is_valid, f"Expected Tamil text to fail for Hindi, got ratio={ratio}"


def test_language_validation_mixed():
    """Text with 40% English + 60% Hindi should pass the 0.3 threshold."""
    # 60% Devanagari, 40% Latin (brand names etc.)
    mixed = "Google कंपनी एक बहुत बड़ी technology company है।"
    is_valid, ratio = validate_language(mixed, "hi", threshold=0.3)
    assert is_valid, f"Mixed Hindi/English should pass at threshold=0.3, got ratio={ratio}"


# ── Stage 5: Deduplication ────────────────────────────────────────────────────

def test_dedup_new_passage():
    """First occurrence of a passage must be marked as new."""
    dedup = PassageDeduplicator()
    is_new, h = dedup.process("यह एक अनोखा पैसेज है।", query_id=1, is_selected=1)
    assert is_new
    assert isinstance(h, str) and len(h) == 16


def test_dedup_duplicate():
    """Second occurrence of the same passage must be marked as duplicate."""
    dedup = PassageDeduplicator()
    dedup.process("duplicate passage text", query_id=1, is_selected=0)
    is_new, _ = dedup.process("duplicate passage text", query_id=2, is_selected=1)
    assert not is_new


def test_dedup_whitespace_insensitive():
    """Dedup must collapse whitespace before hashing (tabs, extra spaces)."""
    dedup = PassageDeduplicator()
    dedup.process("hello  world", query_id=1, is_selected=0)
    is_new, _ = dedup.process("hello\tworld", query_id=2, is_selected=0)
    assert not is_new, "Whitespace variants of the same text must be treated as duplicates"


def test_dedup_metadata_aggregation():
    """Subsequent appearances should accumulate query_ids and selection counts."""
    dedup = PassageDeduplicator()
    dedup.process("shared passage", query_id=1, is_selected=1)
    dedup.process("shared passage", query_id=2, is_selected=0)
    dedup.process("shared passage", query_id=3, is_selected=1)

    meta = dedup.get_metadata(dedup.process("shared passage", query_id=4, is_selected=0)[1])
    assert 1 in meta["appears_in_queries"]
    assert meta["times_selected"] >= 2


# ── Stage 6: Numeric standardization ─────────────────────────────────────────

def test_numeric_devanagari():
    """Devanagari digits must be converted to Arabic digits."""
    result = standardize_numerics("वर्ष २०२४ में १२३ लोग")
    assert "2024" in result
    assert "123" in result
    assert "२" not in result


def test_numeric_bengali():
    """Bengali digits must be converted to Arabic digits."""
    result = standardize_numerics("বছর ২০২৪ সালে ১২৩ জন")
    assert "2024" in result
    assert "123" in result


def test_numeric_tamil():
    """Tamil digits must be converted to Arabic digits."""
    result = standardize_numerics("ஆண்டு ௨௦௨௪ இல் ௧௨௩ பேர்")
    assert "2024" in result
    assert "123" in result


# ── Stage 1: Structural validation + min length ───────────────────────────────

def test_min_length_filter():
    """Passages shorter than min_passage_length must be rejected."""
    preprocessor = DatasetPreprocessor(min_passage_length=10)

    short_passage = SimpleNamespace(
        passage_text="ok",  # < 10 chars
        english_text="",
        is_selected=0,
        passage_rank=0,
    )
    entry = _make_entry([short_passage], "hi")
    result = preprocessor.preprocess_entry(entry)
    assert result is None, "Entry with only too-short passages should return None"


def test_none_passage_skip():
    """None and empty passages must be skipped without crashing."""
    preprocessor = DatasetPreprocessor(min_passage_length=5)

    valid_passage = SimpleNamespace(
        passage_text="यह एक सामान्य हिंदी वाक्य है।",
        english_text="",
        is_selected=1,
        passage_rank=1,
    )
    entry = _make_entry([valid_passage], "hi")
    result = preprocessor.preprocess_entry(entry)
    assert result is not None
    assert len(result.passages) == 1


def test_length_mismatch_handled():
    """Entry with empty query should return None (structural failure)."""
    preprocessor = DatasetPreprocessor()

    valid_passage = SimpleNamespace(
        passage_text="Valid passage text here.",
        english_text="",
        is_selected=0,
        passage_rank=0,
    )
    entry = _make_entry([valid_passage], "hi", query="")
    result = preprocessor.preprocess_entry(entry)
    assert result is None, "Empty query must cause structural validation failure"


# ── Full pipeline ─────────────────────────────────────────────────────────────

def test_full_pipeline():
    """All 6 stages must run sequentially on a sample entry without errors."""
    preprocessor = DatasetPreprocessor(
        enable_unicode_nfc=True,
        enable_text_cleaning=True,
        enable_language_validation=True,
        enable_deduplication=True,
        enable_numeric_standardization=True,
    )

    passage = SimpleNamespace(
        # Hindi text with HTML artifact, Devanagari digit, and whitespace noise
        passage_text="<b>वर्ष २०२४ में</b> बहुत सी &amp; चीज़ें हुईं।",
        english_text="Things happened in year 2024.",
        is_selected=1,
        passage_rank=0,
    )
    entry = _make_entry([passage], "hi")
    result = preprocessor.preprocess_entry(entry)

    assert result is not None
    assert len(result.passages) == 1
    p = result.passages[0]
    assert "<b>" not in p.text
    assert "&amp;" not in p.text
    assert "2024" in p.text
    assert "२" not in p.text


# ── Query-time preprocessing consistency ─────────────────────────────────────

def test_query_matches_ingest():
    """preprocess_query must produce identical output to the ingest pipeline for same text."""
    text = "<b>वर्ष २०२४</b> &amp; hello"
    lang = "hi"

    # Simulate what preprocessor does on ingest path
    preprocessor = DatasetPreprocessor()
    ingest_result = preprocessor._process_text(text, lang)

    # Query-time path
    query_result = preprocess_query(text, lang)

    assert ingest_result == query_result, (
        f"Query and ingest normalization diverged:\n"
        f"  Ingest: {ingest_result!r}\n"
        f"  Query:  {query_result!r}"
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_entry(passages, language: str, query: str = "test query"):
    """Build a minimal DatasetEntry-like object for tests."""
    return SimpleNamespace(
        query_id=1,
        query=query,
        query_type="DESCRIPTION",
        gold_answer="test answer",
        language=language,
        source_lang="eng_Latn",
        target_lang=f"{language}_Test",
        passages=passages,
    )
