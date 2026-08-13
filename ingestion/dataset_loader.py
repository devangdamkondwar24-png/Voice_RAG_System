"""
ingestion/dataset_loader.py
────────────────────────────
MSMARCO-XI HuggingFace dataset loader.

Dataset structure (ai4bharat/MSMARCO-XI):
    Each row:
    {
        "query_id":   int,
        "query":      str,          # Translated query (target language)
        "query_type": str,          # ENTITY/DESCRIPTION/NUMERIC/LOCATION/PERSON
        "Answer":     str,          # Gold answer (translated)
        "passages": {
            "is_selected":       List[int],   # 1=relevant, 0=not
            "English_passages":  List[str],   # Original English
            "Translated_passages": List[str], # Target-language translation
        },
        "source_lang": str,         # e.g. "eng_Latn"
        "target_lang": str,         # e.g. "hin_Deva"
    }

Design rationale:
- Streaming load (datasets iterable) avoids OOM for large languages (Hindi ~500K rows).
- Validation layer catches schema drift between MSMARCO-XI versions.
- DatasetEntry is a plain dataclass — zero dependencies, easy to serialize.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Generator, List, Optional

logger = logging.getLogger(__name__)


# ── ISO 639-1 → HuggingFace config name mapping ───────────────────────────
# MSMARCO-XI uses language codes as HF config names
LANG_CODE_MAP = {
    "hi": "hi",   # Hindi
    "ta": "ta",   # Tamil
    "te": "te",   # Telugu
    "bn": "bn",   # Bengali
    "mr": "mr",   # Marathi
    "gu": "gu",   # Gujarati
    "kn": "kn",   # Kannada
    "ml": "ml",   # Malayalam
    "pa": "pa",   # Punjabi
    "or": "or",   # Odia
    "as": "as",   # Assamese
    "ur": "ur",   # Urdu
}

# Human-readable language names (for prompts and abstention messages)
LANG_NAMES = {
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "bn": "Bengali",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "or": "Odia",
    "as": "Assamese",
    "ur": "Urdu",
}


@dataclass
class PassageEntry:
    """A single passage from MSMARCO-XI with its relevance label."""

    passage_text: str       # Translated passage text
    english_text: str       # Original English passage text
    is_selected: int        # 1 = relevant to query, 0 = not
    passage_rank: int       # Index within the passage list for this query


@dataclass
class DatasetEntry:
    """
    Normalized MSMARCO-XI entry ready for chunking.

    One DatasetEntry = one query + its associated passages.
    """

    query_id: int
    query: str              # Translated query
    query_type: str         # ENTITY | DESCRIPTION | NUMERIC | LOCATION | PERSON
    gold_answer: str        # Gold answer (translated)
    language: str           # ISO 639-1 code (e.g. "hi")
    source_lang: str        # e.g. "eng_Latn"
    target_lang: str        # e.g. "hin_Deva"
    passages: List[PassageEntry] = field(default_factory=list)

    @property
    def relevant_passages(self) -> List[PassageEntry]:
        """Return only the passages marked as selected (is_selected == 1)."""
        return [p for p in self.passages if p.is_selected == 1]

    @property
    def has_relevant_passage(self) -> bool:
        return len(self.relevant_passages) > 0


def _validate_row(row: dict, lang: str) -> bool:
    """
    Validate a raw HuggingFace row has required fields.

    Returns True if valid, False if the row should be skipped.
    """
    try:
        passages = row.get("passages", {})
        if not isinstance(passages, dict):
            return False

        # Required keys
        for key in ("is_selected", "Translated_passages"):
            if key not in passages:
                logger.debug(f"[{lang}] Missing passages.{key} in query_id={row.get('query_id')}")
                return False

        n_selected = len(passages["is_selected"])
        n_trans = len(passages["Translated_passages"])
        if n_selected != n_trans:
            logger.debug(
                f"[{lang}] Length mismatch: is_selected={n_selected} vs "
                f"Translated_passages={n_trans} for query_id={row.get('query_id')}"
            )
            return False

        if not (row.get("query") or "").strip():
            return False

        return True
    except Exception:
        return False


def load_language_dataset(
    language: str,
    dataset_name: str = "ai4bharat/MSMARCO-XI",
    split: str = "train",
    limit: Optional[int] = None,
    hf_token: Optional[str] = None,
) -> Generator[DatasetEntry, None, None]:
    """
    Stream DatasetEntry objects for a single language from MSMARCO-XI.

    Args:
        language:     ISO 639-1 code (e.g. "hi")
        dataset_name: HuggingFace dataset identifier
        split:        Dataset split ("train" | "validation")
        limit:        Maximum number of entries to yield (None = all)
        hf_token:     Optional HuggingFace token for gated repos

    Yields:
        DatasetEntry objects (one per query in the dataset)
    """
    from datasets import load_dataset  # type: ignore

    config_name = LANG_CODE_MAP.get(language)
    if config_name is None:
        raise ValueError(
            f"Unsupported language: '{language}'. "
            f"Supported codes: {sorted(LANG_CODE_MAP.keys())}"
        )

    logger.info(f"Loading MSMARCO-XI [{language}] from HuggingFace (split={split})…")

    try:
        # streaming=True avoids loading the full dataset into RAM
        ds = load_dataset(
            dataset_name,
            config_name,
            split=split,
            streaming=True,
            token=hf_token or None,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load dataset '{dataset_name}' for language '{language}': {exc}"
        ) from exc

    count = 0
    skipped = 0

    for row in ds:
        if limit is not None and count >= limit:
            break

        if not _validate_row(row, language):
            skipped += 1
            continue

        try:
            passages_raw = row["passages"]
            is_selected_list = passages_raw["is_selected"]
            translated_list = passages_raw["Translated_passages"]
            # English passages may not always be present
            english_list = passages_raw.get("English_passages") or ([""] * len(translated_list))

            passages = [
                PassageEntry(
                    passage_text=translated_list[i].strip(),
                    english_text=english_list[i].strip() if i < len(english_list) and english_list[i] else "",
                    is_selected=int(is_selected_list[i]),
                    passage_rank=i,
                )
                for i in range(len(translated_list))
                if translated_list[i] and translated_list[i].strip()  # skip empty passages
            ]

            if not passages:
                skipped += 1
                continue

            entry = DatasetEntry(
                query_id=int(row.get("query_id") or 0),
                query=(row.get("query") or "").strip(),
                query_type=(row.get("query_type") or "DESCRIPTION").upper(),
                gold_answer=(row.get("Answer") or "").strip(),
                language=language,
                source_lang=row.get("source_lang") or "eng_Latn",
                target_lang=row.get("target_lang") or f"{language}_Unknown",
                passages=passages,
            )
            yield entry
            count += 1
        except Exception as exc:
            logger.debug(f"[{language}] Failed to construct entry for row: {exc}")
            skipped += 1
            continue

    logger.info(
        f"[{language}] Loaded {count} entries, skipped {skipped} malformed rows."
    )


def load_all_languages(
    languages: List[str],
    dataset_name: str = "ai4bharat/MSMARCO-XI",
    split: str = "train",
    limit_per_language: Optional[int] = None,
    hf_token: Optional[str] = None,
) -> Generator[DatasetEntry, None, None]:
    """
    Stream DatasetEntry objects for multiple languages sequentially.

    TODO: If network drops midway on a 500K stream, this currently restarts from scratch
    on the next run. Implement cursor/checkpointing based on query_id for resumability.

    Args:
        languages:          List of ISO 639-1 codes
        limit_per_language: Max entries per language (None = all)

    Yields:
        DatasetEntry objects across all languages
    """
    for lang in languages:
        try:
            yield from load_language_dataset(
                language=lang,
                dataset_name=dataset_name,
                split=split,
                limit=limit_per_language,
                hf_token=hf_token,
            )
        except Exception as exc:
            logger.error(f"Failed to load language '{lang}': {exc}. Skipping.")
            continue
