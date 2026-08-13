"""
config/settings.py
──────────────────
Centralized, type-safe configuration for the Voice-Enabled RAG system.

Design rationale:
- Pydantic Settings gives us environment variable override + validation for free.
- All latency budgets, thresholds, and model names live here — one place to tune.
- Field validators enforce sanity (e.g. overlap must be 0-50%).
"""

from __future__ import annotations

from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class SarvamSettings(BaseSettings):
    """Sarvam AI Saaras V3 realtime STT settings."""

    api_key: str = Field(default="", alias="SARVAM_API_KEY")
    ws_url: str = Field(
        default="wss://api.sarvam.ai/speech-to-text-realtime/ws",
        alias="SARVAM_STT_WS_URL",
    )
    model: str = Field(default="saaras:v3-realtime", alias="SARVAM_MODEL")
    sample_rate: int = 16000
    silence_timeout_s: float = 5.0   # seconds of silence → end of utterance
    max_audio_s: float = 10.0         # maximum audio duration accepted

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class QdrantSettings(BaseSettings):
    """Qdrant vector database configuration."""

    url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    api_key: str = Field(default="", alias="QDRANT_API_KEY")
    collection_name: str = Field(
        default="msmarco_xi_rag", alias="QDRANT_COLLECTION_NAME"
    )
    # HNSW graph parameters
    hnsw_m: int = Field(default=16, alias="QDRANT_HNSW_M")
    hnsw_ef_construct: int = Field(default=100, alias="QDRANT_HNSW_EF_CONSTRUCT")
    hnsw_ef_query: int = Field(
        default=64,
        alias="QDRANT_HNSW_EF_QUERY",
        description="Lower → faster, higher → better recall. 64 hits the knee.",
    )
    # Dense vector dimension (BGE-M3 = 1024)
    dense_dim: int = 1024
    # Scalar quantization reduces RAM 4×; negligible accuracy loss for MSMARCO
    use_scalar_quantization: bool = True

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class EmbeddingSettings(BaseSettings):
    """BGE-M3 embedding model settings."""

    model_name: str = Field(default="BAAI/bge-m3", alias="EMBEDDING_MODEL")
    batch_size: int = Field(default=64, alias="EMBEDDING_BATCH_SIZE")
    device: str = Field(default="cuda", alias="EMBEDDING_DEVICE")
    # BGE-M3 natively returns dense + sparse (ColBERT optional) in one pass
    return_dense: bool = True
    return_sparse: bool = True      # For BM25-like lexical matching
    return_colbert: bool = False    # ColBERT adds latency; skip for now

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class RerankerSettings(BaseSettings):
    """BGE-Reranker-v2-m3 cross-encoder settings."""

    model_name: str = Field(
        default="BAAI/bge-reranker-v2-m3", alias="RERANKER_MODEL"
    )
    top_k_input: int = Field(
        default=50,
        alias="RERANKER_TOP_K",
        description="Candidates fed to reranker from hybrid retrieval.",
    )
    top_k_output: int = Field(
        default=10,
        alias="RERANKER_OUTPUT_K",
        description="Reranked passages kept for LLM context.",
    )
    # Below this score → trigger low-confidence guardrail
    confidence_threshold: float = Field(
        default=0.5, alias="RERANKER_CONFIDENCE_THRESHOLD"
    )

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class LLMSettings(BaseSettings):
    """vLLM-served Llama-3.1-8B-Instruct settings."""

    base_url: str = Field(default="http://localhost:8000/v1", alias="VLLM_BASE_URL")
    model: str = Field(
        default="meta-llama/Meta-Llama-3.1-8B-Instruct", alias="VLLM_MODEL"
    )
    max_tokens: int = Field(default=256, alias="VLLM_MAX_TOKENS")
    temperature: float = Field(default=0.1, alias="VLLM_TEMPERATURE")
    # Top-N reranked passages to include in the LLM context window
    context_passages: int = 5
    # Token-by-token streaming to reduce perceived latency
    stream: bool = True
    # HTTP timeout for vLLM API calls (ms)
    timeout_ms: int = 8000

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class GuardrailSettings(BaseSettings):
    """Guardrail model and threshold configuration."""

    # Layer 1 — query safety
    topic_classifier_model: str = Field(
        default="facebook/bart-large-mnli", alias="TOPIC_CLASSIFIER_MODEL"
    )
    safety_model: str = Field(
        default="unitary/toxic-bert", alias="SAFETY_MODEL"
    )
    toxicity_threshold: float = Field(default=0.7, alias="TOXICITY_THRESHOLD")
    # MSMARCO-XI query types used as valid topic labels
    valid_query_types: List[str] = [
        "ENTITY", "DESCRIPTION", "PROCEDURE",
        "NUMERIC", "LOCATION", "PERSON",
    ]
    topic_confidence_min: float = 0.3   # below this → off-topic

    # Layer 3 — generation grounding
    nli_model: str = Field(
        default="cross-encoder/nli-deberta-v3-base", alias="NLI_MODEL"
    )
    grounding_threshold: float = Field(
        default=0.6, alias="GROUNDING_THRESHOLD",
        description="Fraction of claims that must be entailed by retrieved passages."
    )

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class PreprocessingSettings(BaseSettings):
    """Controls for the 6-stage preprocessing pipeline in preprocessor.py."""

    enable_unicode_nfc: bool = True
    enable_indicnlp_normalize: bool = True
    enable_text_cleaning: bool = True
    enable_language_validation: bool = True
    # Min fraction of text that must be in the expected script to pass
    language_validation_threshold: float = Field(
        default=0.3, alias="LANG_VALIDATION_THRESHOLD"
    )
    enable_deduplication: bool = True
    enable_numeric_standardization: bool = True
    # Passages shorter than this are discarded (likely truncated MT output)
    min_passage_length: int = Field(default=10, alias="MIN_PASSAGE_LENGTH")
    # Preserve original pre-normalization text in PassageEntry.raw_text for display
    preserve_raw_text: bool = True
    supported_languages: List[str] = [
        "hi", "ta", "te", "bn", "mr", "gu",
        "kn", "ml", "pa", "or", "as", "ur",
    ]

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class ChunkingSettings(BaseSettings):
    """Hierarchical parent-child chunking configuration."""

    child_min_tokens: int = Field(default=128, alias="CHILD_CHUNK_MIN_TOKENS")
    child_max_tokens: int = Field(default=256, alias="CHILD_CHUNK_MAX_TOKENS")
    parent_min_tokens: int = Field(default=512, alias="PARENT_CHUNK_MIN_TOKENS")
    parent_max_tokens: int = Field(default=1024, alias="PARENT_CHUNK_MAX_TOKENS")
    # Overlap between adjacent child chunks to prevent context-cliff failures
    overlap_percent: float = Field(
        default=0.12,
        alias="CHUNK_OVERLAP_PERCENT",
        description="12% overlap: last ~1.5 sentences of chunk N prepended to N+1.",
    )
    # Cosine similarity threshold for semantic sentence grouping
    semantic_split_threshold: float = Field(
        default=0.65, alias="SEMANTIC_SPLIT_THRESHOLD"
    )

    @field_validator("overlap_percent")
    @classmethod
    def validate_overlap(cls, v: float) -> float:
        if not 0.0 <= v <= 0.5:
            raise ValueError("overlap_percent must be between 0.0 and 0.5")
        return v

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class IngestionSettings(BaseSettings):
    """Dataset ingestion configuration."""

    dataset_name: str = Field(
        default="ai4bharat/MSMARCO-XI", alias="DATASET_NAME"
    )
    # ISO 639-1 codes for the 12 target languages
    languages: List[str] = Field(
        default=[
            "hi", "ta", "te", "bn", "mr", "gu",
            "kn", "ml", "pa", "or", "as", "ur",
        ],
        alias="DATASET_LANGUAGES",
    )
    # Limit per language (set None for full dataset)
    ingest_limit: int | None = Field(default=5000, alias="DATASET_INGEST_LIMIT")
    # HuggingFace dataset split to use
    split: str = "train"

    @field_validator("languages", mode="before")
    @classmethod
    def parse_languages(cls, v: str | list) -> list:
        if isinstance(v, str):
            return [lang.strip() for lang in v.split(",") if lang.strip()]
        return v

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class APISettings(BaseSettings):
    """FastAPI server settings."""

    host: str = Field(default="0.0.0.0", alias="API_HOST")
    port: int = Field(default=8080, alias="API_PORT")
    workers: int = Field(default=4, alias="API_WORKERS")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class Settings(BaseSettings):
    """
    Top-level settings — composes all sub-configs.
    
    Usage:
        from config.settings import get_settings
        cfg = get_settings()
        print(cfg.preprocessing.enable_deduplication)
    """

    sarvam: SarvamSettings = SarvamSettings()
    qdrant: QdrantSettings = QdrantSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    reranker: RerankerSettings = RerankerSettings()
    llm: LLMSettings = LLMSettings()
    guardrails: GuardrailSettings = GuardrailSettings()
    chunking: ChunkingSettings = ChunkingSettings()
    preprocessing: PreprocessingSettings = PreprocessingSettings()
    ingestion: IngestionSettings = IngestionSettings()
    api: APISettings = APISettings()

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# ── Singleton ──────────────────────────────────────────────────────────────
_settings: Settings | None = None


def get_settings() -> Settings:
    """Return a cached Settings singleton (thread-safe via GIL for reads)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
