"""
guardrails/query_guard.py
─────────────────────────
Layer 1 Guardrails: Topic Classification and Safety Filtering.

Design rationale:
- Reject off-topic queries immediately (saves ~150ms of retrieval/generation).
- MSMARCO-XI defines query types: ENTITY, DESCRIPTION, PROCEDURE, NUMERIC, LOCATION, PERSON.
  We use zero-shot classification via `facebook/bart-large-mnli` to enforce this.
- Reject toxic queries using `unitary/toxic-bert`.
- Indic language support: To avoid loading 12 different toxicity models, we use
  zero-shot cross-lingual transfer (BART/mDeBERTa) or transliteration if needed,
  though modern multilingual MNLI models handle Indic scripts reasonably well.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class QueryGuard:
    """
    Evaluates queries for domain relevance (topic) and safety (toxicity).

    Args:
        topic_model_name: Zero-shot classification model
        safety_model_name: Toxicity detection model
        valid_topics: List of allowed topic strings
        topic_threshold: Minimum probability for a topic match
        toxicity_threshold: Maximum allowed toxicity probability
        device: "cuda" | "cpu" | "mps"
    """

    def __init__(
        self,
        topic_model_name: str = "facebook/bart-large-mnli",
        safety_model_name: str = "unitary/toxic-bert",
        valid_topics: Optional[List[str]] = None,
        topic_threshold: float = 0.3,
        toxicity_threshold: float = 0.7,
        device: str = "cuda",
    ) -> None:
        self.topic_model_name = topic_model_name
        self.safety_model_name = safety_model_name
        self.valid_topics = valid_topics or [
            "entity",
            "description",
            "procedure",
            "numeric",
            "location",
            "person",
        ]
        self.topic_threshold = topic_threshold
        self.toxicity_threshold = toxicity_threshold
        self.device = device
        self._topic_pipeline = None
        self._safety_pipeline = None

    def _load_topic_model(self) -> None:
        if self._topic_pipeline is not None:
            return
        from transformers import pipeline  # type: ignore

        logger.info(f"Loading topic classifier '{self.topic_model_name}' on {self.device}…")
        self._topic_pipeline = pipeline(
            "zero-shot-classification",
            model=self.topic_model_name,
            device=self.device,
        )

    def _load_safety_model(self) -> None:
        if self._safety_pipeline is not None:
            return
        from transformers import pipeline  # type: ignore

        logger.info(f"Loading safety filter '{self.safety_model_name}' on {self.device}…")
        self._safety_pipeline = pipeline(
            "text-classification",
            model=self.safety_model_name,
            device=self.device,
        )

    def check_topic(self, query: str) -> Tuple[bool, str, float]:
        """
        Check if query matches allowed topics.
        
        Returns:
            (is_valid, matched_topic, confidence)
        """
        self._load_topic_model()

        # Zero-shot classification
        result = self._topic_pipeline(
            query,
            candidate_labels=self.valid_topics,
            multi_label=False,
        )

        top_topic = result["labels"][0]
        top_score = result["scores"][0]

        is_valid = top_score >= self.topic_threshold
        return is_valid, top_topic, top_score

    def check_safety(self, query: str) -> Tuple[bool, float]:
        """
        Check if query contains toxic content.
        
        Returns:
            (is_safe, toxicity_score)
        """
        self._load_safety_model()

        # toxic-bert usually outputs 'toxic' or 'clean' labels
        result = self._safety_pipeline(query, top_k=None)
        
        # Flatten pipeline output format
        if isinstance(result, list) and isinstance(result[0], list):
            result = result[0]
            
        toxicity_score = 0.0
        for label_dict in result:
            label = label_dict["label"].lower()
            if label in ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]:
                toxicity_score = max(toxicity_score, label_dict["score"])

        is_safe = toxicity_score < self.toxicity_threshold
        return is_safe, toxicity_score

    def validate_query(self, query: str) -> Dict[str, any]:
        """
        Run both topic and safety checks.
        
        Returns dict with results and combined pass/fail status.
        """
        is_safe, tox_score = self.check_safety(query)
        
        # Fast-fail: don't check topic if query is toxic
        if not is_safe:
            return {
                "passed": False,
                "reason": "safety_violation",
                "is_safe": False,
                "toxicity_score": tox_score,
                "is_on_topic": False,
                "topic": "none",
                "topic_score": 0.0,
            }
            
        is_on_topic, topic, topic_score = self.check_topic(query)
        
        return {
            "passed": is_safe and is_on_topic,
            "reason": None if is_on_topic else "off_topic",
            "is_safe": is_safe,
            "toxicity_score": tox_score,
            "is_on_topic": is_on_topic,
            "topic": topic,
            "topic_score": topic_score,
        }
