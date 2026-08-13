"""
guardrails/query_guard.py
─────────────────────────
Layer 1 Guardrails: Topic Classification and Safety Filtering.

Design rationale:
- Reject off-topic queries immediately (saves ~150ms of retrieval/generation).
- MSMARCO-XI defines query types: ENTITY, DESCRIPTION, PROCEDURE, NUMERIC, LOCATION, PERSON.
  We use zero-shot classification via `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` (multilingual) 
  to enforce this across all 12 Indic languages.
- Reject toxic queries using `Hate-speech-CNERG/indic-abusive-allInOne-roberta-cross-project`.
- Both checks are run concurrently via ThreadPoolExecutor to minimize the critical path latency.
"""

from __future__ import annotations

import concurrent.futures
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class QueryGuard:
    """
    Evaluates queries for domain relevance (topic) and safety (toxicity).

    Args:
        topic_model_name: Zero-shot classification model (multilingual preferred)
        safety_model_name: Toxicity detection model (multilingual preferred)
        valid_topics: List of allowed topic strings
        topic_threshold: Minimum probability for a topic match
        toxicity_threshold: Maximum allowed toxicity probability
        device: "cuda" | "cpu" | "mps"
    """

    def __init__(
        self,
        topic_model_name: str = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        safety_model_name: str = "Hate-speech-CNERG/indic-abusive-allInOne-roberta-cross-project",
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

        top_topic = result["labels"][0].upper()  # Enforce taxonomy casing (e.g. ENTITY)
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

        # pipeline usually outputs lists of dicts
        result = self._safety_pipeline(query, top_k=None)
        
        # Flatten pipeline output format
        if isinstance(result, list) and isinstance(result[0], list):
            result = result[0]
            
        toxicity_score = 0.0
        for label_dict in result:
            label = label_dict["label"].lower()
            # Catch labels from toxic-bert as well as indic-abusive models
            if label in ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate", "abusive", "hate"]:
                toxicity_score = max(toxicity_score, label_dict["score"])

        is_safe = toxicity_score < self.toxicity_threshold
        return is_safe, toxicity_score

    def validate_query(self, query: str) -> Dict[str, any]:
        """
        Run both topic and safety checks concurrently via thread pool.
        
        Returns dict with results and combined pass/fail status.
        """
        # Pre-load to prevent thread racing on initialization
        self._load_safety_model()
        self._load_topic_model()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_safety = executor.submit(self.check_safety, query)
            future_topic = executor.submit(self.check_topic, query)
            
            is_safe, tox_score = future_safety.result()
            is_on_topic, topic, topic_score = future_topic.result()
        
        # Determine failure reason
        reason = None
        if not is_safe:
            reason = "safety_violation"
        elif not is_on_topic:
            reason = "off_topic"
            
        return {
            "passed": is_safe and is_on_topic,
            "reason": reason,
            "is_safe": is_safe,
            "toxicity_score": tox_score,
            "is_on_topic": is_on_topic,
            "topic": topic if is_safe else "NONE",
            "topic_score": topic_score if is_safe else 0.0,
        }
