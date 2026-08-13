"""
guardrails/retrieval_guard.py
──────────────────────────────
Layer 2 Guardrails: Confidence Gating.

Design rationale:
- If the reranker is not confident in ANY of the retrieved passages, we should
  abstain rather than hallucinating an answer.
- Additionally, because we are using MSMARCO-XI, we have ground truth `is_selected`
  data. If our retrieval pulls up 10 passages and NONE of them were marked as 
  relevant for ANY query in the dataset, it's a strong signal we are off-topic.
"""

from __future__ import annotations

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


def check_retrieval_confidence(
    reranked_chunks: List[Dict],
    confidence_threshold: float = 0.5,
) -> Dict[str, any]:
    """
    Check if the retrieved and reranked chunks meet confidence requirements.
    
    Args:
        reranked_chunks: List of chunk payloads (with `_rerank_score`)
        confidence_threshold: Minimum acceptable rerank score
        
    Returns:
        Dict with status, reasoning, and metrics.
    """
    if not reranked_chunks:
        return {
            "passed": False,
            "reason": "no_chunks_retrieved",
            "max_score": 0.0,
            "has_relevant_match": False,
        }

    # Extract max score
    max_score = max(chunk.get("_rerank_score", 0.0) for chunk in reranked_chunks)
    
    # Check if ANY chunk in the top K was originally marked as relevant for its query
    has_relevant_match = any(chunk.get("is_selected", 0) > 0 for chunk in reranked_chunks)

    passed_score = max_score >= confidence_threshold
    
    passed = passed_score and has_relevant_match
    
    reason = None
    if not passed_score:
        reason = "low_retrieval_confidence"
    elif not has_relevant_match:
        reason = "no_relevant_passages_found"

    return {
        "passed": passed,
        "reason": reason,
        "max_score": max_score,
        "has_relevant_match": has_relevant_match,
    }
