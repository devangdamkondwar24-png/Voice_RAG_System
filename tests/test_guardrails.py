"""
tests/test_guardrails.py
────────────────────────
Unit tests for Layer 1 and Layer 3 Guardrails.
"""

from __future__ import annotations

import pytest

from guardrails.generation_guard import GenerationGuard, _extract_claims_with_citations
from guardrails.query_guard import QueryGuard


# ── Query Guard (Layer 1) ─────────────────────────────────────────────────────

def test_query_guard_taxonomy_mapping():
    """Verify that QueryGuard correctly outputs the MSMARCO-XI taxonomy in uppercase."""
    # We will mock the pipeline to avoid downloading models in basic tests
    guard = QueryGuard()
    
    # Mock loaders so they don't download models
    guard._load_topic_model = lambda: None
    guard._load_safety_model = lambda: None
    
    # Mock the topic pipeline
    guard._topic_pipeline = lambda q, candidate_labels, multi_label: {
        "labels": ["entity"], # The model usually returns lower case if candidate_labels is lower case
        "scores": [0.95]
    }
    
    # Mock the safety pipeline
    guard._safety_pipeline = lambda q, top_k: [{"label": "clean", "score": 0.99}]
    
    result = guard.validate_query("Who is the prime minister of India?")
    
    assert result["passed"] is True
    assert result["is_safe"] is True
    assert result["is_on_topic"] is True
    assert result["topic"] == "ENTITY", "Topic must be uppercase to match taxonomy"


def test_query_guard_safety_violation():
    """Verify that toxic queries fast-fail and return appropriate reason."""
    guard = QueryGuard()
    
    # Mock loaders
    guard._load_topic_model = lambda: None
    guard._load_safety_model = lambda: None
    
    guard._safety_pipeline = lambda q, top_k: [{"label": "toxic", "score": 0.95}]
    
    # We must mock topic pipeline because ThreadPoolExecutor runs both concurrently
    guard._topic_pipeline = lambda q, candidate_labels, multi_label: {
        "labels": ["entity"],
        "scores": [0.1]
    }
    
    result = guard.validate_query("Some incredibly toxic text here.")
    
    assert result["passed"] is False
    assert result["is_safe"] is False
    assert result["reason"] == "safety_violation"
    assert result["topic"] == "NONE"


def test_query_guard_off_topic():
    """Verify that safe but low-relevance queries fail with off_topic reason."""
    guard = QueryGuard(topic_threshold=0.8)
    
    # Mock loaders
    guard._load_topic_model = lambda: None
    guard._load_safety_model = lambda: None
    
    # Score below threshold
    guard._topic_pipeline = lambda q, candidate_labels, multi_label: {
        "labels": ["entity"],
        "scores": [0.3]
    }
    
    guard._safety_pipeline = lambda q, top_k: [{"label": "clean", "score": 0.99}]
    
    result = guard.validate_query("Translate this sentence to French.")
    
    assert result["passed"] is False
    assert result["is_safe"] is True
    assert result["is_on_topic"] is False
    assert result["reason"] == "off_topic"


# ── Generation Guard (Layer 3) ────────────────────────────────────────────────

def test_extract_claims_with_citations():
    """Verify that citations are correctly extracted and mapped to claims."""
    text = "The capital of France is Paris [Passage 1]. It has a large population [Passage 2]."
    
    claims = _extract_claims_with_citations(text)
    
    assert len(claims) == 2
    assert claims[0]["claim"] == "The capital of France is Paris"
    assert claims[0]["passage_ranks"] == [1]
    
    assert claims[1]["claim"] == "It has a large population"
    assert claims[1]["passage_ranks"] == [2]


def test_extract_multiple_citations_per_claim():
    """Verify parsing multiple citation ranks from a single sentence."""
    text = "Water freezes at 0 degrees Celsius [Passage 1] [Passage 3]."
    
    claims = _extract_claims_with_citations(text)
    assert len(claims) == 1
    assert claims[0]["passage_ranks"] == [1, 3]


def test_generation_guard_no_citations():
    """Verify that generation without claims/citations fails if there are no citations."""
    guard = GenerationGuard()
    
    # Mock loader
    guard._load_model = lambda: None
    
    # Mock pipeline since a claim IS extracted (len > 10)
    guard._pipeline = lambda pairs, batch_size: [[{"label": "neutral", "score": 0.9}]]
    
    # Use a string shorter than 10 chars so it's not parsed as a claim
    result = guard.check_grounding(
        generated_answer="Sorry.",
        retrieved_passages=[]
    )
    
    assert result["passed"] is False
    assert result["reason"] == "no_claims_extracted"
    assert len(result["citations"]) == 0


def test_generation_guard_batched_entailment():
    """Verify batched NLI entailment logic and citation formatting."""
    guard = GenerationGuard(threshold=0.5)
    
    # Mock loader
    guard._load_model = lambda: None
    
    # Mock pipeline to return entailment for both claims
    def mock_pipeline(pairs, batch_size):
        # We expect 2 pairs since there are 2 claims
        assert len(pairs) == 2
        return [
            [{"label": "entailment", "score": 0.9}],
            [{"label": "entailment", "score": 0.8}]
        ]
        
    guard._pipeline = mock_pipeline
    
    generated = "Earth is the third planet [Passage 1]. It orbits the sun [Passage 2]."
    passages = [{"text": "The Earth is the third planet from the Sun."}]
    
    result = guard.check_grounding(generated, passages)
    
    assert result["passed"] is True
    assert result["grounding_score"] == 1.0
    assert result["entailed_claims"] == 2
    assert result["total_claims"] == 2
    
    # Verify citations array formatting
    citations = result["citations"]
    assert len(citations) == 2
    assert {"passage_rank": 1, "claim": "Earth is the third planet"} in citations
    assert {"passage_rank": 2, "claim": "It orbits the sun"} in citations


def test_generation_guard_hallucination():
    """Verify that contradiction/neutral scores below threshold trigger hallucination."""
    guard = GenerationGuard(threshold=0.6)
    
    # Mock loader
    guard._load_model = lambda: None
    
    # Mock pipeline to return contradiction for first claim, entailment for second
    # Grounding score = 1/2 = 0.5 < 0.6 threshold -> Fail
    def mock_pipeline(pairs, batch_size):
        return [
            [{"label": "contradiction", "score": 0.9}],
            [{"label": "entailment", "score": 0.9}]
        ]
        
    guard._pipeline = mock_pipeline
    
    generated = "Mars is the third planet [Passage 1]. It orbits the sun [Passage 2]."
    passages = [{"text": "The Earth is the third planet from the Sun."}]
    
    result = guard.check_grounding(generated, passages)
    
    assert result["passed"] is False
    assert result["reason"] == "hallucination_detected"
    assert result["grounding_score"] == 0.5
