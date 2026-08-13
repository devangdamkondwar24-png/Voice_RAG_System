"""
orchestration/state.py
───────────────────────
LangGraph State definition for the RAG pipeline.

Design rationale:
- StateGraph requires a TypedDict as the single source of truth passed 
  between nodes.
- Using a typed schema ensures that nodes don't accidentally overwrite 
  or expect missing keys.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class RAGState(TypedDict):
    """
    The state dictionary that flows through the LangGraph workflow.
    """
    
    # ── Input ───────────────────────────────────────────
    request_id: str
    query: str
    language: str              # ISO code (e.g. "hi")
    
    # ── Guardrail Layer 1 (Query) ───────────────────────
    query_is_safe: bool
    query_is_on_topic: bool
    query_topic: str
    
    # ── Retrieval ───────────────────────────────────────
    retrieved_chunks: List[Dict] # Qdrant payloads (child chunks)
    
    # ── Reranking & Guardrail Layer 2 ───────────────────
    reranked_chunks: List[Dict]
    expanded_parents: List[Dict] # Expanded parent context for LLM
    retrieval_confidence_passed: bool
    
    # ── Generation & Guardrail Layer 3 ──────────────────
    generated_answer: str
    generation_grounded: bool
    grounding_score: float
    
    # ── Orchestration state ─────────────────────────────
    should_abstain: bool
    abstention_reason: Optional[str]
    error: Optional[str]
