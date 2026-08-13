"""
orchestration/graph.py
──────────────────────
LangGraph workflow definition.

Design rationale:
- StateGraph provides a clean execution model with conditional routing.
- Nodes are pure functions that update the RAGState.
- Edges route flow based on guardrail checks.
- If ANY guardrail fails, execution routes directly to the `abstain_node`,
  skipping expensive downstream computation.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Dict, Any

from langgraph.graph import END, START, StateGraph  # type: ignore
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import get_settings
from guardrails.abstention import get_abstention_message
from guardrails.generation_guard import GenerationGuard
from guardrails.query_guard import QueryGuard
from guardrails.retrieval_guard import check_retrieval_confidence
from orchestration.state import RAGState
from retrieval.hybrid_search import HybridRetriever
from retrieval.reranker import CrossEncoderReranker
from generation.prompts import build_rag_prompt
from generation.llm_client import LLMClient
from ingestion.indexer import QdrantIndexer
from ingestion.preprocessor import preprocess_query

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def record_latency(state: RAGState, updates: dict, stage_name: str):
    """Context manager to record stage execution latency and safely update the state dict."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = (time.perf_counter() - start) * 1000.0
        lats = state.get("stage_latencies", {}).copy()
        lats[stage_name] = elapsed
        updates["stage_latencies"] = lats


class RAGOrchestrator:
    """
    Constructs and executes the LangGraph workflow.
    """

    def __init__(self):
        self.settings = get_settings()
        
        # Initialize components
        self.query_guard = QueryGuard(
            topic_model_name=self.settings.guardrails.topic_classifier_model,
            safety_model_name=self.settings.guardrails.safety_model,
            valid_topics=self.settings.guardrails.valid_query_types,
            topic_threshold=self.settings.guardrails.topic_confidence_min,
            toxicity_threshold=self.settings.guardrails.toxicity_threshold,
            device=self.settings.embedding.device,
        )
        
        self.retriever = HybridRetriever(
            url=self.settings.qdrant.url,
            api_key=self.settings.qdrant.api_key,
            collection_name=self.settings.qdrant.collection_name,
            top_k=self.settings.reranker.top_k_input,
            hnsw_ef=self.settings.qdrant.hnsw_ef_query,
            # Embedder initialized internally by HybridRetriever
        )
        
        self.reranker = CrossEncoderReranker(
            model_name=self.settings.reranker.model_name,
            device=self.settings.embedding.device,
        )
        
        self.indexer = QdrantIndexer(
            url=self.settings.qdrant.url,
            api_key=self.settings.qdrant.api_key,
            collection_name=self.settings.qdrant.collection_name,
        )
        
        self.llm_client = LLMClient(
            base_url=self.settings.llm.base_url,
            model=self.settings.llm.model,
            max_tokens=self.settings.llm.max_tokens,
            temperature=self.settings.llm.temperature,
        )
        
        self.generation_guard = GenerationGuard(
            model_name=self.settings.guardrails.nli_model,
            threshold=self.settings.guardrails.grounding_threshold,
            device=self.settings.embedding.device,
        )
        
        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self):
        """Compile the StateGraph."""
        builder = StateGraph(RAGState)

        # ── Nodes ──────────────────────────────────────────────────────────
        builder.add_node("check_query", self.node_check_query)
        builder.add_node("retrieve", self.node_retrieve)
        builder.add_node("rerank", self.node_rerank)
        builder.add_node("generate", self.node_generate)
        builder.add_node("abstain", self.node_abstain)

        # ── Edges ──────────────────────────────────────────────────────────
        builder.add_edge(START, "check_query")

        # After check_query: route to abstain if unsafe/off-topic, else retrieve
        def route_after_query(state: RAGState) -> str:
            if state.get("should_abstain"):
                return "abstain"
            return "retrieve"

        builder.add_conditional_edges(
            "check_query",
            route_after_query,
            {"abstain": "abstain", "retrieve": "retrieve"}
        )

        builder.add_edge("retrieve", "rerank")

        # After rerank: route to abstain if low confidence, else generate
        def route_after_rerank(state: RAGState) -> str:
            if state.get("should_abstain"):
                return "abstain"
            return "generate"

        builder.add_conditional_edges(
            "rerank",
            route_after_rerank,
            {"abstain": "abstain", "generate": "generate"}
        )
        
        # After generate: route to abstain if ungrounded, else END
        def route_after_generate(state: RAGState) -> str:
            if state.get("should_abstain"):
                return "abstain"
            return "end"

        builder.add_conditional_edges(
            "generate",
            route_after_generate,
            {"abstain": "abstain", "end": END}
        )
        
        builder.add_edge("abstain", END)

        return builder.compile()

    # ── Retries (Async Wrappers for external calls) ────────────────────────
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=4))
    async def _retrieve_with_retry(self, query: str, language: str):
        # Qdrant network call: wrap in to_thread to avoid blocking event loop
        return await asyncio.to_thread(self.retriever.search, query=query, language=language)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=4))
    async def _rerank_with_retry(self, query: str, retrieved_chunks: list, top_k: int):
        # Heavy GPU inference: wrap in to_thread
        return await asyncio.to_thread(self.reranker.rerank, query=query, retrieved_chunks=retrieved_chunks, top_k=top_k)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=4))
    async def _generate_with_retry(self, prompt: str):
        # vLLM network call
        return await self.llm_client.generate(prompt)

    # ── Node Functions ─────────────────────────────────────────────────────

    async def node_check_query(self, state: RAGState) -> Dict[str, Any]:
        """Node: Run Layer 1 Guardrails (Safety & Topic)."""
        logger.debug(f"[{state['request_id']}] Node: check_query")
        updates: Dict[str, Any] = {}
        
        with record_latency(state, updates, "check_query"):
            try:
                # Wrap CPU-bound classification in to_thread
                result = await asyncio.to_thread(self.query_guard.validate_query, state["query"])
                
                updates.update({
                    "query_is_safe": result["is_safe"],
                    "query_is_on_topic": result["is_on_topic"],
                    "query_topic": result["topic"],
                })
                
                if not result["passed"]:
                    updates.update({
                        "should_abstain": True,
                        "abstention_reason": result["reason"]
                    })
                    
            except Exception as exc:
                logger.error(f"check_query failed: {exc}")
                updates.update({"error": str(exc), "should_abstain": True, "abstention_reason": "internal_error"})
                
        return updates

    async def node_retrieve(self, state: RAGState) -> Dict[str, Any]:
        """Node: Execute Qdrant hybrid search."""
        logger.debug(f"[{state['request_id']}] Node: retrieve")
        updates: Dict[str, Any] = {}
        
        with record_latency(state, updates, "retrieve"):
            try:
                chunks = await self._retrieve_with_retry(state["query"], state["language"])
                updates["retrieved_chunks"] = chunks
            except Exception as exc:
                logger.error(f"retrieve failed: {exc}")
                updates.update({"error": str(exc), "should_abstain": True, "abstention_reason": "retrieval_failed"})
                
        return updates

    async def node_rerank(self, state: RAGState) -> Dict[str, Any]:
        """Node: Cross-encoder reranking and Layer 2 Guardrail."""
        logger.debug(f"[{state['request_id']}] Node: rerank")
        updates: Dict[str, Any] = {}
        
        with record_latency(state, updates, "rerank"):
            try:
                chunks = state.get("retrieved_chunks", [])
                
                # Rerank with retry
                reranked = await self._rerank_with_retry(
                    query=state["query"],
                    retrieved_chunks=chunks,
                    top_k=self.settings.reranker.top_k_output,
                )
                
                # Check confidence (Layer 2)
                conf = await asyncio.to_thread(
                    check_retrieval_confidence,
                    reranked_chunks=reranked,
                    confidence_threshold=self.settings.reranker.confidence_threshold,
                )
                
                updates.update({
                    "reranked_chunks": reranked,
                    "retrieval_confidence_passed": conf["passed"],
                })
                
                if not conf["passed"]:
                    updates.update({"should_abstain": True, "abstention_reason": conf["reason"]})
                else:
                    # Expand to parents if we have confidence (network call wrapped)
                    expanded = await asyncio.to_thread(
                        self.reranker.expand_to_parents,
                        reranked_children=reranked,
                        qdrant_indexer=self.indexer,
                        top_k=self.settings.llm.context_passages,
                    )
                    updates["expanded_parents"] = expanded
                    
            except Exception as exc:
                logger.error(f"rerank failed: {exc}")
                updates.update({"error": str(exc), "should_abstain": True, "abstention_reason": "rerank_failed"})
                
        return updates

    async def node_generate(self, state: RAGState) -> Dict[str, Any]:
        """
        Node: Generate answer via vLLM and run Layer 3 Guardrail (NLI).
        Uses async execution since generation involves network I/O.
        """
        logger.debug(f"[{state['request_id']}] Node: generate")
        updates: Dict[str, Any] = {}
        
        with record_latency(state, updates, "generate"):
            try:
                prompt = build_rag_prompt(
                    query=state["query"],
                    language_code=state["language"],
                    passages=state.get("expanded_parents", []),
                )
                
                # Non-streaming generation for the graph (API handles streaming separately)
                answer = await self._generate_with_retry(prompt)
                updates["generated_answer"] = answer
                
                # Run grounding check (Layer 3)
                grounding = await asyncio.to_thread(
                    self.generation_guard.check_grounding,
                    generated_answer=answer,
                    retrieved_passages=state.get("expanded_parents", []),
                )
                
                updates.update({
                    "generation_grounded": grounding["passed"],
                    "grounding_score": grounding["grounding_score"],
                })
                
                if not grounding["passed"]:
                    updates.update({"should_abstain": True, "abstention_reason": grounding["reason"]})
                    
            except Exception as exc:
                logger.error(f"generate failed: {exc}")
                updates.update({"error": str(exc), "should_abstain": True, "abstention_reason": "generation_failed"})
                
        return updates

    async def node_abstain(self, state: RAGState) -> Dict[str, Any]:
        """Node: Generate language-specific abstention message."""
        logger.debug(f"[{state['request_id']}] Node: abstain (Reason: {state.get('abstention_reason')})")
        updates: Dict[str, Any] = {}
        
        with record_latency(state, updates, "abstain"):
            # A fast synchronous operation, but wrapped for latency consistency
            msg = get_abstention_message(state["language"])
            updates["generated_answer"] = msg
            
        return updates

    async def run(self, request_id: str, query: str, language: str) -> Dict[str, Any]:
        """Execute the workflow for a single query."""
        total_start = time.perf_counter()
        
        # Apply the same normalization applied at ingest time to guarantee the
        # query embedding lands in the same vector-space region as indexed passages.
        normalized_query = preprocess_query(query, language)

        initial_state: RAGState = {
            "request_id": request_id,
            "query": normalized_query,
            "language": language,
            
            # Defaults
            "query_is_safe": True,
            "query_is_on_topic": True,
            "query_topic": "",
            "retrieved_chunks": [],
            "reranked_chunks": [],
            "expanded_parents": [],
            "retrieval_confidence_passed": True,
            "generated_answer": "",
            "generation_grounded": True,
            "grounding_score": 1.0,
            "citations": [],
            "stage_latencies": {},
            "total_latency_ms": 0.0,
            "should_abstain": False,
            "abstention_reason": None,
            "error": None,
        }
        
        # Run graph (ainvoke for async execution)
        final_state = await self.graph.ainvoke(initial_state)
        
        # Calculate total latency outside the graph execution
        final_state["total_latency_ms"] = (time.perf_counter() - total_start) * 1000.0
        return final_state
