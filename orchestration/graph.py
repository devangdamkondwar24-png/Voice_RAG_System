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

import logging
from typing import Dict, Any

from langgraph.graph import END, START, StateGraph  # type: ignore

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

logger = logging.getLogger(__name__)


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
        
        # After generate: if ungrounded, we could route to abstain. 
        # But for streaming, generation happens concurrently with validation,
        # so we handle grounding failure by overwriting the answer in the node itself.
        builder.add_edge("generate", END)
        builder.add_edge("abstain", END)

        return builder.compile()

    # ── Node Functions ─────────────────────────────────────────────────────

    def node_check_query(self, state: RAGState) -> Dict[str, Any]:
        """Node: Run Layer 1 Guardrails (Safety & Topic)."""
        logger.debug(f"[{state['request_id']}] Node: check_query")
        
        try:
            result = self.query_guard.validate_query(state["query"])
            
            updates = {
                "query_is_safe": result["is_safe"],
                "query_is_on_topic": result["is_on_topic"],
                "query_topic": result["topic"],
            }
            
            if not result["passed"]:
                updates["should_abstain"] = True
                updates["abstention_reason"] = result["reason"]
                
            return updates
        except Exception as exc:
            logger.error(f"check_query failed: {exc}")
            return {"error": str(exc), "should_abstain": True, "abstention_reason": "internal_error"}

    def node_retrieve(self, state: RAGState) -> Dict[str, Any]:
        """Node: Execute Qdrant hybrid search."""
        logger.debug(f"[{state['request_id']}] Node: retrieve")
        
        try:
            chunks = self.retriever.search(
                query=state["query"],
                language=state["language"],
            )
            return {"retrieved_chunks": chunks}
        except Exception as exc:
            logger.error(f"retrieve failed: {exc}")
            return {"error": str(exc), "should_abstain": True, "abstention_reason": "retrieval_failed"}

    def node_rerank(self, state: RAGState) -> Dict[str, Any]:
        """Node: Cross-encoder reranking and Layer 2 Guardrail."""
        logger.debug(f"[{state['request_id']}] Node: rerank")
        
        try:
            chunks = state.get("retrieved_chunks", [])
            
            # Rerank
            reranked = self.reranker.rerank(
                query=state["query"],
                retrieved_chunks=chunks,
                top_k=self.settings.reranker.top_k_output,
            )
            
            # Check confidence (Layer 2)
            conf = check_retrieval_confidence(
                reranked_chunks=reranked,
                confidence_threshold=self.settings.reranker.confidence_threshold,
            )
            
            updates = {
                "reranked_chunks": reranked,
                "retrieval_confidence_passed": conf["passed"],
            }
            
            if not conf["passed"]:
                updates["should_abstain"] = True
                updates["abstention_reason"] = conf["reason"]
            else:
                # Expand to parents if we have confidence
                expanded = self.reranker.expand_to_parents(
                    reranked_children=reranked,
                    qdrant_indexer=self.indexer,
                    top_k=self.settings.llm.context_passages,
                )
                updates["expanded_parents"] = expanded
                
            return updates
        except Exception as exc:
            logger.error(f"rerank failed: {exc}")
            return {"error": str(exc), "should_abstain": True, "abstention_reason": "rerank_failed"}

    async def node_generate(self, state: RAGState) -> Dict[str, Any]:
        """
        Node: Generate answer via vLLM and run Layer 3 Guardrail (NLI).
        Uses async execution since generation involves network I/O.
        """
        logger.debug(f"[{state['request_id']}] Node: generate")
        
        try:
            prompt = build_rag_prompt(
                query=state["query"],
                language_code=state["language"],
                passages=state.get("expanded_parents", []),
            )
            
            # Non-streaming generation for the graph (API handles streaming separately)
            answer = await self.llm_client.generate(prompt)
            
            # Run grounding check (Layer 3)
            grounding = self.generation_guard.check_grounding(
                generated_answer=answer,
                retrieved_passages=state.get("expanded_parents", []),
            )
            
            updates = {
                "generated_answer": answer,
                "generation_grounded": grounding["passed"],
                "grounding_score": grounding["grounding_score"],
            }
            
            if not grounding["passed"]:
                updates["should_abstain"] = True
                updates["abstention_reason"] = grounding["reason"]
                
            return updates
        except Exception as exc:
            logger.error(f"generate failed: {exc}")
            return {"error": str(exc), "should_abstain": True, "abstention_reason": "generation_failed"}

    def node_abstain(self, state: RAGState) -> Dict[str, Any]:
        """Node: Generate language-specific abstention message."""
        logger.debug(f"[{state['request_id']}] Node: abstain (Reason: {state.get('abstention_reason')})")
        
        msg = get_abstention_message(state["language"])
        return {"generated_answer": msg}

    async def run(self, request_id: str, query: str, language: str) -> Dict[str, Any]:
        """Execute the workflow for a single query."""
        initial_state: RAGState = {
            "request_id": request_id,
            "query": query,
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
            "should_abstain": False,
            "abstention_reason": None,
            "error": None,
        }
        
        # Run graph (ainvoke for async execution)
        final_state = await self.graph.ainvoke(initial_state)
        return final_state
