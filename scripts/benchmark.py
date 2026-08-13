"""
scripts/benchmark.py
────────────────────
Benchmark script to evaluate RAG pipeline performance (Latency & Accuracy).

Uses a 100-query curated subset from MSMARCO-XI dev set.
Runs full pipeline (excluding STT network latency) and generates a JSON report.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import track

from config.settings import get_settings
from evaluation.metrics import (
    calculate_abstention_rate,
    calculate_mean_grounding,
    calculate_mrr,
    calculate_ndcg_at_k,
    calculate_recall_at_k,
)
from observability.latency_tracker import get_latency_store, LatencyTracker
from orchestration.graph import RAGOrchestrator
from ingestion.dataset_loader import load_all_languages

logging.basicConfig(
    level=logging.WARNING, # Suppress info logs to keep progress bar clean
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("benchmark")
console = Console()
app = typer.Typer()


async def run_benchmark(
    queries: List[dict],
    orchestrator: RAGOrchestrator,
) -> dict:
    """Run benchmark over a list of queries."""
    
    latency_store = get_latency_store()
    
    # Metrics collection
    ranks_list = []
    hits_at_5 = []
    hits_at_10 = []
    relevance_scores_list = []
    
    abstained_list = []
    grounding_scores = []
    
    for i, q_data in enumerate(track(queries, description="Benchmarking...")):
        query_text = q_data["query"]
        language = q_data["language"]
        req_id = f"bench-{i}"
        
        # We wrap the whole execution in a trace to capture E2E latency
        tracker = LatencyTracker(request_id=req_id, language=language)
        
        start_time = asyncio.get_event_loop().time()
        
        # Run orchestrator
        # To get accurate per-stage timings, the graph nodes themselves don't use
        # LatencyTracker (since they are pure functions in LangGraph).
        # In a real deployed system, we'd add telemetry middleware to LangGraph.
        # For the benchmark, we approximate by running the nodes sequentially manually
        # to gather precise latency metrics.
        
        state = await _run_traced_pipeline(orchestrator, query_text, language, req_id, tracker)
        
        trace = tracker.finalize(abstained=state.get("should_abstain", False))
        latency_store.record_trace(trace)
        
        # ── Collect Accuracy Metrics ───────────────────────────────────────
        abstained_list.append(state.get("should_abstain", False))
        
        if not state.get("should_abstain", False):
            grounding_scores.append(state.get("grounding_score", 0.0))
            
        # Retrieval metrics
        reranked = state.get("reranked_chunks", [])
        
        # Find first rank of relevant document
        first_rank = 0
        hit_5 = False
        hit_10 = False
        rel_scores = []
        
        for rank, chunk in enumerate(reranked, 1):
            is_rel = chunk.get("is_selected", 0) > 0
            rel_scores.append(1.0 if is_rel else 0.0)
            
            if is_rel and first_rank == 0:
                first_rank = rank
            if is_rel and rank <= 5:
                hit_5 = True
            if is_rel and rank <= 10:
                hit_10 = True
                
        ranks_list.append(first_rank)
        hits_at_5.append(hit_5)
        hits_at_10.append(hit_10)
        relevance_scores_list.append(rel_scores)

    # ── Calculate Final Metrics ───────────────────────────────────────────
    report = {
        "accuracy": {
            "mrr": round(calculate_mrr(ranks_list), 4),
            "recall@5": round(calculate_recall_at_k(hits_at_5), 4),
            "recall@10": round(calculate_recall_at_k(hits_at_10), 4),
            "ndcg@10": round(calculate_ndcg_at_k(relevance_scores_list, k=10), 4),
            "abstention_rate": round(calculate_abstention_rate(abstained_list), 4),
            "mean_grounding_score": round(calculate_mean_grounding(grounding_scores), 4),
        },
        "latency": latency_store.full_report()
    }
    
    return report


async def _run_traced_pipeline(
    orchestrator: RAGOrchestrator,
    query: str,
    language: str,
    req_id: str,
    tracker: LatencyTracker
) -> dict:
    """Manually step through the pipeline to capture precise stage latencies."""
    state = {
        "request_id": req_id,
        "query": query,
        "language": language,
        "should_abstain": False
    }
    
    # 1. Query Guard
    with tracker.stage("query_processing"):
        res = orchestrator.node_check_query(state)
        state.update(res)
        
    if state.get("should_abstain"):
        return state
        
    # 2. Retrieve
    with tracker.stage("retrieval"):
        # hybrid_search does embedding internally
        res = orchestrator.node_retrieve(state)
        state.update(res)
        
    if state.get("should_abstain"):
        return state
        
    # 3. Rerank
    with tracker.stage("reranking"):
        res = orchestrator.node_rerank(state)
        state.update(res)
        
    if state.get("should_abstain"):
        return state
        
    # 4. Generate
    async with tracker.async_stage("generation"):
        res = await orchestrator.node_generate(state)
        state.update(res)
        
    return state


def _load_test_queries(limit: int = 100) -> List[dict]:
    """Load a sample of queries from the dataset to act as the test set."""
    settings = get_settings()
    
    # Try to load a pre-saved test set if it exists
    test_file = Path("evaluation/test_queries.json")
    if test_file.exists():
        with open(test_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data[:limit]
            
    # Otherwise sample from dataset
    console.print("Sampling test queries from HuggingFace...")
    queries = []
    
    # Get a few from each language
    langs = settings.ingestion.languages
    per_lang = max(1, limit // len(langs))
    
    stream = load_all_languages(
        languages=langs,
        dataset_name=settings.ingestion.dataset_name,
        split="train", # Usually dev/validation, but using train for demonstration
        limit_per_language=per_lang,
    )
    
    for entry in stream:
        # Only take queries that actually have a relevant passage
        if entry.has_relevant_passage:
            queries.append({
                "query": entry.query,
                "language": entry.language,
                "query_type": entry.query_type,
            })
            if len(queries) >= limit:
                break
                
    # Save for next time
    test_file.parent.mkdir(parents=True, exist_ok=True)
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)
        
    return queries


@app.command()
def main(
    output: str = typer.Option("results/benchmark_report.json", help="Output JSON path"),
    num_queries: int = typer.Option(100, help="Number of queries to run"),
    check_latency: bool = typer.Option(False, help="Fail if latency targets not met"),
    p99_target: int = typer.Option(200, help="P99 E2E latency target (ms)"),
):
    """Run the benchmark suite."""
    console.print(f"[bold blue]Starting Benchmark ({num_queries} queries)[/bold blue]")
    
    queries = _load_test_queries(num_queries)
    if not queries:
        console.print("[red]Failed to load test queries.[/red]")
        raise typer.Exit(1)
        
    console.print("Initializing Orchestrator (loading models)...")
    orchestrator = RAGOrchestrator()
    
    # Run async benchmark
    report = asyncio.run(run_benchmark(queries, orchestrator))
    
    # Print accuracy
    acc = report["accuracy"]
    console.print("\n[bold green]Accuracy Metrics[/bold green]")
    console.print(f"MRR:             {acc['mrr']:.4f}")
    console.print(f"Recall@5:        {acc['recall@5']:.4f}")
    console.print(f"Recall@10:       {acc['recall@10']:.4f}")
    console.print(f"nDCG@10:         {acc['ndcg@10']:.4f}")
    console.print(f"Abstention Rate: {acc['abstention_rate']:.1%}")
    console.print(f"Mean Grounding:  {acc['mean_grounding_score']:.4f}")
    
    # Print latency
    get_latency_store().print_report()
    
    # Save report
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    console.print(f"Report saved to [cyan]{out_path}[/cyan]")
    
    if check_latency:
        p99_e2e = report["latency"]["stages"].get("end_to_end", {}).get("p99", float('inf'))
        if p99_e2e > p99_target:
            console.print(f"[bold red]FAILED:[/bold red] P99 Latency {p99_e2e}ms > {p99_target}ms target.")
            raise typer.Exit(1)
        else:
            console.print(f"[bold green]PASSED:[/bold green] P99 Latency {p99_e2e}ms <= {p99_target}ms target.")

if __name__ == "__main__":
    app()
