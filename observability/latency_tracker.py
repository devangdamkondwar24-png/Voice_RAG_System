"""
observability/latency_tracker.py
──────────────────────────────────
Thread-safe, per-stage latency tracker with P50/P70/P99 percentile reporting.

Design rationale:
- Context-manager API keeps timing code clean: `async with tracker.stage("retrieval"):`
- All measurements stored in-memory as a deque (bounded at 10 000 samples to prevent
  unbounded growth in long-running servers).
- numpy.percentile gives O(n log n) exact percentiles — fast enough for reporting.
- LatencyTracker is request-scoped; LatencyStore is server-scoped and aggregates
  across all requests.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import AsyncIterator, Dict, Iterator, List, Optional

import numpy as np


# ── Stage names (canonical set so callers don't typo strings) ─────────────
STAGE_STT = "stt"
STAGE_QUERY_PROCESS = "query_processing"
STAGE_EMBEDDING = "embedding"
STAGE_RETRIEVAL = "retrieval"
STAGE_RERANKING = "reranking"
STAGE_GUARDRAIL_PRE = "guardrail_pre_gen"
STAGE_GENERATION = "generation"
STAGE_GUARDRAIL_POST = "guardrail_post_gen"
STAGE_END_TO_END = "end_to_end"

ALL_STAGES = [
    STAGE_STT,
    STAGE_QUERY_PROCESS,
    STAGE_EMBEDDING,
    STAGE_RETRIEVAL,
    STAGE_RERANKING,
    STAGE_GUARDRAIL_PRE,
    STAGE_GENERATION,
    STAGE_GUARDRAIL_POST,
    STAGE_END_TO_END,
]


@dataclass
class StageResult:
    """Timing result for a single stage in a single request."""

    stage: str
    duration_ms: float
    success: bool = True
    error: Optional[str] = None


@dataclass
class RequestTrace:
    """Complete latency trace for one end-to-end request."""

    request_id: str
    query_language: str
    stages: List[StageResult] = field(default_factory=list)
    abstained: bool = False
    timestamp: float = field(default_factory=time.time)

    @property
    def total_ms(self) -> float:
        """Sum of all stage durations (excludes STT streaming overlap)."""
        return sum(s.duration_ms for s in self.stages if s.stage != STAGE_STT)

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "language": self.query_language,
            "abstained": self.abstained,
            "timestamp": self.timestamp,
            "total_ms": self.total_ms,
            "stages": [
                {
                    "stage": s.stage,
                    "duration_ms": round(s.duration_ms, 2),
                    "success": s.success,
                    "error": s.error,
                }
                for s in self.stages
            ],
        }


class LatencyTracker:
    """
    Request-scoped latency tracker.

    Usage (sync):
        tracker = LatencyTracker(request_id="abc-123", language="hi")
        with tracker.stage("embedding"):
            vec = embed(query)
        trace = tracker.finalize()

    Usage (async):
        async with tracker.async_stage("retrieval"):
            results = await qdrant.search(...)
    """

    def __init__(self, request_id: str, language: str = "unknown") -> None:
        self.request_id = request_id
        self.language = language
        self._stages: List[StageResult] = []
        self._start_time: float = time.perf_counter()

    # ── Sync context manager ───────────────────────────────────────────────
    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        error: Optional[str] = None
        success = True
        try:
            yield
        except Exception as exc:
            success = False
            error = str(exc)
            raise
        finally:
            duration_ms = (time.perf_counter() - t0) * 1000
            self._stages.append(
                StageResult(stage=name, duration_ms=duration_ms, success=success, error=error)
            )

    # ── Async context manager ──────────────────────────────────────────────
    @asynccontextmanager
    async def async_stage(self, name: str) -> AsyncIterator[None]:
        t0 = time.perf_counter()
        error: Optional[str] = None
        success = True
        try:
            yield
        except Exception as exc:
            success = False
            error = str(exc)
            raise
        finally:
            duration_ms = (time.perf_counter() - t0) * 1000
            self._stages.append(
                StageResult(stage=name, duration_ms=duration_ms, success=success, error=error)
            )

    def record(self, name: str, duration_ms: float, success: bool = True) -> None:
        """Manually record a stage result (e.g., when timing spans async boundaries)."""
        self._stages.append(StageResult(stage=name, duration_ms=duration_ms, success=success))

    def finalize(self, abstained: bool = False) -> RequestTrace:
        """Seal the trace and record end-to-end duration."""
        e2e_ms = (time.perf_counter() - self._start_time) * 1000
        self._stages.append(
            StageResult(stage=STAGE_END_TO_END, duration_ms=e2e_ms)
        )
        return RequestTrace(
            request_id=self.request_id,
            query_language=self.language,
            stages=self._stages,
            abstained=abstained,
        )


class LatencyStore:
    """
    Server-scoped, thread-safe store for aggregated latency statistics.

    Keeps a bounded deque (default 10 000 samples) per stage.
    Reports P50, P70, P99 and additional stats on demand.
    """

    def __init__(self, max_samples: int = 10_000) -> None:
        self._max_samples = max_samples
        # stage_name → deque of ms values
        self._data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_samples))
        self._lock = Lock()
        self._traces: deque = deque(maxlen=max_samples)

    def record_trace(self, trace: RequestTrace) -> None:
        """Ingest a completed request trace."""
        with self._lock:
            self._traces.append(trace)
            for stage_result in trace.stages:
                self._data[stage_result.stage].append(stage_result.duration_ms)

    def percentiles(
        self,
        stage: str,
        pcts: List[float] = [50, 70, 99],
    ) -> Dict[str, float]:
        """
        Compute latency percentiles for a specific stage.
        
        Returns dict like {"p50": 12.3, "p70": 18.5, "p99": 45.1, "count": 1234}
        """
        with self._lock:
            values = list(self._data.get(stage, []))

        if not values:
            return {f"p{int(p)}": 0.0 for p in pcts} | {"count": 0, "mean": 0.0}

        arr = np.array(values, dtype=np.float64)
        result: Dict[str, float] = {}
        for p in pcts:
            result[f"p{int(p)}"] = round(float(np.percentile(arr, p)), 2)
        result["mean"] = round(float(np.mean(arr)), 2)
        result["min"] = round(float(np.min(arr)), 2)
        result["max"] = round(float(np.max(arr)), 2)
        result["count"] = len(values)
        return result

    def full_report(self) -> dict:
        """Generate a complete latency report for all stages."""
        report: dict = {"stages": {}, "summary": {}}

        for stage in ALL_STAGES:
            report["stages"][stage] = self.percentiles(stage)

        # Overall pipeline stats
        with self._lock:
            total_requests = len(self._traces)
            abstentions = sum(1 for t in self._traces if t.abstained)

        report["summary"] = {
            "total_requests": total_requests,
            "abstention_rate": round(abstentions / max(total_requests, 1), 4),
            "abstention_count": abstentions,
        }
        return report

    def save_report(self, path: str | Path) -> None:
        """Write the full report to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.full_report(), fh, indent=2)

    def print_report(self) -> None:
        """Pretty-print a concise latency summary to stdout."""
        report = self.full_report()
        print("\n" + "=" * 65)
        print("  LATENCY REPORT (ms)")
        print("=" * 65)
        header = f"{'Stage':<28} {'P50':>8} {'P70':>8} {'P99':>8} {'N':>7}"
        print(header)
        print("-" * 65)
        for stage in ALL_STAGES:
            stats = report["stages"].get(stage, {})
            if stats.get("count", 0) == 0:
                continue
            print(
                f"{stage:<28} "
                f"{stats.get('p50', 0):>8.1f} "
                f"{stats.get('p70', 0):>8.1f} "
                f"{stats.get('p99', 0):>8.1f} "
                f"{stats.get('count', 0):>7}"
            )
        print("=" * 65)
        summary = report["summary"]
        print(
            f"Total requests: {summary['total_requests']}  |  "
            f"Abstention rate: {summary['abstention_rate']:.1%}"
        )
        print("=" * 65 + "\n")


# ── Global singleton store (used by API and benchmark) ────────────────────
_store: LatencyStore | None = None


def get_latency_store() -> LatencyStore:
    """Return the global LatencyStore singleton."""
    global _store
    if _store is None:
        _store = LatencyStore()
    return _store
