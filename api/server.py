"""
api/server.py
─────────────
FastAPI application exposing the Voice-Enabled RAG System.

Endpoints:
- POST /query  : Text query → RAG Pipeline
- WS /voice    : WebSocket for audio stream → STT → RAG Pipeline
- GET /metrics : P50/P70/P99 latency stats and abstention rate
- GET /health  : Basic liveness check
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from config.settings import get_settings
from observability.latency_tracker import get_latency_store, LatencyTracker
from orchestration.graph import RAGOrchestrator
from voice.stt_client import SarvamSTTClient

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

app = FastAPI(
    title="Voice-Enabled Indic RAG API",
    description="Multilingual RAG with Sarvam AI Voice Input and Sub-200ms Latency.",
    version="1.0.0",
)

# Global instances initialized on startup
orchestrator: RAGOrchestrator = None  # type: ignore


@app.on_event("startup")
async def startup_event():
    global orchestrator
    logger.info("Initializing RAG Orchestrator (loading models into memory)...")
    orchestrator = RAGOrchestrator()
    logger.info("RAG Orchestrator ready.")


# ── Pydantic Models ────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    language: str = "hi"


class QueryResponse(BaseModel):
    request_id: str
    query: str
    language: str
    answer: str
    abstained: bool
    abstention_reason: str | None = None
    grounding_score: float
    total_latency_ms: float


class MetricsResponse(BaseModel):
    status: str = "ok"
    metrics: dict


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Liveness probe."""
    return {"status": "ok"}


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Returns P50/P70/P99 latency statistics and abstention rate."""
    store = get_latency_store()
    return {"status": "ok", "metrics": store.full_report()}


@app.post("/query", response_model=QueryResponse)
async def process_text_query(req: QueryRequest):
    """
    Process a text query through the RAG pipeline.
    """
    req_id = str(uuid.uuid4())
    tracker = LatencyTracker(request_id=req_id, language=req.language)
    store = get_latency_store()
    
    try:
        # We use LatencyTracker here just for the end-to-end timing of the API call.
        # The benchmark script does more granular per-stage tracking.
        with tracker.stage("end_to_end_api"):
            result = await orchestrator.run(
                request_id=req_id,
                query=req.query,
                language=req.language,
            )
            
        trace = tracker.finalize(abstained=result.get("should_abstain", False))
        store.record_trace(trace)
        
        if result.get("error"):
            logger.error(f"[{req_id}] Pipeline error: {result['error']}")
            
        return QueryResponse(
            request_id=req_id,
            query=req.query,
            language=req.language,
            answer=result.get("generated_answer", ""),
            abstained=result.get("should_abstain", False),
            abstention_reason=result.get("abstention_reason"),
            grounding_score=result.get("grounding_score", 0.0),
            total_latency_ms=trace.total_ms,
        )
        
    except Exception as exc:
        logger.error(f"[{req_id}] Unhandled API exception: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.websocket("/voice")
async def voice_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for end-to-end voice RAG.
    
    Flow:
    1. Client connects via WS.
    2. Client streams PCM 16kHz audio bytes.
    3. Server proxies bytes to Sarvam Saaras V3 Realtime.
    4. Client sends empty bytes (b"") or disconnects to indicate end of speech.
    5. Sarvam returns final transcript and detected language.
    6. Server runs transcript through RAG pipeline.
    7. Server sends text answer back via WS.
    """
    await websocket.accept()
    req_id = str(uuid.uuid4())
    logger.info(f"[{req_id}] WebSocket connection established.")
    
    stt_client = SarvamSTTClient()
    
    async def audio_receiver() -> AsyncGenerator[bytes, None]:
        """Generator that yields audio bytes from the WebSocket client."""
        try:
            while True:
                data = await websocket.receive_bytes()
                if not data:
                    break
                yield data
        except WebSocketDisconnect:
            logger.info(f"[{req_id}] WebSocket client disconnected during audio stream.")
        except Exception as exc:
            logger.error(f"[{req_id}] WebSocket receive error: {exc}")

    try:
        # ── 1. Speech-to-Text ──────────────────────────────────────────────
        await websocket.send_json({"type": "status", "message": "Listening..."})
        
        # We can optionally send partials back to the client for UX
        def on_partial(text: str):
            # In a real app, you might want to async send this to the client, 
            # but we keep it simple here.
            pass
            
        transcript, language = await stt_client.transcribe_stream(
            audio_stream=audio_receiver(),
            on_partial=on_partial,
        )
        
        if not transcript:
            await websocket.send_json({
                "type": "error",
                "message": "No speech detected or transcription failed."
            })
            await websocket.close()
            return
            
        logger.info(f"[{req_id}] STT Result: '{transcript}' (lang: {language})")
        await websocket.send_json({
            "type": "transcript",
            "transcript": transcript,
            "language": language
        })
        
        # ── 2. RAG Pipeline ────────────────────────────────────────────────
        await websocket.send_json({"type": "status", "message": "Thinking..."})
        
        result = await orchestrator.run(
            request_id=req_id,
            query=transcript,
            language=language,
        )
        
        if result.get("error"):
            await websocket.send_json({
                "type": "error",
                "message": f"Pipeline error: {result['error']}"
            })
        else:
            await websocket.send_json({
                "type": "answer",
                "answer": result.get("generated_answer", ""),
                "abstained": result.get("should_abstain", False),
                "reason": result.get("abstention_reason"),
            })
            
    except Exception as exc:
        logger.error(f"[{req_id}] WebSocket exception: {exc}")
        try:
            await websocket.send_json({"type": "error", "message": "Internal Server Error"})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
        logger.info(f"[{req_id}] WebSocket connection closed.")


if __name__ == "__main__":
    settings = get_settings().api
    uvicorn.run(
        "api.server:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level=settings.log_level.lower(),
    )
