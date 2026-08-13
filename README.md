# 🎙️ Voice-Enabled RAG System for Indian Languages

A production-grade **Voice-Enabled Retrieval-Augmented Generation (RAG)** system supporting **12 Indian languages** via the `ai4bharat/MSMARCO-XI` dataset.

## Architecture

```
Voice Input → Sarvam STT (Saaras V3) → Hierarchical Chunking & Retrieval (Qdrant HNSW) → Reranking (BGE) → LLM Generation (Llama 3.1) → Grounded Answer
```

## Key Features

- 🎤 **Voice-first**: Sarvam AI Saaras V3 realtime WebSocket STT
- 🌐 **12 Indian languages**: Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Odia, Assamese, Urdu
- 🔍 **Hierarchical chunking**: Parent-child chunks with semantic boundaries
- ⚡ **Sub-200ms pipeline**: Qdrant HNSW + BGE-M3 hybrid search + RRF fusion
- 🛡️ **4-layer guardrails**: Query safety → Retrieval confidence → NLI grounding → Abstention
- 📊 **Full evaluation**: P50/P70/P99 latency, MRR, Recall@K, nDCG@10

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/devangdamkondwar24-png/Voice_RAG_System.git
cd Voice_RAG_System
cp .env.example .env
# Edit .env with your SARVAM_API_KEY and other settings

# 2. Start services (Qdrant + vLLM + App)
docker compose up -d

# 3. Ingest dataset (default: 5K queries per language)
python -m scripts.ingest --languages hi,ta,te,bn,mr,gu,kn,ml,pa,or,as,ur --limit 5000

# 4. Run benchmark
python -m scripts.benchmark --output results/benchmark_report.json

# 5. Query via API
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"query": "भारत की राजधानी क्या है?", "language": "hi"}'
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `POST /query` | POST | Text query → RAG answer |
| `WS /voice` | WebSocket | Audio stream → STT → RAG answer (streaming) |
| `GET /metrics` | GET | P50/P70/P99 latency, abstention rate |
| `GET /health` | GET | All-component health check |

## Tech Stack

| Component | Technology |
|-----------|------------|
| Speech-to-Text | Sarvam AI Saaras V3 Realtime |
| Embedding | BGE-M3 (1024-dim, 100+ languages) |
| Vector DB | Qdrant (HNSW, m=16, ef_construct=100) |
| Retrieval | Hybrid Dense+BM25 with RRF |
| Reranker | BGE-Reranker-v2-m3 (cross-encoder) |
| LLM | Llama-3.1-8B-Instruct (AWQ, via vLLM) |
| Grounding | DeBERTa-v3 NLI |
| Orchestration | LangGraph StateGraph |
| API | FastAPI + WebSocket |

## Latency Budget (Post-STT)

| Stage | Target | P99 Realistic |
|-------|--------|---------------|
| Query Processing | <5ms | 3-5ms |
| BGE-M3 Embedding | <20ms | 15-25ms |
| Qdrant Retrieval (Hybrid) | <15ms | 8-15ms |
| BGE Reranking | <25ms | 20-35ms |
| Llama Generation (TTFT) | <40ms | 30-50ms |
| Guardrails (NLI) | <20ms | 15-25ms |
| **Total** | **<125ms** | **~100-160ms** |

## Evaluation Results

Run `python -m scripts.benchmark` to generate live results. Targets:
- ✅ P99 end-to-end < 200ms
- ✅ MRR > 0.7
- ✅ Abstention rate 5-15%
- ✅ Faithfulness > 0.8

## Project Structure

```
voice_rag/
├── config/          # Centralized settings
├── ingestion/       # Dataset loading, chunking, embedding, indexing
├── retrieval/       # Hybrid search + reranking
├── generation/      # vLLM client + prompt templates
├── guardrails/      # 4-layer safety + grounding system
├── orchestration/   # LangGraph state machine
├── voice/           # Sarvam STT WebSocket client
├── evaluation/      # Benchmarking + metrics
├── observability/   # Latency tracking
├── api/             # FastAPI server
└── scripts/         # CLI tools
```

## License

MIT
