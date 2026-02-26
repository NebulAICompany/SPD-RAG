# SPD-RAG: Sub-Agent Per Document Retrieval-Augmented Generation

A multi-agent RAG system that assigns a dedicated sub-agent to each document, enabling parallel, focused retrieval and a hierarchical summarization pipeline for complex multi-document question answering.

## Architecture

```
User Query
    │
    ▼
┌──────────────┐
│ Orchestrator │ ── Decomposes query into extraction tasks
└──────┬───────┘    + writes a synthesis directive
       │
       │  Fan-out (LangGraph Send API)
       ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Sub-Agent 1 │  │  Sub-Agent 2 │  │  Sub-Agent N │
│  (Doc A)     │  │  (Doc B)     │  │  (Doc N)     │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       │  Each runs an iterative retrieval loop:
       │    search → results → search → ... → finalize
       │
       └─────────┬─────────┬───────────────┘
                 │ Reduce  │
                 ▼         ▼
        ┌────────────────────────┐
        │ Recursive Summarizer   │ ── Hierarchical agglomerative
        │ (Similarity-Ordered)   │    clustering + batched synthesis
        └────────────┬───────────┘
                     │
                     ▼
              Final Response
```

### Key Components

| Component | Description |
|---|---|
| **Orchestrator** (`core/nodes.py`) | Decomposes user queries into atomic extraction tasks using structured output |
| **Sub-Agents** (`core/nodes.py`) | Each processes one document via an iterative retrieval loop (search → results → finalize) |
| **Recursive Summarizer** (`core/nodes.py`) | Merges findings using agglomerative clustering for similarity-ordered batching |
| **Hybrid Retrieval** (`retrieval/`) | Vector search (Cohere embed-v4.0 + Qdrant) combined with BM25 keyword search |
| **Reranker** (`retrieval/reranker.py`) | Cohere rerank-v4.0 for precision filtering |
| **Robust LLM Wrapper** (`utils/google_genai_robust.py`) | Automatic retries, API key rotation, and smart backoff |

## Setup

### Prerequisites

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-org>/spd-rag.git
cd spd-rag

# Install dependencies
uv sync

# Configure environment
cp .env.example .env
# Edit .env with your API keys (see .env.example for required variables)
```

### Required API Keys

| Key | Provider | Purpose |
|---|---|---|
| `GOOGLE_API_KEYS` | Google AI | Gemini LLM (orchestrator, sub-agents, synthesis) |
| `COHERE_API_KEY` | Cohere | Embeddings (embed-v4.0) and reranking (rerank-v4.0) |
| `OPENAI_API_KEY` | OpenAI | GPT judge for benchmark evaluation |

## Usage

### Running the API Server

```bash
uv run python -m backend.main
# Server starts at http://127.0.0.1:8001
```

### API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/files` | List uploaded files |
| `POST` | `/upload` | Upload and process a document |
| `POST` | `/chat` | Send a query to the SPD-RAG pipeline |

### Running Benchmarks

```bash
# Loong benchmark (multi-document QA)
uv run python benchmark/loong/loong_evaluator.py --upload-docs --limit 5

# MoNaCo benchmark (complex multi-document reasoning)
uv run python benchmark/monaco/new_monaco_evaluator.py --upload-docs

# OOLONG benchmark (long-context aggregation)
uv run python benchmark/oolong/oolong_evaluator.py --dataset 150k --mode direct_llm
```

See [`benchmark/README.md`](benchmark/README.md) for detailed benchmark usage.

## Project Structure

```
spd-rag/
├── backend/
│   ├── core/           # LangGraph agent nodes, state, prompts
│   │   ├── graph.py    # Graph definition and compilation
│   │   ├── nodes.py    # Orchestrator, sub-agent, synthesis nodes
│   │   ├── prompts.py  # System prompts for each agent role
│   │   ├── state.py    # Pydantic state schemas
│   │   └── tools/      # RAG search tools
│   ├── pipeline/       # Document upload and vectorization
│   ├── retrieval/      # Vector search, BM25, and reranking
│   ├── shared/         # Constants, logging configuration
│   └── utils/          # Robust LLM wrapper, file parsers
├── benchmark/          # Evaluation harnesses (Loong, MoNaCo, OOLONG)
├── frontend/           # React (Vite) chat interface
├── tests/              # Unit and integration tests
└── scripts/            # Utility scripts
```

## Citation

If you use SPD-RAG in your research, please cite:

```bibtex
@article{spdrag2025,
  title={SPD-RAG: Sub-Agent Per Document Retrieval-Augmented Generation},
  year={2025}
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
