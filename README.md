# SPD-RAG: Sub-Agent Per Document Retrieval-Augmented Generation

A hierarchical multi-agent framework for exhaustive cross-document question answering. SPD-RAG decomposes complex queries along the document axis, assigning a dedicated sub-agent to each document for parallel, isolated retrieval. It then merges partial answers through a token-bounded recursive synthesis layer, mitigating both incomplete evidence coverage in standard RAG and "lost-in-the-middle" degradation in long-context models.

## 🚀 Performance (Loong Benchmark)

Evaluated on the English + Set 4 (200k-250k tokens) portion of the [Loong benchmark](https://arxiv.org/abs/2406.12648) using a **GPT-5 judge**, SPD-RAG demonstrates state-of-the-art efficiency and reasoning capabilities over large document sets:

| System | Avg Score | Perfect Rate (PR) | Cost per Query |
| --- | --- | --- | --- |
| **Baseline (Full Context)** | 68.0 | 31.4% | $0.273 |
| **Normal RAG** | 33.0 | 13.7% | $0.080 |
| **Agentic RAG** | 32.8 | 8.8% | $0.098 |
| **SPD-RAG (Ours)** | **58.1** | **18.6%** | **$0.103** |

* **Accuracy:** Achieves a ~76% relative improvement (+25 absolute points) over standard RAG baselines.
* **Cost-Efficiency:** Attains 85.4% of the full-context baseline quality at only **37.9% of the API cost**, driven by offloading document-level reasoning to a lighter model (Gemini 2.5 Flash).
* **Robustness:** Successfully recovers missing evidence in dense academic papers where standard top-*K* retrieval systems score a 0% Perfect Rate.

## 🏗️ Architecture

```text
User Query
    │
    ▼
┌──────────────────────┐
│  Coordination Layer  │ ── Decomposes query into atomic tasks
│  (Gemini 2.5 Pro)    │    + writes a synthesis directive
└──────────┬───────────┘
           │
           │  Fan-out (LangGraph Send API)
           ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Sub-Agent 1 │  │  Sub-Agent 2 │  │  Sub-Agent N │  Parallel Retrieval Layer
│  (Doc A)     │  │  (Doc B)     │  │  (Doc N)     │  (Gemini 2.5 Flash)
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       │  Each runs an isolated, iterative retrieval loop:
       │  search (Qdrant + Cohere) → assess → finalize
       │
       └─────────┬─────────┬───────────────┘
                 │ Reduce  │
                 ▼         ▼
        ┌────────────────────────┐
        │    Synthesis Layer     │ ── Merges findings via recursive,
        │    (Gemini 2.5 Pro)    │    similarity-ordered agglomerative clustering
        └────────────┬───────────┘    (Budget: 750k tokens/batch)
                     │
                     ▼
              Final Response

```

### Key Components

| Component | Description |
| --- | --- |
| **Coordination Layer** (`core/nodes.py`) | Uses Gemini 2.5 Pro to decompose queries into a *Shared Instruction Set* and a *Synthesis Directive*. |
| **Parallel Retrieval Layer** (`core/nodes.py`) | Uses Gemini 2.5 Flash sub-agents. Each runs an iterative loop (max 5 searches) exclusively on its assigned document to ensure isolated, distractor-free extraction. |
| **Synthesis Layer** (`core/nodes.py`) | Aggregates outputs using precomputed cosine-distance and UPGMA agglomerative clustering. Batches and synthesizes findings up to a 750,000 token budget recursively using Gemini 2.5 Pro. |
| **Hybrid Retrieval** (`retrieval/`) | Dense vector search via Cohere `embed-v4.0` (1536-dim) stored in Qdrant. |
| **Reranker** (`retrieval/reranker.py`) | Cohere `rerank-v4.0-fast` for precision filtering (top-5). |
| **Robust LLM Wrapper** (`utils/google_genai_robust.py`) | Automatic retries, API key rotation, and smart backoff for concurrent agent execution. |

## ⚙️ Setup

### Prerequisites

* Python ≥ 3.11
* [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-org>/spd-rag.git
cd spd-rag

# Install dependencies
uv sync

# Configure environment
cp .env.example .env
# Edit .env with your API keys

```

### Required API Keys

| Key | Provider | Purpose |
| --- | --- | --- |
| `GOOGLE_API_KEYS` | Google AI | Gemini 2.5 Pro (coordinator/synthesis) & Gemini 2.5 Flash (sub-agents) |
| `COHERE_API_KEY` | Cohere | Embeddings (`embed-v4.0`) and reranking (`rerank-v4.0-fast`) |
| `OPENAI_API_KEY` | OpenAI | GPT-5 judge for benchmark evaluation |

## 💻 Usage
```bash
# Loong benchmark (multi-document QA)
uv run python benchmark/loong/loong_evaluator.py --upload-docs --limit 5
```

See [`benchmark/README.md`](https://www.google.com/search?q=benchmark/README.md) for detailed benchmark usage.

## 📁 Project Structure

```text
spd-rag/
├── backend/
│   ├── core/           # LangGraph agent nodes, state, prompts
│   │   ├── graph.py    # Graph definition and compilation
│   │   ├── nodes.py    # Coordinator, retrieval sub-agents, synthesis nodes
│   │   ├── prompts.py  # System prompts defined in paper appendix
│   │   ├── state.py    # Pydantic state schemas
│   │   └── tools/      # RAG search tools
│   ├── pipeline/       # Document upload and vectorization
│   ├── retrieval/      # Vector search and reranking
│   ├── shared/         # Constants, logging configuration
│   └── utils/          # Robust LLM wrapper, file parsers
├── benchmark/          # Evaluation harnesses (Loong, MoNaCo, OOLONG)
├── frontend/           # React (Vite) chat interface
├── tests/              # Unit and integration tests
└── scripts/            # Utility scripts

```

## 📜 Citation

If you use SPD-RAG in your research, please cite our work:

```bibtex
@article{akay2026spdrag,
  title={SPD-RAG: Sub-Agent Per Document Retrieval-Augmented Generation},
  author={Akay, Yagiz Can and Kartal, Muhammed Yusuf and Alparslan, Esra and Ortakoyluoglu, Faruk and Akpinar, Arda},
  year={2026}
}

```

## 📄 License

This project is released under the MIT License. See [LICENSE](https://www.google.com/search?q=LICENSE) for details.
