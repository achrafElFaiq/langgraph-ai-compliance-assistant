# LangGraph AI Compliance Assistant

This project is a LangGraph-based compliance assistant for EU regulations such as MiCA, DORA, the AI Act, and GDPR. It combines a local legal knowledge base, retrieval-augmented generation, and a multi-step reasoning graph to answer compliance questions from regulation text rather than from generic model memory.

In practice, it fetches regulations from EUR-Lex, parses and stores them in PostgreSQL with pgvector embeddings, retrieves the most relevant articles for a query, grounds the answer on those articles, and can run evaluation workflows to measure answer quality and retrieval quality.

---

## Setup

### 1. Prerequisites

- Python 3.12+
- [`uv`](https://github.com/astral-sh/uv)
- PostgreSQL
- `pgvector`
- An OpenRouter API key for the LLMs

Install Python dependencies:

```bash
uv sync
```

### 2. Environment variables

Create a `.env` file at the project root.

Required variables:

```bash
OPENROUTER_API_KEY=your_openrouter_key
DATABASE_URL=postgresql://localhost/compliance_db
```

Optional variables for Langfuse traceability:

```bash
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com
```

Notes:

- `OPENROUTER_API_KEY` is used by the agent LLM, critic LLM, grounder LLM, and evaluator LLM.
- `DATABASE_URL` is used by the PostgreSQL repository.
- Langfuse is optional, but recommended if you want evaluation traces and observability.

### 3. Database setup

Create the database and initialize the schema from `db/init.sql`.

Example on macOS:

```bash
createdb compliance_db
psql compliance_db -f db/init.sql
```

The schema creates:

- `articles` for canonical regulation articles
- `article_chunks` for chunked text plus `pgvector` embeddings

If `pgvector` is not already installed in your PostgreSQL instance, install it before running the SQL initialization.

### 4. Ingest the regulations

Run the ingestion pipeline to fetch EUR-Lex content, parse articles, chunk them, embed them, and store them in PostgreSQL:

```bash
uv run python main.py
```

At the moment, `main.py` runs the ingestion flow directly.

### 5. Run a query against the LangGraph agent

There is not yet a public API/UI entrypoint committed in this repository, but you can invoke the compiled graph directly from Python:

```bash
uv run python - <<'PY'
import asyncio
from dotenv import load_dotenv

load_dotenv()

from src.config.init_store import store
from src.application.agent.graph import compiled_graph

async def main():
    await store.connect()
    result = await compiled_graph.ainvoke(
        {"input_text": "We are launching a crypto token in France, do we need a license?"},
        config={"configurable": {"thread_id": "readme-demo"}},
    )
    print(result["answer"])
    await store.close()

asyncio.run(main())
PY
```

---

## What It Does

For a user question, the agent can generate retrieval questions, decide whether additional research is needed, retrieve relevant articles with hybrid search, ground the reasoning on the retrieved legal text, answer the question, critique its own answer, and synthesize a final report when needed.

Current supported regulations:

| Alias | Regulation | In Force |
|---|---|---|
| `mica` | Markets in Crypto-Assets (MiCA) | 2024-12-30 |
| `dora` | Digital Operational Resilience Act (DORA) | 2025-01-17 |
| `ai_act` | EU Artificial Intelligence Act | 2024-08-01 |
| `gdpr` | General Data Protection Regulation (GDPR) | 2018-05-25 |

---

## Evaluation

The repository includes an evaluation pipeline under `src/pipelines/evaluation.py` and datasets under `datasets/eval/`.

Run the evaluation pipeline with:

```bash
uv run python src/pipelines/evaluation.py
```

The evaluation flow uses the labeled dataset in `datasets/eval/dataset.json` and writes score outputs into the `datasets/eval/` directory, including result files for RAGAS and DeepEval-based experiments.

This is useful for measuring:

- faithfulness
- factual correctness / answer relevancy
- context recall
- context precision

---

## Traceability and Observability

Langfuse is wired into the project through `src/config/init_langfuse.py` and is currently used in the evaluation flow to capture LLM traces and callbacks. If Langfuse environment variables are configured, you can inspect evaluation runs, prompts, and model interactions with better observability.

This gives you a traceable path from:

- input question
- retrieved evidence
- model output
- evaluation result

---

## Architecture

The project follows a **ports-and-adapters (hexagonal) architecture**.

- `src/domain/ports/` defines the interfaces for fetching, chunking, embedding, storage, and evaluation.
- `src/infrastructure/` contains concrete adapters such as the EUR-Lex fetcher, PostgreSQL store, sentence-transformers embedder, and evaluation judges.
- `src/application/agent/` contains the LangGraph state, nodes, and graph orchestration.
- `src/pipelines/` contains higher-level workflows such as ingestion and evaluation.
- `src/config/` centralizes model, store, prompt, and tracing initialization.

Current high-level structure:

```text
src/
├── application/
│   └── agent/          # LangGraph nodes, state, and graph wiring
├── config/             # LLM, store, prompt, embedder, and Langfuse initialization
├── domain/
│   ├── models/         # Shared data models
│   └── ports/          # Abstract interfaces (hexagonal ports)
├── infrastructure/
│   ├── chunk/          # Text chunking adapter
│   ├── embed/          # Sentence-transformers adapter
│   ├── eval/           # Evaluation adapters (RAGAS / DeepEval)
│   ├── fetch/          # EUR-Lex fetching and parsing adapters
│   └── store/          # PostgreSQL + pgvector adapter
└── pipelines/
    ├── ingestion.py
    └── evaluation.py
```

---

## Notes

- The current committed entrypoint is ingestion-first; API routes and containerized deployment are not yet exposed as the main runtime path.
- LangGraph report generation exists in the graph flow, while API/report delivery can be layered on top later without changing the core compliance reasoning.

