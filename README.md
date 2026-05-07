# LangGraph AI Compliance Assistant

An agentic compliance research assistant that answers regulatory questions about EU financial and AI legislation (MiCA, DORA, AI Act, GDPR). It combines a retrieval-augmented generation (RAG) pipeline with a LangGraph multi-node reasoning loop to produce accurate, well-grounded compliance answers, and structured reports when required.

---

## What It Does

Given a user query about a regulation, the assistant:

1. **Decides** whether the query needs a full structured report or a direct answer
2. **Generates** a set of targeted sub-questions to investigate
3. **Retrieves** relevant regulation articles from a local PostgreSQL vector store using hybrid search (vector similarity + BM25 full-text)
4. **Answers** each sub-question grounded in retrieved articles
5. **Critiques** its own answers and loops back if quality is insufficient
6. **Synthesizes** a final report (if requested) or returns the direct answer

---

## Architecture

The project is structured around a clean ports-and-adapters (hexagonal) architecture:

```
src/
├── config/             # Regulation aliases, SPARQL endpoint, logging config
├── core/
│   ├── agent/          # LangGraph graph definition, nodes, and state
│   ├── ingestion/      # EUR-Lex fetcher, HTML parser, article chunker
│   ├── postgres_store.py       # PostgreSQL repository (articles + chunks + hybrid retrieval)
│   └── transformers_embedder.py  # Sentence-Transformers embedding adapter
├── domain/
│   ├── models/         # Pydantic models: Article, ArticleChunk, FetchResult
│   └── ports/          # Abstract interfaces: fetch, chunk, embed, store
└── pipelines/
    └── ingestion.py    # Orchestration: fetch → chunk → embed → store
```

### Ingestion Pipeline

Runs once to populate the database before the agent is used:

```
EUR-Lex SPARQL API
      │
      ▼
 EurLexFetcher       — resolves CELEX → work URI → XHTML download
      │
      ▼
 EurLexParser        — extracts articles with title/chapter context
      │
      ▼
 ArticleChunker      — splits articles into fixed-size overlapping chunks
      │
      ▼
 TransformersEmbedder — encodes chunks with BAAI/bge-m3
      │
      ▼
 PostgresStore       — persists articles + chunks with pgvector embeddings
```

### LangGraph Agent Graph

The reasoning loop is implemented as a stateful LangGraph graph:

```
          START
            │
     Need report ?
    ┌─────────────┐
    │ No          │ Yes
    │             ▼
    │     Generate questions
    │             │
    │    Do research or not ?
    │        ├── Yes ──► Do research (hybrid retrieval)
    │        │                │
    │        └──────────────► Answer
    │                         │
    │                       Critic ◄──── Good or not
    │                    ┌────┴────┐
    │                    │ bad     │ good
    │                    └──► (loop back to Generate questions)
    │                              │
    └──────────► Synthesizer ◄─────┘
                       │
                      END
```

| Node | Responsibility |
|---|---|
| **Generate questions** | Breaks the user query into focused sub-questions |
| **Do research** | Runs hybrid vector + BM25 retrieval against the regulation corpus |
| **Answer** | Answers each sub-question grounded in retrieved articles |
| **Critic** | Evaluates answer quality; triggers re-research if insufficient |
| **Synthesizer** | Assembles sub-answers into a final structured report |

---

## Supported Regulations

| Alias | Regulation | In Force |
|---|---|---|
| `mica` | Markets in Crypto-Assets (MiCA) | 2024-12-30 |
| `dora` | Digital Operational Resilience Act (DORA) | 2025-01-17 |
| `ai_act` | EU Artificial Intelligence Act | 2024-08-01 |
| `gdpr` | General Data Protection Regulation (GDPR) | 2018-05-25 |

---

## Upcoming

The following components are planned and will be implemented before project completion:

| Component | Description |
|---|---|
| **Evaluation pipeline** | Automated end-to-end evaluation of answer accuracy and retrieval quality against a labeled compliance QA dataset |
| **Monitoring pipeline** | Runtime observability — latency, retrieval hit rate, token usage, and answer quality metrics collected per request |
| **Security pipeline (Red Teaming)** | Systematic adversarial testing for prompt injection, jailbreaks, hallucination under ambiguity, and regulatory misquotation |
| **Docker deployment** | Full containerised deployment with `docker-compose` covering the API, agent, PostgreSQL + pgvector, and the ingestion worker |

---

## Setup

> **⚠️ Setup instructions will be provided once the project is complete.**

---

