# Architecture Overview

Project Delphi is built on a layered agent architecture.  The entry point is a FastAPI application that exposes a REST endpoint; all agent logic runs server-side.

## Ingestion Layer

The ingestion layer reads source documents from the `documents/` directory, parses them by file type, and splits the resulting text into overlapping chunks using a structural chunker that respects Markdown header boundaries, paragraph boundaries, and a hard character limit.

Each chunk is embedded with BAAI/bge-small-en-v1.5, producing a 384-dimensional normalized float vector that is stored alongside the raw text in DuckDB.

## Retrieval Layer

The RAG Agent accepts a natural-language query, generates its embedding, and uses DuckDB's HNSW index (via the VSS extension) to retrieve the top-K most similar chunks.  If the VSS extension is unavailable, the agent falls back to a pure SQL cosine similarity computation using `list_cosine_similarity`.

## Orchestration Layer

The Coordinator Agent is the top-level orchestrator.  It receives the user question, classifies the intent, and dispatches to one or more specialist agents:

- **RAG Agent** — semantic document lookup
- **Calculator Agent** — safe arithmetic evaluation via asteval
- **Web Search Agent** — live web retrieval via Tavily API

### Response Synthesis

After all sub-agents respond, the Coordinator collects their outputs and synthesizes a final answer using the LLM gateway (OpenRouter / Llama 3.3 70B).

## Configuration

All secrets and tunable constants live in `config.py`, loaded from a `.env` file via `python-dotenv`. Key constants include `CHUNK_SIZE` (500), `CHUNK_OVERLAP` (50), and `TOP_K` (5).
