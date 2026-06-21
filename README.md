# AI Research Assistant — Project Delphi

## Overview

Project Delphi is a production-grade multi-agent AI research assistant built on the Agno AgentOS
framework. A FastAPI backend receives natural language queries and routes them through an
**Ollama-powered intent classifier** (phi3.5-mini, fully local) to one of three specialist agents:
a **RAG Agent** that performs semantic search over a private DuckDB knowledge base, a
**Calculator Agent** that evaluates mathematical expressions via the asteval safe interpreter, and
a **Web Search Agent** that fetches live results through the Tavily API. All sub-agents use
**OpenRouter (Llama 3.3 70B)** as the primary LLM with **Groq (Llama 4 Scout)** as an automatic
fallback, ensuring the system remains operational under rate limits or quota exhaustion.
The entire stack is exposed through **Agno AgentOS UI** at `https://app.agno.com`, where
every tool call, retrieved chunk, source file, and similarity score is visible in the interface.

## Architecture Diagram

![Architecture](docs/architecture.png)

---

## System Architecture

### Agent Layer

#### 1. Coordinator Agent

The Coordinator is the entry point for all queries submitted to the REST API (`POST /query`).
Its job is purely **classification and dispatch** — it never answers a question directly.

**Intent Classification (Ollama phi3.5-mini)**
Routing uses a local Ollama model (`phi3.5`) at `http://localhost:11434/api/generate`. A
carefully engineered few-shot system prompt with 10 labelled examples teaches the model to
distinguish between three intent classes based on _what the user is trying to accomplish_,
not the words they use:

| Class | When it fires |
|---|---|
| `RAG` | User wants information from uploaded documents or the private knowledge base |
| `CALCULATOR` | User wants a precise numeric result, even if described in plain language |
| `SEARCH` | User wants world knowledge, current events, or facts not in private docs |

**Keyword Fallback**
If Ollama is unreachable (service down, model not pulled), `run_coordinator()` falls back to
deterministic regex pattern matching. This is an intentional design decision: production systems
must never have a single point of failure on the routing path. The keyword fallback is
predictable and debuggable, at the cost of occasionally misclassifying intent-based queries.

**Three-Tier LLM Fallback (sub-agents)**
Each routing function wraps a full fallback chain:
1. **Tier 1 — OpenRouter** (primary): Calls the agent with `meta-llama/llama-3.3-70b-instruct`
2. **Tier 2 — Groq** (fallback): Triggered on any 429, 5xx, or connection error. Uses Agno's
   native `Groq` class (not `OpenAILike`) because Groq's tool-call schema differs from OpenAI's
3. **Tier 3 — Direct Python call** (last resort): Calls the tool function directly without an
   LLM, guaranteeing a real result is always returned even if both providers are unavailable

---

#### 2. RAG Agent

Handles all document knowledge base queries. Backed by a DuckDB vector store loaded from
the `documents/` directory.

- **Always calls `document_lookup`** before answering — the system prompt explicitly forbids
  answering from memory alone
- Returns retrieved chunks with **source file name** and **cosine similarity score** for
  every result, satisfying the case study's transparency requirement
- The embedding model (`BAAI/bge-small-en-v1.5`) runs fully locally — no external API calls
  for search; the entire retrieval pipeline is private and offline-capable

---

#### 3. Calculator Agent

Handles all mathematical queries, including those described in plain language
("three dozen eggs, use half" → `36 / 2`).

- **Always calls `safe_calculate`** — the agent is instructed never to compute results itself
- Uses **asteval** (AST-based interpreter) instead of Python `eval()`. `eval()` allows
  arbitrary code execution and is a critical security vulnerability. `asteval` compiles the
  expression to an Abstract Syntax Tree and only evaluates mathematical nodes, preventing
  injection of shell commands, file reads, or network calls
- Clearly states the expression and its numeric result in the response

---

#### 4. Web Search Agent

Handles current events, world knowledge, and factual queries not present in the private
document collection.

- **Always calls `web_search`** before answering — never relies on the LLM's training data
  for facts that can be verified live
- Uses the **Tavily API**, which is purpose-built for LLM agents and returns clean structured
  results with title, URL, and content excerpt
- **Cites all sources** (title + URL) in every response

---

### RAG Pipeline

The retrieval pipeline processes documents in `documents/` at ingestion time and stores
chunk embeddings in a persistent DuckDB database.

```
documents/          ←  PDF, Markdown, TXT files
    │
    ▼
ingest.py           ←  file type detection + text extraction
    │
    ▼ Structural / Recursive Chunking
    │   Priority order:
    │     1. Markdown headers  (#, ##, ###)
    │     2. Double newlines   (paragraph boundaries)
    │     3. Single newlines
    │     4. Hard character limit (CHUNK_SIZE=500, CHUNK_OVERLAP=50)
    │   Preserves semantic context — sentences are not split mid-thought
    │
    ▼
BAAI/bge-small-en-v1.5  ←  33M params, 384 dimensions, CPU, fully local
    │   Batch encodes all chunks; no external API call
    │
    ▼
DuckDB (./db/embeddings.db)
    │   Table: document_chunks
    │     - id, source_file, chunk_index, content, embedding FLOAT[384],
    │       timestamp
    │   HNSW index on embedding column (vss extension)
    │   Pure SQL cosine fallback: list_cosine_similarity()
    │     → system works even if vss extension fails to load
    │
    ▼  (at query time)
database.search_chunks(query, top_k=5)
    │   1. Embed query with same bge-small-en-v1.5 model
    │   2. HNSW approximate nearest-neighbour search
    │   3. Return top-K chunks with similarity scores
    │
    ▼
RAG Agent response  ←  chunks injected into context, shown explicitly
```

**Ingestion is idempotent**: chunks are upserted via `INSERT OR REPLACE` keyed on
`(source_file, chunk_index)`, so re-running `ingest.py` does not duplicate data.

---

### LLM Gateway — Three-Tier Architecture

```
Query arrives at sub-agent
        │
        ▼
[Tier 1] OpenRouter  ──────────────────────────────────────────────┐
         Model : meta-llama/llama-3.3-70b-instruct                 │
         Endpoint: https://openrouter.ai/api/v1                    │
         Client : OpenAILike (Agno) + httpx.Client(verify=False)   │
         httpx.AsyncClient(verify=False) for AgentOS async path    │
                                                                    │
         On: 429, 402, 5xx, connection error, timeout              │
         ▼                                                          │
[Tier 2] Groq  ─────────────────────────────────────────────────── │
         Model : meta-llama/llama-4-scout-17b-16e-instruct         │
         Client : agno.models.groq.Groq  (native — NOT OpenAILike) │
         Reason: Groq's function-call schema differs from OpenAI's; │
                 OpenAILike sends malformed tool calls to Groq       │
                                                                    │
         On: tool-call schema error, function name mismatch         │
         ▼                                                          │
[Tier 3] Direct Python tool call ◄──────────────────────────────── ┘
         No LLM involved — tool function called directly
         Guarantees a real result is always returned
```

**Routing uses Ollama (Tier 0)** — the classification step is completely independent of
the sub-agent LLM tiers. Ollama is local and has no rate limits. If Ollama fails, keyword
regex takes over. If OpenRouter fails, Groq takes over. If Groq fails, the tool is called
directly. There is no path through the system that returns an empty or error response.

---

### Tech Stack

| Component | Technology | Reason |
|---|---|---|
| Orchestration | Agno AgentOS 2.6.18 | Native tool calling, FastAPI backend, AgentOS UI, streaming |
| Intent Routing | Ollama phi3.5-mini (local) | No rate limits, no API key, fully offline, reliable 3-way classification |
| LLM Primary | OpenRouter Llama 3.3 70B | Top-tier open-source reasoning, structured output, wide model access |
| LLM Fallback | Groq Llama 4 Scout | Fast inference, free tier, reliable tool calling via native Agno client |
| Vector DB | DuckDB + vss extension | Embedded, in-process, SQL + vector queries, zero infrastructure |
| Embeddings | BAAI/bge-small-en-v1.5 | 33M params, 384D, fully local, no API latency, CPU-friendly |
| Web Search | Tavily API | Structured results, purpose-built for LLM agents, source citations |
| Safe Math | asteval | AST-based interpreter, prevents arbitrary code execution |
| UI | Agno AgentOS | Native streaming, tool call visibility, chunk display, session history |
| Document Parsing | pypdf + python-docx | PDF and DOCX text extraction alongside native Markdown/TXT handling |

---

## Design Decisions & Tradeoffs

### 1. Why Ollama for routing instead of an API model

Routing is the most critical path in the system — every query passes through it. Using a
cloud API (OpenRouter, Groq) for routing would mean that rate limits or quota exhaustion on
the routing layer cascade into total system failure, even if the documents and tools are
perfectly available. Ollama runs **phi3.5-mini locally**, with no network call, no API key,
and no rate limit. The model is more than sufficient for a 3-way classification task; it does
not need 70B parameters to distinguish "math" from "document lookup" from "web search".
The system remains capable of classifying and routing queries **fully offline** for the intent
classification step. The tradeoff is that Ollama must be installed on the host machine and
`phi3.5` must be pulled (`ollama pull phi3.5`) before the server starts.

### 2. Why deterministic keyword fallback exists alongside LLM routing

No production system should have a single point of failure on a path that runs for every
request. If Ollama is unreachable — the service crashed, the model was not pulled, or the
machine has no spare RAM — the keyword regex fallback activates automatically, logging
`[coordinator] Ollama unreachable, falling back to keyword routing`. The system degrades
gracefully rather than catastrophically. The tradeoff is that keyword routing can
misclassify intent-heavy queries (e.g., "tell me about the architecture" relies on no math
or news keyword, so correctly falls to RAG, but more ambiguous phrasing might not). The
LLM classifier exists precisely to handle those cases when available.

### 3. Why DuckDB instead of a dedicated vector database

The case study targets a Phase 1 baseline where operational simplicity matters as much as
capability. DuckDB runs in-process as a single file (`./db/embeddings.db`), requires zero
infrastructure, and supports both SQL queries and vector similarity search simultaneously
in the same query. The `vss` extension provides HNSW approximate nearest-neighbour search;
if the extension fails to load on any platform, the fallback uses `list_cosine_similarity()`
— a pure SQL expression that works on any DuckDB version. The tradeoff is scalability:
DuckDB is not suitable for millions of vectors or multi-process writes, making it appropriate
for the embedded single-process architecture of this project but not for a production
multi-tenant deployment.

### 4. Why structural/recursive chunking over fixed-size chunking

Fixed-size chunking (split every N characters) is simple but destroys semantic coherence —
a sentence explaining a concept can be split across two chunks that neither chunk can fully
represent. Structural chunking respects the document's own boundaries: Markdown section
headers, paragraph breaks, and sentence boundaries are tried in that priority order before
falling back to a hard character limit. The result is chunks that contain **contextually
complete units of meaning**, which improves both embedding quality and retrieval precision.
The tradeoff is uneven chunk sizes, which can occasionally produce very short chunks (a
single header line) or very long chunks (a dense paragraph over the size limit), both of
which can affect embedding quality relative to the rest of the corpus.

### 5. Why RAG context is always shown explicitly

The case study evaluation criteria explicitly require that retrieved context be visible to the
user. Beyond compliance, showing the chunks also builds **interpretability and trust**: the
user can see which source document and which passage produced the answer, and can judge
whether the model synthesised the chunks correctly. This is essential for a research
assistant where hallucination on retrieved content is a risk. The tradeoff is longer
responses; the agent always shows chunk content, source file, and similarity score before
the synthesised answer.

### 6. Why asteval over Python `eval()`

Python's built-in `eval()` executes arbitrary expressions in the interpreter. A user (or a
prompt injection attack) could pass `__import__('os').system('rm -rf /')` and the agent
would execute it with the process's full permissions. `asteval` compiles the input string to
an **Abstract Syntax Tree** and only evaluates nodes that correspond to mathematical
operations (arithmetic, functions, comparisons). Shell commands, module imports, file
operations, and attribute access are rejected at the AST validation stage. The tradeoff is
that `asteval` does not support all Python expressions — which is entirely intentional.

---

## Setup Instructions

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) installed with phi3.5 pulled:
  ```bash
  ollama pull phi3.5
  ```
- API keys (all have free tiers):
  - [OpenRouter](https://openrouter.ai) — primary LLM gateway
  - [Tavily](https://tavily.com) — web search
  - [Groq](https://console.groq.com) — LLM fallback

### Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd MultiAgentAISystem_CaseStudy

# 2. Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env and fill in your API keys:
#   OPENROUTER_API_KEY=sk-or-v1-...
#   TAVILY_API_KEY=tvly-...
#   GROQ_API_KEY=gsk_...
#   OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
#   LLM_MODEL=meta-llama/llama-3.3-70b-instruct:free

# 5. Initialise the database
python database.py

# 6. Add your documents to documents/
#    Supported formats: PDF (.pdf), Markdown (.md), Plain text (.txt)
cp your_documents/* documents/

# 7. Run ingestion
python ingest.py
#    Output: "Ingested N chunks from M files"

# 8. Start the Agno AgentOS UI server
python app.py
#    Server starts at http://localhost:7777

# 9. Open https://app.agno.com in your browser
#    Click "Add endpoint" and enter: http://localhost:7777
#    All four agents will appear: Coordinator, RAG Agent,
#    Calculator Agent, Web Search Agent

# 10. (Alternative) Use the REST API directly
uvicorn main:app --reload --port 8000
```

### REST API Usage

```bash
# Health check
curl http://localhost:8000/health

# Calculator query (routes to Calculator Agent)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is 144 divided by 12 plus 37?"}'

# Expected response:
# {
#   "query": "What is 144 divided by 12 plus 37?",
#   "routed_to": "Calculator Agent",
#   "response": "The expression 144/12+37 equals 49.0",
#   "status": "success",
#   "classification": "CALCULATOR",
#   "routing_method": "ollama_llm"
# }

# RAG query (routes to RAG Agent)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What does the document say about the embedding model?"}'

# Web search query (routes to Web Search Agent)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Who invented the internet?"}'

# List ingested chunks
curl "http://localhost:8000/chunks?limit=10"
```

---

## Common Pitfalls Addressed

The case study documentation identifies several common failure modes in student
implementations. This section maps each pitfall to the specific mechanism that addresses it.

**"Always calling RAG (no decision logic)"**
Every query passes through the Ollama phi3.5-mini intent classifier before any agent is
invoked. The classifier reasons about what the user is trying to accomplish, not what words
they use. "Who invented the internet?" classifies as `SEARCH` (routes to Tavily), not `RAG`.
"What is 15% of 240?" classifies as `CALCULATOR`, not `RAG`. Only queries that are genuinely
about the uploaded documents classify as `RAG`. The `routing_method` field in every API
response shows whether the decision came from the LLM or the keyword fallback.

**"Not showing retrieved context"**
The RAG Agent's system prompt explicitly forbids answering from memory alone. The
`document_lookup` tool returns a structured dict containing the full content of each
retrieved chunk, its source file, and its cosine similarity score. The agent is instructed
to show each retrieved chunk before giving its synthesised answer. This is verified in
every RAG test: the response always begins with chunk contents and source attributions.

**"Hardcoded or shallow implementations"**
The system has three independent layers of fallback: Ollama → keyword regex for routing;
OpenRouter → Groq → direct tool call for LLM inference; HNSW index → pure SQL cosine for
vector search. Every path is tested and every failure mode has a defined recovery. Async SSL
bypass (`httpx.AsyncClient(verify=False)`) is handled by injecting pre-built SDK clients
into Agno's model objects, not by monkey-patching or environment variables. Ingestion is
idempotent via `INSERT OR REPLACE`.

**"Fake multi-agent setups (no real separation)"**
Each agent has a distinct system prompt, a distinct set of tools, and handles a genuinely
different kind of query. The Coordinator has **no tools for answering questions** — it can
only call `route_to_rag`, `route_to_calculator`, or `route_to_web_search`. It is
structurally incapable of answering directly. The RAG Agent has only `document_lookup`.
The Calculator Agent has only `safe_calculate`. The Web Search Agent has only `web_search`.
Agent boundaries are enforced by the tool schema, not just by instructions.

---

## Project Structure

```
MultiAgentAISystem_CaseStudy/
│
├── app.py                  # Agno AgentOS UI server (port 7777)
├── main.py                 # FastAPI REST API server (port 8000)
├── config.py               # All configuration constants + .env loading
├── database.py             # DuckDB init, embeddings, vss search, SQL fallback
├── ingest.py               # Document ingestion pipeline (PDF/MD/TXT → DuckDB)
├── llm_client.py           # Model factory: OpenRouter (primary), Groq (fallback)
│                           # Handles async SSL bypass for corporate proxies
│
├── agents/
│   ├── coordinator.py      # Ollama classifier + keyword fallback + 3-tier routing
│   ├── rag_agent.py        # RAG Agent + document_lookup tool
│   ├── calculator_agent.py # Calculator Agent + safe_calculate tool (asteval)
│   └── web_search_agent.py # Web Search Agent + web_search tool (Tavily)
│
├── documents/              # Source documents for RAG (PDF, Markdown, TXT)
│   ├── architecture.md
│   └── overview.txt
│
├── db/
│   └── embeddings.db       # DuckDB vector store (auto-created, not committed)
│
├── docs/
│   ├── architecture.png    # Architecture diagram (generated by generate_diagram.py)
│   └── generate_diagram.py # Diagram source — regenerate with: python docs/generate_diagram.py
│
├── requirements.txt        # Python dependencies
├── .env                    # API keys (not committed — see .env.example)
├── .gitignore
├── progress_tracker.json   # Feature completion tracking
└── claude-progress.txt     # Session-level build log
```

---

## Evaluation Criteria Mapping

| Criterion | Where Implemented |
|---|---|
| **Agent Design** — clear reasoning, not hardcoded routing | Ollama phi3.5 intent classification in `coordinator.py:_classify_with_ollama()` with 10 few-shot examples |
| **RAG Quality** — relevant retrieval with transparency | Structural chunking in `ingest.py`, HNSW + cosine search in `database.py:search_chunks()`, chunk + score display in `rag_agent.py` |
| **Tool Usage** — structured and meaningful | Typed inputs/outputs for `safe_calculate` (asteval), `document_lookup` (DuckDB), `web_search` (Tavily) |
| **System Design** — modular and clean | Separate files per concern: `config.py`, `llm_client.py`, `database.py`, `ingest.py`, one file per agent |
| **Fallback Robustness** — no single point of failure | Three-tier LLM fallback in `coordinator.py:_run_with_fallback()`, Ollama → keyword in `_classify_with_ollama()`, VSS → SQL in `database.py:search_chunks()` |
| **Code Quality** — readability and correctness | Docstrings on all public functions, typed return annotations, `progress_tracker.json` for feature state |
| **Security** — no code injection | `asteval` (not `eval()`) in `calculator_agent.py`, parameterised DuckDB queries, no shell execution anywhere |
| **Thinking** — depth of design decisions | This document, `claude-progress.txt`, commit messages with rationale |
