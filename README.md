# AI Research Assistant

> A multi agent AI system featuring a five agent architecture,
> KB aware intent classification, autonomous document retrieval,
> multi-tool orchestration, and verbose reasoning.

---

## Table of Contents
1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Five-Node Pipeline](#3-five-node-pipeline)
4. [Agent Roster](#4-agent-roster)
5. [RAG Pipeline](#5-rag-pipeline)
6. [Tech Stack](#6-tech-stack)
7. [Design Decisions and Tradeoffs](#7-design-decisions-and-tradeoffs)
8. [Setup Instructions](#8-setup-instructions)
9. [Usage](#9-usage)
10. [Common Pitfalls Addressed](#10-common-pitfalls-addressed)
11. [Evaluation Criteria Mapping](#11-evaluation-criteria-mapping)
12. [Bonus Features](#12-bonus-features-implemented)
13. [Project Structure](#13-project-structure)

---

## 1. Overview

This is a multi agent AI Research Assistant that answers user queries
using RAG, tool usage, and intelligent agent orchestration. The system 
is built around a five agent architecture where each agent has a single, 
well defined responsibility.

The core architectural principle is separation of concerns, ie, reasoning
is separated from execution to prevent LLMs from hallucinations and tool bypass. 
LLMs handle what they excel at: understanding intent, rewriting queries, evaluating 
retrieved content, and synthesizing answers. Deterministic Python handles tool dispatch, 
guaranteeing reliability and eliminating a class of non deterministic failures
encountered during development where LLMs would inconsistently format
tool calls depending on whether they already knew the answer from
training data.

As a result we get a system where every component is independently testable,
every decision is logged and traceable, and every failure mode has a
documented fallback.

---

## 2. Architecture

### Architecture Diagram
![Architecture](docs/architecture.png)

### High-Level Flow

```
User Query
    |
    v
NODE 1: Intent Classifier
Ollama phi3.5 (local, keep_alive=5m)
KB-aware prompt + 15 few-shot examples
Output: RAG / CALCULATOR / SEARCH / MULTI
    |
    v
NODE 2: Similarity Safety Net
Python only — DuckDB vector check
Only triggers when Node 1 outputs SEARCH
If top similarity > 0.25 — override to RAG
    |
    v
NODE 3: Query Rewriter
Ollama phi3.5 (already hot via keep_alive)
Input optimisation + target rewriting
in ONE atomic Ollama call
    |
    v
NODE 4: Tool Execution
Python dispatches directly
RAG    -> RAG Agent (Groq qwen3-32b)
CALC   -> safe_calculate (asteval)
SEARCH -> web_search (Tavily)
MULTI  -> Tracker Agent (Groq qwen3-32b)
    |
    v
NODE 5: General Reasoning Agent
Groq qwen3-32b
Synthesizes final answer from tool result only
Shows explicit 6-step reasoning
    |
    v
Agno AgentOS UI
Tool calls visible in dropdown
6-step reasoning visible in chat
Raw chunks with similarity scores shown
```

---

## 3. Five-Node Pipeline

### Node 1 - Intent Classifier (Ollama phi3.5, local)

Single job: classify query into RAG, CALCULATOR, SEARCH, or MULTI.

Model: phi3.5 running locally via Ollama with keep_alive=5m so the
model stays hot in RAM between Node 1 and Node 3, eliminating the
20-30 second model reload penalty on CPU hardware.

KB-aware prompt: the system prompt explicitly describes what the
knowledge base contains so the model makes informed routing decisions
based on content awareness rather than surface keyword matching.
Without this, a query like "What is the chunk overlap value?" would
be misclassified as SEARCH because it contains no document keywords.

Few-shot examples: 15 examples covering edge cases including
plain language math ("three dozen eggs"), domain specific queries
without keywords, and world knowledge questions.

MULTI classification: detects compound queries that require more than
one tool and routes them to the Tracker Agent.

Fallback: if Ollama is unreachable, falls back to keyword routing so
the system never crashes.

### Node 2 - Similarity Safety Net (Python, no LLM)

Single job: verify whether the KB has relevant content
before confirming a SEARCH classification.

Only triggers when Node 1 classifies SEARCH.

Mechanism: runs a fast DuckDB vector similarity check on the original
query. If the top chunk similarity score exceeds 0.25, the route is
overridden to RAG.

Why this matters: even with KB aware prompting, ambiguous queries can
slip through. This node makes the RAG vs SEARCH decision empirically
rather than probabilistically. It directly satisfies the case study
requirement to avoid always calling RAG with no decision logic, hence RAG
is only called when KB relevance is empirically confirmed.

Cost: DuckDB is embedded and local, adding approximately 50ms maximum.

### Node 3 - Query Rewriter (Ollama phi3.5, already hot)

Single job: optimise raw user input AND rewrite for the target agent
in one atomic Ollama call.

Already hot: phi3.5 stays in RAM from Node 1 via keep_alive so this
call has near-zero additional latency.

Input optimisation: fixes typos, removes filler words such as "umm",
"basically", "can you", and normalises the text before rewriting.

Target specific rewriting:
- CALCULATOR: extract clean math expression for asteval
  example: "three dozen eggs use half" becomes "(3 * 12) / 2"
- RAG: clean keyword search query for DuckDB vector search
  example: "tell me abt the chunking thing" becomes "chunking strategy method"
- SEARCH: clean factual web search question
- MULTI: split into two typed sub-queries returned as JSON

Why separate from Node 1: compound tasks cause attention dilution in
small models. A model asked to classify AND rewrite will sometimes
attempt to answer the query directly. Separation guarantees reliable
single task output from each node.

### Node 4 - Tool Execution (Python dispatch)

Single job: call the correct tool with the rewritten query from Node 3.

During development, Llama models on Groq non deterministically switched
between JSON and Hermes XML format for tool calls depending on whether
they recognised the query from training data. Groq rejects Hermes XML,
causing unpredictable failures that could not be resolved through
prompting alone. Python dispatch separates the routing concern from
the reasoning concern and makes tool execution fully deterministic.

RAG path: calls the RAG Agent which autonomously decides its search
strategy, evaluates chunk relevance, and retries with refined terms
if initial results are poor quality.

CALCULATOR path: calls safe_calculate() directly with the pre-extracted
math expression from Node 3.

SEARCH path: calls web_search() directly with the pre-cleaned search
query from Node 3.

MULTI path: calls the Tracker Agent which has all three tools available
and orchestrates them autonomously based on the sub queries from Node 3.

### Node 5 - General Reasoning Agent (Groq qwen3-32b)

Single job: synthesize the final answer from the raw tool result only.

Never calls tools. Pure reasoning over provided content.

Strictly grounded: constrained to use only information from the tool
result. Never supplements from training data.

Explicit 6-step reasoning shown in the UI:
- Step 1: Restate the question
- Step 2: Identify available information
- Step 3: Evaluate information quality and relevance
- Step 4: Reason through the answer
- Step 5: State the final answer clearly
- Step 6: Cite all sources - chunk number and file for RAG,
  expression and result for calculator, URLs for search

Why verbose reasoning: makes the system thinking transparent and
auditable, satisfies the logging and tracing bonus requirement, and
demonstrates genuine deliberate reasoning rather than opaque chaining.

---

## 4. Agent Roster

| Agent | Model | Job |
|---|---|---|
| Coordinator | Ollama phi3.5 | Orchestrates all five nodes, classifies intent |
| Query Rewriter | Ollama phi3.5 | Input optimisation and target-specific rewriting |
| RAG Agent | Groq qwen3-32b | Autonomous retrieval with self-evaluation and retry logic |
| Tracker Agent | Groq qwen3-32b | Multi-tool orchestration for compound queries |
| General Reasoning Agent | Groq qwen3-32b | Verbose 6-step synthesis from tool results |

### Tool Functions (Pure Python)

| Function | File | Interacts With |
|---|---|---|
| document_lookup() | agents/rag_agent.py | DuckDB vector store |
| safe_calculate() | agents/calculator_agent.py | asteval interpreter |
| web_search() | agents/web_search_agent.py | Tavily Search API |

### Why this hybrid architecture?

This system uses the Brain and Workers pattern from production
multi agent systems.

Brain agents (LLM powered) handle probabilistic language driven tasks:
understanding intent, rewriting queries, evaluating retrieved content,
reasoning over results, and synthesizing coherent answers.

Worker functions (deterministic Python) handle tasks where reliability
and exactness matter: math evaluation, vector search, and web API calls.

This separation means LLM non determinism only affects the parts of
the system where it is acceptable (language understanding and generation) and not 
the parts where it is dangerous (tool execution and routing).

---

## 5. RAG Pipeline

The RAG pipeline has five stages: ingestion, chunking, embedding,
storage, and retrieval.

```
documents/ (PDF, Markdown, TXT)
    |
    v
Structural/Recursive Chunker (ingest.py)
Priority: Markdown headers > double newlines > single newlines > char limit
CHUNK_SIZE=500, CHUNK_OVERLAP=50 chars of context carried forward
    |
    v
BAAI/bge-small-en-v1.5 (local, 33M params, 384 dimensions)
Fully local - zero API cost and zero latency penalty
Generates normalized 384-dimensional float vector per chunk
    |
    v
DuckDB + vss extension
HNSW index for fast approximate nearest-neighbour search
Pure SQL cosine fallback via list_cosine_similarity() if vss unavailable
Idempotent ingestion via INSERT OR REPLACE
Schema: id, content, embedding FLOAT[384], source_file,
        file_type, chunk_index, timestamp
    |
    v
RAG Agent - search_chunks(rewritten_query, TOP_K=5)
Generates query embedding
HNSW index search returns top-K chunks with similarity scores
Self-evaluates: if all scores below 0.15 then retries with refined query
Returns raw chunks: content + source_file + similarity score
    |
    v
General Reasoning Agent
Synthesizes answer strictly from retrieved chunks
Cites chunk number and source file in Step 6
```

---

## 6. Tech Stack

| Component | Technology | Reason |
|---|---|---|
| UI | Agno AgentOS | Native tool call visibility, chunk display, session tracking |
| Intent Classification | Ollama phi3.5 (local) | Zero API cost, keep_alive optimization, no rate limits |
| Query Rewriting | Ollama phi3.5 (local) | Already hot in RAM, zero token cost, atomic single task |
| RAG Agent and Synthesis | Groq qwen3-32b | Strong instruction following, reliable tool calls |
| Multi-tool Orchestration | Groq qwen3-32b | Handles compound queries reliably across tool types |
| Coordinator Shell | Groq gpt-oss-20b | Consistent JSON tool call format for AgentOS registration |
| Vector Database | DuckDB + vss extension | Embedded, zero infrastructure, SQL and vector queries simultaneously |
| Embeddings | BAAI/bge-small-en-v1.5 | 33M params, 384D, fully local, no API latency or cost |
| Web Search | Tavily API | Structured results purpose-built for LLM agent workflows |
| Safe Math | asteval | AST-based interpreter, prevents arbitrary code execution |
| Primary LLM Gateway | OpenRouter | Wide model access, attempted first on every request |
| Fallback LLM Gateway | Groq | Fast free-tier inference, activated when OpenRouter unavailable |

---

## 7. Design Decisions and Tradeoffs

### Decision 1: Atomic nodes with single responsibilities

Problem: When a small local model is given compound tasks such as
"classify this query AND rewrite it for the target agent", it suffers
attention dilution. It gets distracted by context, forgets output
constraints, and sometimes attempts to answer the query directly
instead of performing the assigned transformation.

Solution: Every node has exactly one job, one input format, and one
output format. Node 1 outputs one word. Node 3 outputs one rewritten
query string. Node 5 outputs one synthesized answer. Each node is
independently testable and debuggable.

Tradeoff: More sequential calls increase latency. Mitigated by
keep_alive which eliminates model reload overhead between nodes.

### Decision 2: Separation of routing from execution

Problem: During development, Llama models on Groq non-deterministically
switched between JSON and Hermes XML format for tool calls depending
on whether the model recognised the query from training data. This made
the failure mode impossible to reproduce consistently or fix through
prompting. The same query would succeed on one run and fail on the next.

Solution: Routing decisions are made by Ollama in Node 1 and Python
in Node 4. LLMs are given only language tasks - rewriting and synthesis.
This separates the reasoning concern from the execution concern and
makes tool dispatch fully deterministic.

Tradeoff: The system cannot dynamically discover new tool combinations
at runtime which is acceptable for our use case which has a fixed tool set.

### Decision 3: keep_alive=5m for Ollama

Problem: Without keep_alive, Ollama unloads phi3.5 from RAM after each
request. On CPU-only hardware, reloading 2.2GB of model weights from
disk adds 20-30 seconds of latency for every Node 3 call.

Solution: keep_alive=5m is passed in every Ollama API call. phi3.5
stays hot in RAM. Node 3 fires almost instantly after Node 1 because
the model is already loaded.

Tradeoff: Holds 2.2GB RAM for 5 minutes between requests.

### Decision 4: KB description in the classification prompt

Problem: Without context about what the knowledge base contains,
phi3.5 defaults domain specific queries to SEARCH. Queries like
"What is the chunk overlap value?" and "What are the Phase 2
optimizations?" have no surface level document keywords but their
answers exist in the KB.

Solution: The classification prompt explicitly describes KB contents.
phi3.5 makes informed routing decisions based on content awareness
rather than keyword matching.

Tradeoff: Longer system prompt increases tokens per classification
call. Negligible since Ollama runs locally at zero API cost.

### Decision 5: Similarity safety net in Node 2

Problem: Even with KB-aware prompting, ambiguous queries occasionally
receive incorrect SEARCH classifications. A missed RAG routing means
the user gets a web result when the answer exists in their documents.

Solution: Every SEARCH classification triggers a fast DuckDB similarity
check. If the top chunk similarity score exceeds 0.25, the route is
overridden to RAG. The decision is empirical — based on actual vector
distance — not a second probabilistic guess.

Tradeoff: Every SEARCH query performs one additional DB lookup.
DuckDB is embedded and local so this adds approximately 50ms.

### Decision 6: Local Ollama for routing and rewriting

Problem: Using an API model for every routing and rewriting call burns
token quota on the least valuable tasks in the pipeline. At scale,
every query would consume multiple API calls before any meaningful
work is done.

Solution: phi3.5 handles classification and rewriting locally. Zero
token cost on the routing path. API quota is reserved for RAG retrieval
evaluation and final synthesis where model quality genuinely matters.

Tradeoff: Requires Ollama installed locally. System degrades gracefully
to keyword routing if Ollama is unreachable.

### Decision 7: Structural chunking over fixed size chunking

Problem: Fixed size chunking cuts text at arbitrary character positions,
frequently splitting sentences and paragraphs mid-thought. This produces
semantically incoherent chunks that degrade retrieval quality.

Solution: Structural chunking respects logical document boundaries.
Markdown headers are tried first, then paragraph breaks, then sentence
breaks, then character limit as a last resort. Each chunk represents a
complete semantic unit.

Tradeoff: Produces uneven chunk sizes. Accepted because semantic
coherence improves retrieval precision.

### Decision 8: asteval over Python eval()

Problem: Python eval() executes arbitrary code. If a user sends a
malicious expression, eval() would execute it with full Python
privileges, which poses a critical security vulnerability.

Solution: asteval compiles expressions to an Abstract Syntax Tree and
only permits mathematical operations. Arbitrary code execution is
structurally impossible.

Tradeoff: Does not support full Python syntax. This is intentional since
the calculator should only evaluate math expressions.

---

## 8. Setup Instructions

### Prerequisites
- Python 3.10+
- Ollama installed from https://ollama.ai
- phi3.5 model pulled: `ollama pull phi3.5`
- Groq API key free tier from https://console.groq.com
- Tavily API key free tier from https://tavily.com
- OpenRouter API key free tier from https://openrouter.ai

### Installation

```bash
git clone https://github.com/Ayushmaan101/MultiAgentAISystem_CaseStudy.git
cd MultiAgentAISystem_CaseStudy
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
python database.py
python ingest.py
ollama serve
python app.py
```

### Connecting the AgentOS UI
1. Open https://os.agno.com
2. Click Connect OS
3. Select Local
4. Enter endpoint: http://localhost:7777
5. Click Connect
6. Select Coordinator from the agents list

### REST API Usage

```bash
uvicorn main:app --reload --port 8000

curl http://localhost:8000/health

curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"What is the chunking strategy used?\"}"

curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"What is 15 percent of 240?\"}"

curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"Who is the current CEO of OpenAI?\"}"

curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"What does the doc say about chunking and what is 500 divided by 6?\"}"

curl http://localhost:8000/chunks
```

---

## 9. Usage

### Query Examples by Type

RAG queries answered from knowledge base:
```
What is the chunking strategy used in this system?
What are the Phase 2 optimizations planned?
What embedding model is used?
What is the VSS extension?
What are the components of the retrieval layer?
What is the chunk overlap value?
```

Calculator queries:
```
What is 15 percent of 240?
If I have three dozen eggs and use half, how many remain?
What is the square root of 256?
What is 144 divided by 12 plus 37?
```

Web search queries:
```
Who is the current CEO of OpenAI?
When was the Eiffel Tower built?
What are the latest developments in AI agents?
```

MULTI queries using multiple tools:
```
What does the document say about chunking and what is 500 divided by 6?
What is the embedding model used and who invented the internet?
```

---

## 10. Common Pitfalls Addressed

| Pitfall | How This System Avoids It |
|---|---|
| Always calling RAG with no decision logic | Node 1 classifies intent with KB-aware prompt and 15 examples. Node 2 empirically confirms via similarity check. Math goes to Calculator. World knowledge goes to Search. RAG is only called when KB relevance is confirmed. |
| Not showing retrieved context | RAG Agent returns raw chunks with source file and similarity scores visible in the AgentOS tool dropdown. General Reasoning Agent cites chunk number and source file in Step 6 of every response. |
| Hardcoded or shallow implementations | Five atomic nodes. keep_alive optimization. Pure SQL cosine fallback. Similarity safety net with configurable threshold. Async SSL handling. Idempotent ingestion. Groq rate-limit retry with exponential backoff. |
| Fake multi-agent setups with no real separation | Each agent has distinct model, instructions, tools, and responsibility. Coordinator never answers directly. RAG Agent self-evaluates and retries. Tracker Agent handles compound queries across multiple tools. |

---

## 11. Evaluation Criteria Mapping

| Criterion | Implementation |
|---|---|
| Agent Design — clear reasoning vs prompt chaining | Five agents each with single responsibility. Coordinator classifies before routing. No blind sequential chaining. General Reasoning Agent shows explicit 6-step reasoning on every response. |
| RAG Quality — relevant retrieval not noise | Structural chunking preserves semantic boundaries. HNSW vector search. Similarity safety net prevents wrong routing. RAG Agent self-evaluates and retries on poor results. |
| Tool Usage — structured and meaningful | Python direct dispatch with typed inputs and formatted outputs. asteval for math safety. Tavily for structured web results. DuckDB for vector retrieval. |
| System Design — modular and clean | One file per concern: config.py, llm_client.py, database.py, ingest.py, one file per agent. Each component independently runnable and testable. |
| Code Quality — readability and organization | Docstrings on all node functions. Typed return values. Every node logs classification, rewritten query, and tool result. |
| Thinking — depth of explanation in README | This document. |

---

## 12. Bonus Features Implemented

| Bonus | Implementation |
|---|---|
| Query classification before routing | Node 1 uses Ollama phi3.5 with KB description and 15 few-shot examples covering edge cases including plain-language math and domain queries without keywords. |
| Avoid unnecessary RAG calls | Node 2 similarity safety net. CALCULATOR and SEARCH paths bypass RAG entirely. MULTI only calls RAG when the sub-query type requires it. |
| Logging and tracing | Every node logs classification, rewritten query, similarity scores, tool results, and timing. General Reasoning Agent shows explicit 6-step reasoning trace visible in the UI on every response. |
| Performance optimizations | keep_alive=5m eliminates model reload overhead. Local BAAI embeddings with zero API cost. In-process DuckDB with no network overhead. Groq rate-limit retry with exponential backoff in Node 5. |

---

## 13. Project Structure

```
MultiAgentAISystem_CaseStudy/
|
+-- agents/
|   +-- __init__.py
|   +-- calculator_agent.py         safe_calculate() asteval math tool
|   +-- coordinator.py              Five-node pipeline orchestration
|   +-- general_reasoning_agent.py  Agno Agent verbose 6-step synthesis
|   +-- query_rewriter.py           Ollama query optimisation and rewriting
|   +-- rag_agent.py                Agno Agent autonomous document retrieval
|   +-- tracker_agent.py            Agno Agent multi-tool orchestration
|   +-- web_search_agent.py         web_search() Tavily search tool
|
+-- docs/
|   +-- architecture.png            System architecture diagram
|   +-- generate_diagram.py         Diagram generation script
|
+-- documents/                      Knowledge base source files
|   +-- GenAI_Intern_-_Case_Study.pdf
|   +-- architecture.md
|   +-- case_study_delphi.md
|   +-- overview.txt
|
+-- app.py                          Agno AgentOS entry point port 7777
+-- config.py                       All constants and environment variables
+-- database.py                     DuckDB init embeddings vector search
+-- ingest.py                       Document ingestion pipeline
+-- llm_client.py                   LLM provider with OpenRouter to Groq fallback
+-- main.py                         FastAPI REST API port 8000
+-- requirements.txt                Python dependencies
+-- .env.example                    Environment variable template
+-- .gitignore                      Excludes .env db/ venv/ logs
```
