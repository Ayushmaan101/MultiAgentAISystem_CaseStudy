
**Phase 1: Finalized Tech Stack & Baseline Strategy**

The core objective of Phase 1 is **functional predictability**. We want to ensure that data flows seamlessly from ingested files into DuckDB, gets accurately routed by the Agno Coordinator, and populates the Agno UI backend without any overlapping points of failure.

|**Component**|**Technology**|**Why We Are Implementing It**|
| :- | :- | :- |
|**Ingestion Pipeline**|Python Native + Agno Loaders (PDF, Markdown, Text)|Natively preserves structural semantics (like markdown headers) at the mouth of the pipeline.|
|**Chunking Strategy**|**Structural/Recursive Chunking**|Slices text by logical boundaries (#, \n\n) instead of fixed character counts, keeping contextual paragraphs intact.|
|**Embedding Engine**|Local BAAI/bge-small-en-v1.5|Extremely lightweight (33M parameters), high-performing local model that eliminates API latency and cost during testing.|
|**Vector Storage**|**DuckDB** + vss Extension|Embedded database running completely in-process; allows us to query vectors and relational metadata simultaneously using SQL. Includes a pure SQL cosine-similarity fallback script.|
|**Orchestration Layer**|**Agno Agent OS** (FastAPI Backend)|Handles agent routing, history management, and native tool-calling. Exposes an automated FastAPI server out of the box.|
|**User Interface**|**Agno UI** (Next.js App)|Directly satisfies the project case study requirements, allowing us to stream agent execution steps and retrieved chunks natively.|
|**LLM Gateway**|**OpenRouter** (Llama 3.3 70B / DeepSeek V3)|Provides access to top-tier reasoning models with structured JSON output enforcement, guarded by a 20 req/min rate limiter filter.|
|**Tool Suite**|Tavily API, asteval Calculator, DB Lookup|Equips sub-agents with web search, mathematical execution (via a safe AST compiler to prevent arbitrary code execution), and local RAG access.|

**Phase 1 Execution Strategy**

1. **Database & Schema Initialization:** We configure DuckDB, activate the vss extension, and create a single unified table that stores the raw text chunk, its embedding vector, and comprehensive metadata (source file, file type, timestamp).
1. **Deterministic Processing:** Files dropped into the ingestion directory are parsed based on extension. Markdown headings are given highest priority for chunk boundaries.
1. **The Orchestration Test:** We spin up the Agno App as a local FastAPI application. We test the Coordinator agent purely via terminal/curl commands first to ensure it perfectly routes queries to the Document Lookup agent, Calculator agent, or Web Search agent without throwing errors.

**Phase 2: Advanced Pipeline Upgrades (The "Gold Standard")**

Once Phase 1 (our Recursive chunking + vss baseline) is 100% verified, we will systematically layer in these advanced optimizations to maximize accuracy and efficiency:

- **1. Chunking Upgrade (Parent-Child / Semantic)**
  - **What:** Transitioning from basic Recursive chunking to Parent-Child (embedding small sentences but returning the larger paragraph) or Semantic chunking (splitting text where the topic shifts).
  - **Why:** Solves the RAG dilemma. It gives the database hyper-specific text to search against, while giving the LLM the broad surrounding context it needs to generate a cohesive answer.
- **2. Query Betterment (Expansion)**
  - **What:** Using a lightweight, fast LLM call to generate 3-4 semantic variations of the user's initial prompt before it hits the database. *(Note: We will make this toggleable to protect your OpenRouter rate limits).*
  - **Why:** Prevents vocabulary mismatch. If a user searches for "uptime" but the document says "reliability," query expansion ensures the database still catches it.
- **3. Hybrid Search (VSS + FTS/BM25)**
  - **What:** Activating DuckDB’s native Full-Text Search (FTS/BM25) alongside our existing Vector Similarity Search (vss).
  - **Why:** Vectors are great for "vibes" and concepts, but terrible at exact matches. Hybrid search guarantees you don't miss highly specific alphanumeric IDs, acronyms, or proper nouns.
- **4. Double Retrieval (Cross-Encoder Reranking)**
  - **What:** Pulling a wide net of chunks (e.g., Top 50) using Hybrid Search, then passing them through a local Cross-Encoder model (BAAI/bge-reranker-base) to accurately re-score their true logical relevance to the prompt.
  - **Why:** This is the ultimate noise filter. It stops mathematically similar but logically irrelevant chunks from poisoning the LLM's context window.
- **5. Maximal Marginal Relevance (MMR)**
  - **What:** A mathematical filter applied strictly *after* the Reranker to shave the Top 50 down to the final Top 5.
  - **Why:** It penalizes duplicate information. If the top chunks all say the exact same thing, MMR swaps out the duplicates for unique secondary context, giving the LLM a broader, multi-dimensional view of the topic.


