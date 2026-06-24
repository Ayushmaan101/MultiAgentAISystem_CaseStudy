# Evaluation Dataset

Small evaluation dataset (10 queries) covering all four query types for testing the multi-agent system.

## Dataset

| # | Query | Type | Expected Result |
|---|---|---|---|
| 1 | What is the chunking strategy used in this system? | RAG | Retrieved chunks from architecture.md or case_study_delphi.md describing parent-child chunking |
| 2 | What are the Phase 2 pipeline upgrades planned? | RAG | All 5 upgrades: Parent-Child chunking, Query Expansion, Hybrid Search, Cross-Encoder Reranking, MMR |
| 3 | What is 15 percent of 240? | CALCULATOR | 36.0 |
| 4 | If I have three dozen eggs and use half, how many remain? | CALCULATOR | 18.0 |
| 5 | Who is the current CEO of OpenAI? | SEARCH | Web search result with URL citing Sam Altman |
| 6 | When was the Eiffel Tower built? | SEARCH | Web search result with construction date |
| 7 | What does the document say about chunking and what is 500 divided by 6? | MULTI | Both RAG result (chunking details) AND calculator result (83.33) in one answer |
| 8 | What embedding model is used and who invented the internet? | MULTI | Both RAG result (BAAI/bge-small-en-v1.5) AND search result (who invented internet) in one answer |
| 9 | What is the chunk overlap value? | RAG | Should route to RAG (not SEARCH) due to similarity safety net, answer: 0 (parent-child has no overlap) |
| 10 | What are the components of the retrieval layer? | RAG | Retrieved chunks describing DuckDB, embeddings, HNSW index, parent-child chunking |

## Coverage

- **RAG queries (4):** Tests document retrieval, similarity scoring, and KB awareness
- **CALCULATOR queries (2):** Tests plain-language math and expression extraction
- **SEARCH queries (2):** Tests web search routing and world knowledge questions
- **MULTI queries (2):** Tests compound query orchestration across multiple tools
