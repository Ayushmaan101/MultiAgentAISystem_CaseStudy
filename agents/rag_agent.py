import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agno.agent import Agent

import config
import database
from llm_client import get_fallback_model, get_precise_model


def document_lookup(query: str) -> str:
    """Search the local knowledge base for chunks relevant to the query."""
    results = database.search_chunks(query, config.TOP_K)
    if not results:
        return "=== RETRIEVED CHUNKS ===\n\nNo chunks found for this query.\n\n=== END RETRIEVED CHUNKS ==="

    lines = ["=== RETRIEVED CHUNKS ===\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"[Chunk {i}]")
        lines.append(f"Source: {r['source_file']}")
        lines.append(f"Similarity: {r['similarity']:.6f}")
        lines.append(f"Content: {r['content']}")
        lines.append("")
    lines.append("=== END RETRIEVED CHUNKS ===")
    return "\n".join(lines)


rag_agent = Agent(
    name="RAG Agent",
    model=get_precise_model(),
    tools=[document_lookup],
    instructions=[
        "You are a document retrieval assistant.",
        "Your ONLY available tool is document_lookup. You have no other tools.",
        "You MUST call document_lookup for every query. Do NOT answer from memory.",
        "Step 1: Call document_lookup with the user's query.",
        "Step 2: Copy the entire === RETRIEVED CHUNKS === block verbatim into your response first.",
        "Step 3: Write a synthesized answer citing chunk number and source file.",
    ],
    markdown=True,
)
