import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agno.agent import Agent

import config
import database
from llm_client import get_model


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
    model=get_model(),
    tools=[document_lookup],
    instructions=[
        "You are a document retrieval assistant.",
        "Always call document_lookup before answering — never rely on memory.",
        "When you receive the tool result, copy the entire RETRIEVED CHUNKS block verbatim into your response first.",
        "After the chunks block, write your synthesized answer.",
        "In your answer, cite which chunk number and source file each claim comes from.",
    ],
    markdown=True,
)
