import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agno.agent import Agent
from agno.tools import tool

import database
import config
from llm_client import get_synthesis_model


@tool
def document_lookup(query: str) -> str:
    """
    Search the internal knowledge base for relevant document chunks.
    Returns top-K chunks with source file and similarity scores.
    Always call this before answering any document question.
    """
    results = database.search_chunks(query, config.TOP_K)

    if not results:
        return "No relevant chunks found in the knowledge base."

    formatted = "=== RETRIEVED CHUNKS ===\n"
    for i, r in enumerate(results, 1):
        content = r["content"]
        source = r["source_file"]
        score = r["similarity"]
        formatted += f"\n[Chunk {i}]\n"
        formatted += f"Source: {source}\n"
        formatted += f"Similarity: {score:.6f}\n"
        formatted += f"Content: {content}\n"
        formatted += "-" * 40 + "\n"
    formatted += "\n=== END RETRIEVED CHUNKS ==="
    return formatted


rag_agent = Agent(
    name="RAG Agent",
    model=get_synthesis_model(),
    tools=[document_lookup],
    debug_mode=True,
    markdown=True,
    description=(
        "Specialist document retrieval agent that searches "
        "the internal knowledge base and evaluates chunk relevance."
    ),
    instructions=[
        "You are a specialist document retrieval agent.",
        "You have access to an internal knowledge base via document_lookup.",
        "",
        "Follow these steps for EVERY query:",
        "Step 1: Analyze the query to identify the core information need.",
        "Step 2: Call document_lookup with the query as provided.",
        "Step 3: Evaluate the retrieved chunks:",
        "   - Check similarity scores",
        "   - If ALL similarity scores are below 0.15, state that "
        "     relevant content was not found and recommend web search",
        "   - If at least one chunk has similarity above 0.15, proceed",
        "Step 4: If initial results are poor quality, refine your search "
        "        term and call document_lookup ONE more time with better "
        "        keywords. Only retry once.",
        "Step 5: Identify the most relevant chunks and explain why "
        "        they are relevant to the query.",
        "Step 6: Return the raw retrieved chunks verbatim so the user "
        "        can see the source material.",
        "Step 7: Always cite the chunk number and source file.",
        "",
        "STRICT RULES:",
        "- ALWAYS call document_lookup before responding",
        "- NEVER answer from your own training data",
        "- NEVER skip the tool call",
        "- Show your reasoning at every step",
        "- If chunks are insufficient, say so honestly",
    ],
)
