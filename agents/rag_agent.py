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
        "Your ONLY job is to retrieve relevant chunks from the knowledge base.",
        "You do NOT analyze, summarize, or synthesize. That is done elsewhere.",
        "",
        "Follow these steps for EVERY query:",
        "Step 1: Call document_lookup with the query as provided.",
        "Step 2: Check the similarity scores of returned chunks.",
        "   - If ALL scores are below 0.15: return the chunks anyway",
        "     and add a note: 'Low similarity — consider web search'",
        "   - If at least one score is above 0.15: chunks are relevant",
        "Step 3: If ALL scores are below 0.10, refine the search term",
        "        and call document_lookup ONE more time with better keywords.",
        "        Only retry once.",
        "Step 4: Your ENTIRE response must be the VERBATIM output of document_lookup.",
        "        Copy and paste it exactly — do NOT add any text before or after.",
        "        Do NOT add analysis, summaries, explanations, or bullet lists.",
        "        The raw document_lookup output IS your complete response.",
        "",
        "STRICT RULES:",
        "- ALWAYS call document_lookup before responding",
        "- NEVER answer from your own training data",
        "- NEVER analyze or synthesize the chunks",
        "- NEVER skip the tool call",
        "- Return raw chunks only",
    ],
)
