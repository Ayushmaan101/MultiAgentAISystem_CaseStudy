import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agno.agent import Agent
from agno.tools import tool

import database
import config
from llm_client import get_synthesis_model


_MAX_CTX_CHARS = 2200  # max parent context chars per chunk — captures full sections, stays under TPM


def _context_window(parent: str, child: str, window: int = _MAX_CTX_CHARS) -> str:
    """Return up to `window` chars of parent centered on where child appears."""
    idx = parent.find(child[:60]) if child else -1
    if idx == -1:
        # Child not located in parent — return start of parent
        return parent[:window] + ("..." if len(parent) > window else "")
    start = max(0, idx - window // 4)
    end = min(len(parent), idx + len(child) + (window * 3 // 4))
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(parent) else ""
    return prefix + parent[start:end] + suffix


@tool
def document_lookup(query: str) -> str:
    """
    Search the internal knowledge base for relevant document chunks.
    Returns top-K chunks with source file, similarity score, matched
    paragraph, and surrounding section context.
    Always call this before answering any document question.
    """
    results = database.search_chunks(query, config.TOP_K)

    if not results:
        return "No relevant chunks found in the knowledge base."

    formatted = "=== RETRIEVED CHUNKS ===\n"
    for i, (parent_content, source, score, child_content) in enumerate(results, 1):
        ctx = _context_window(parent_content, child_content)
        formatted += f"\n[Chunk {i}]\n"
        formatted += f"Source: {source}\n"
        formatted += f"Similarity: {score:.6f}\n"
        formatted += f"Matched paragraph: {child_content}\n"
        formatted += f"Full section context:\n{ctx}\n"
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
