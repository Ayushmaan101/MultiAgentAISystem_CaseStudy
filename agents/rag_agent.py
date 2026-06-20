import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agno.agent import Agent
from agno.models.openai.like import OpenAILike

import config
import database


def document_lookup(query: str) -> dict:
    """Search the local knowledge base for chunks relevant to the query."""
    results = database.search_chunks(query, config.TOP_K)
    chunks = [
        {
            "content": r["content"],
            "source_file": r["source_file"],
            "similarity": r["similarity"],
        }
        for r in results
    ]
    return {
        "query": query,
        "Retrieved Context": chunks,
    }


rag_agent = Agent(
    name="RAG Agent",
    model=OpenAILike(
        id=config.LLM_MODEL,
        api_key=config.OPENROUTER_API_KEY,
        base_url=config.OPENROUTER_BASE_URL,
    ),
    tools=[document_lookup],
    instructions=[
        "You are a document retrieval assistant.",
        "Always call document_lookup before answering any question.",
        "Always show the retrieved chunks (source file, similarity, content) in your response before giving the final answer.",
        "Never answer document-related questions from memory alone — always ground your answer in the retrieved context.",
    ],
    markdown=True,
)
