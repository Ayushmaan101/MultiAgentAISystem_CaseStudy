import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agno.agent import Agent

import config
import database
from llm_client import get_model


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
    model=get_model(),
    tools=[document_lookup],
    instructions=[
        "You are a document retrieval assistant.",
        "Always call document_lookup before answering — never rely on memory.",
        "Show each retrieved chunk (source file, similarity score, content excerpt) before giving your answer.",
    ],
    markdown=True,
)
