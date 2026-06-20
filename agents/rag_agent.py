import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import httpx
from agno.agent import Agent
from agno.models.groq import Groq

import config
import database

_http_client = httpx.Client(verify=False)


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
    model=Groq(
        id=config.GROQ_MODEL,
        api_key=config.GROQ_API_KEY,
        http_client=_http_client,
    ),
    tools=[document_lookup],
    instructions=[
        "You are a document retrieval assistant.",
        "Always call document_lookup before answering — never rely on memory.",
        "Show each retrieved chunk (source file, similarity score, content excerpt) before giving your answer.",
    ],
    markdown=True,
)
