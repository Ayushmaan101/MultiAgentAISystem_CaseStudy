import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agno.agent import Agent
from agno.models.openai.like import OpenAILike
from tavily import TavilyClient

import config


def web_search(query: str) -> dict:
    """Search the web via Tavily and return the top 3 results."""
    try:
        client = TavilyClient(api_key=config.TAVILY_API_KEY)
        response = client.search(query, max_results=3)
        results = [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
            }
            for r in response.get("results", [])
        ]
        return {"query": query, "results": results}
    except Exception as exc:
        return {"query": query, "error": str(exc)}


web_search_agent = Agent(
    name="Web Search Agent",
    model=OpenAILike(
        id=config.LLM_MODEL,
        api_key=config.OPENROUTER_API_KEY,
        base_url=config.OPENROUTER_BASE_URL,
    ),
    tools=[web_search],
    instructions=[
        "You are a web research assistant.",
        "Always call web_search before answering any question that requires current or external information.",
        "Never answer from memory when fresh web data is available.",
        "Cite all sources (title and URL) in your response.",
    ],
    markdown=True,
)
