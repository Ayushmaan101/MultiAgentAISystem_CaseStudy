import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agno.agent import Agent
from agno.models.openai.like import OpenAILike

import config
from agents.rag_agent import rag_agent
from agents.calculator_agent import calculator_agent
from agents.web_search_agent import web_search_agent


def _route_to_rag(query: str) -> str:
    """Route query to the RAG Agent for document/knowledge-base lookups."""
    response = rag_agent.run(query)
    return response.content if hasattr(response, "content") else str(response)


def _route_to_calculator(query: str) -> str:
    """Route query to the Calculator Agent for math and numerical computations."""
    response = calculator_agent.run(query)
    return response.content if hasattr(response, "content") else str(response)


def _route_to_web_search(query: str) -> str:
    """Route query to the Web Search Agent for current or external information."""
    response = web_search_agent.run(query)
    return response.content if hasattr(response, "content") else str(response)


coordinator = Agent(
    name="Coordinator",
    model=OpenAILike(
        id=config.LLM_MODEL,
        api_key=config.OPENROUTER_API_KEY,
        base_url=config.OPENROUTER_BASE_URL,
    ),
    tools=[_route_to_rag, _route_to_calculator, _route_to_web_search],
    instructions=[
        "You are the Coordinator of a multi-agent AI research assistant.",
        "For every user query you must:",
        "  1. Classify the intent and state which agent you are routing to.",
        "  2. Call exactly one of the routing tools and return its full response.",
        "  3. Never answer directly without routing through a sub-agent.",
        "",
        "Routing rules (apply the first that matches):",
        "  - Query is about documents, files, or knowledge-base content → call _route_to_rag",
        "  - Query contains math, numbers, formulas, or calculations → call _route_to_calculator",
        "  - Query needs current events, live data, or external information → call _route_to_web_search",
        "  - Uncertain → call _route_to_rag as the safe default",
    ],
    markdown=True,
)
