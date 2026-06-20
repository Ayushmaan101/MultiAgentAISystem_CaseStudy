import sys
import os
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import httpx
from agno.agent import Agent
from agno.models.groq import Groq

import config
from agents.rag_agent import rag_agent
from agents.calculator_agent import calculator_agent
from agents.web_search_agent import web_search_agent

_http_client = httpx.Client(verify=False)

# Keyword patterns for deterministic routing
_MATH_RE = re.compile(
    r'\b(calculat|comput|arithmeti|divid|multipl|percent|sqrt|logarithm|integrat|derivativ)\b'
    r'|what\s+is\s+\d'
    r'|\d+\s*[+\-*/]\s*\d'
    r'|\b(sum|minus|plus|times|divided\s+by|added|subtracted|multiplied|squared)\b',
    re.IGNORECASE,
)
_WEB_RE = re.compile(
    r'\b(current|today|latest|news|recent|live|stock|weather|price|trending|2025|2026)\b',
    re.IGNORECASE,
)


def _response_text(run_output) -> str:
    content = getattr(run_output, "content", None) or ""
    if not isinstance(content, str):
        content = str(content)
    has_words = any(c.isalpha() for c in content)
    if has_words and content.strip():
        return content
    for t in (getattr(run_output, "tools", None) or []):
        if not getattr(t, "tool_call_error", True) and t.result:
            return str(t.result)
    return content


def route_to_rag(query: str) -> str:
    """Route query to the RAG Agent for document/knowledge-base lookups."""
    return _response_text(rag_agent.run(query))


def route_to_calculator(query: str) -> str:
    """Route query to the Calculator Agent for math and numerical computations."""
    return _response_text(calculator_agent.run(query))


def route_to_web_search(query: str) -> str:
    """Route query to the Web Search Agent for current or external information."""
    return _response_text(web_search_agent.run(query))


def run_coordinator(query: str) -> tuple:
    """Deterministic keyword router — calls the right sub-agent directly."""
    if _MATH_RE.search(query):
        return "Calculator Agent", route_to_calculator(query)
    if _WEB_RE.search(query):
        return "Web Search Agent", route_to_web_search(query)
    return "RAG Agent", route_to_rag(query)


# Agent object kept for completeness; routing is handled by run_coordinator().
coordinator = Agent(
    name="Coordinator",
    model=Groq(
        id=config.GROQ_MODEL,
        api_key=config.GROQ_API_KEY,
        http_client=_http_client,
    ),
    tools=[route_to_rag, route_to_calculator, route_to_web_search],
    tool_call_limit=1,
    instructions=[
        "You are a routing coordinator. ALWAYS call one of the routing tools.",
        "  - Math/numbers/calculations => call route_to_calculator",
        "  - Documents/knowledge base => call route_to_rag",
        "  - Current events/web => call route_to_web_search",
        "Never answer directly. Call a tool immediately.",
    ],
    markdown=True,
)
