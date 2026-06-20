import sys
import os
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agno.agent import Agent

import config
from llm_client import get_model, get_fallback_model, _is_retryable, _is_error_response
from agents.rag_agent import rag_agent, document_lookup
from agents.calculator_agent import calculator_agent, safe_calculate
from agents.web_search_agent import web_search_agent, web_search

# ── Keyword patterns for deterministic routing ─────────────────────────────────
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
    """Extract the best available text from a sub-agent RunOutput."""
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


# ── Routing functions with OpenRouter primary and Groq fallback ────────────────

def _run_with_fallback(primary_agent, query: str, fallback_tools, fallback_instructions, agent_name: str) -> str:
    """
    Run primary_agent (OpenRouter). On retryable or error response, retry with a
    Groq-backed agent. If Groq also fails, call the first Python tool directly as a
    last resort so real data is always returned.

    Tier 1 — OpenRouter LLM (primary)
    Tier 2 — Groq LLM (native GroqModel class; better tool-call schema handling)
    Tier 3 — Direct Python tool call (no LLM; guarantees a real result)
    """
    try:
        text = _response_text(primary_agent.run(query))
    except Exception as exc:
        text = str(exc)

    if _is_retryable(Exception(text)) or _is_error_response(text):
        print(f"[llm_client] OpenRouter error for {agent_name}: {text[:120]}")
        fb = Agent(
            name=agent_name,
            model=get_fallback_model(),
            tools=fallback_tools,
            instructions=fallback_instructions,
            markdown=True,
        )
        try:
            fb_text = _response_text(fb.run(query))
        except Exception as exc2:
            fb_text = str(exc2)

        if _is_error_response(fb_text):
            # Tier 3: Groq also failed — call the Python tool directly
            tool_fn = fallback_tools[0]
            print(f"[llm_client] Groq fallback failed for {agent_name}, calling {tool_fn.__name__} directly")
            return str(tool_fn(query))

        return fb_text

    return text


def route_to_rag(query: str) -> str:
    """Route to RAG Agent (OpenRouter primary, Groq fallback on retryable error)."""
    return _run_with_fallback(
        rag_agent, query,
        fallback_tools=[document_lookup],
        fallback_instructions=[
            "You are a document retrieval assistant.",
            "Always call document_lookup before answering.",
            "Show each retrieved chunk before giving your answer.",
        ],
        agent_name="RAG Agent",
    )


def route_to_calculator(query: str) -> str:
    """Route to Calculator Agent (OpenRouter primary, Groq fallback on retryable error)."""
    return _run_with_fallback(
        calculator_agent, query,
        fallback_tools=[safe_calculate],
        fallback_instructions=[
            "You are a mathematical computation assistant.",
            "Always call safe_calculate — never compute yourself.",
            "State the expression and its numeric result.",
        ],
        agent_name="Calculator Agent",
    )


def route_to_web_search(query: str) -> str:
    """Route to Web Search Agent (OpenRouter primary, Groq fallback on retryable error)."""
    return _run_with_fallback(
        web_search_agent, query,
        fallback_tools=[web_search],
        fallback_instructions=[
            "You are a web research assistant.",
            "Always call web_search before answering.",
            "Cite all sources (title and URL).",
        ],
        agent_name="Web Search Agent",
    )


def run_coordinator(query: str) -> tuple:
    """
    Route a query to the appropriate sub-agent and return (agent_label, response).

    Deterministic keyword routing is used here for reliability and predictability.
    Regex patterns match the query intent (math, web, or document lookup) and
    dispatch to the correct sub-agent without an LLM routing call.
    Sub-agents remain fully LLM-powered: each uses OpenRouter as the primary
    provider and automatically falls back to Groq on rate-limit or quota errors.

    Returns
    -------
    tuple[str, str]
        (routed_to_label, response_text)
    """
    if _MATH_RE.search(query):
        return "Calculator Agent", route_to_calculator(query)
    if _WEB_RE.search(query):
        return "Web Search Agent", route_to_web_search(query)
    return "RAG Agent", route_to_rag(query)


# ── Coordinator Agent (kept for completeness; routing uses run_coordinator()) ──
coordinator = Agent(
    name="Coordinator",
    model=get_model(),
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
