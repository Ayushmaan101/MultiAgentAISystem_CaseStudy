import sys
import os
import re
import json

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agno.agent import Agent

import config
from llm_client import get_model, get_fallback_model, _is_retryable, _is_error_response
from agents.rag_agent import rag_agent, document_lookup
from agents.calculator_agent import calculator_agent, safe_calculate
from agents.web_search_agent import web_search_agent, web_search

# ── Ollama intent classifier ───────────────────────────────────────────────────

_OLLAMA_SYSTEM_PROMPT = """You are an intent classification system. Classify the user query into exactly one of these three categories:
- RAG: Questions about documents, papers, knowledge base content, architecture, system design, retrieval pipelines, or anything that requires looking up stored information.
- CALCULATOR: Mathematical computations, arithmetic, algebra, percentages, unit conversions, or any question that requires numeric calculation.
- SEARCH: Questions about current events, news, real-time data, people, places, recent facts, or anything requiring live web search.

Respond with ONLY the category label: RAG, CALCULATOR, or SEARCH. No explanation. No punctuation. Just the label."""


def _classify_with_ollama(query: str) -> tuple[str, str]:
    """
    Call Ollama /api/generate with llama3.2 to classify intent.
    Returns (classification, routing_method).
    classification is one of: RAG, CALCULATOR, SEARCH
    routing_method is "ollama_llm" on success or "keyword_fallback" on failure.
    """
    try:
        payload = {
            "model": config.OLLAMA_ROUTING_MODEL,
            "prompt": f"System: {_OLLAMA_SYSTEM_PROMPT}\n\nQuery: {query}",
            "stream": False,
        }
        resp = httpx.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=15.0,
            verify=False,
        )
        resp.raise_for_status()
        data = resp.json()
        raw = data.get("response", "").strip().upper()
        # Accept the label even if the model padded it slightly
        for label in ("CALCULATOR", "SEARCH", "RAG"):
            if label in raw:
                print(f"[coordinator] Ollama classified '{query[:60]}' => {label}")
                return label, "ollama_llm"
        # Model returned something unexpected — fall through to keyword
        print(f"[coordinator] Ollama returned unexpected label '{raw}', falling back to keyword routing")
        return _keyword_classify(query), "keyword_fallback"
    except Exception as exc:
        print(f"[coordinator] Ollama unreachable, falling back to keyword routing: {exc}")
        return _keyword_classify(query), "keyword_fallback"


# ── Keyword fallback classifier ────────────────────────────────────────────────

_MATH_RE = re.compile(
    r'\b(calculat|comput|arithmeti|divid|multipl|percent|sqrt|logarithm|integrat|derivativ)\b'
    r'|what\s+is\s+\d'
    r'|\d+\s*[+\-*/]\s*\d'
    r'|\b(sum|minus|plus|times|divided\s+by|added|subtracted|multiplied|squared|dozen|remain|eggs)\b',
    re.IGNORECASE,
)
_WEB_RE = re.compile(
    r'\b(current|today|latest|news|recent|live|stock|weather|price|trending|2025|2026|invented|who\s+is|who\s+was|who\s+created|who\s+made|who\s+built|history\s+of)\b',
    re.IGNORECASE,
)


def _keyword_classify(query: str) -> str:
    """Deterministic keyword-based fallback classification."""
    if _MATH_RE.search(query):
        return "CALCULATOR"
    if _WEB_RE.search(query):
        return "SEARCH"
    return "RAG"


# ── Response extraction ────────────────────────────────────────────────────────

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

    Tier 1 -- OpenRouter LLM (primary)
    Tier 2 -- Groq LLM (native GroqModel class; better tool-call schema handling)
    Tier 3 -- Direct Python tool call (no LLM; guarantees a real result)
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
            # Tier 3: Groq also failed -- call the Python tool directly
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
            "Always call safe_calculate -- never compute yourself.",
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
    Classify intent with Ollama (llama3.2) then route to the appropriate sub-agent.

    Classification:
        - Ollama /api/generate is called first with a strict system prompt that returns
          exactly RAG, CALCULATOR, or SEARCH.
        - If Ollama is unreachable or returns an unexpected label, keyword regex is used
          as a deterministic fallback (routing_method = "keyword_fallback").

    Sub-agents remain fully LLM-powered with OpenRouter primary and Groq fallback.

    Returns
    -------
    tuple[str, str, str, str]
        (routed_to_label, response_text, classification, routing_method)
    """
    classification, routing_method = _classify_with_ollama(query)

    if classification == "CALCULATOR":
        return "Calculator Agent", route_to_calculator(query), classification, routing_method
    if classification == "SEARCH":
        return "Web Search Agent", route_to_web_search(query), classification, routing_method
    return "RAG Agent", route_to_rag(query), classification, routing_method


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
