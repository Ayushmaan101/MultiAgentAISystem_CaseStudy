import sys
import os
import re

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import config
from agents.rag_agent import rag_agent
from agents.calculator_agent import calculator_agent
from agents.web_search_agent import web_search_agent

# ── Ollama intent classifier ───────────────────────────────────────────────────

_OLLAMA_SYSTEM_PROMPT = """You are a query router for an AI Research Assistant.
Classify the user query into exactly one of three categories based on
what the user is ultimately trying to accomplish — not the words they use.

RAG - the user wants information from uploaded documents, research papers,
      or a private knowledge base. Summaries, explanations, or questions
      about specific stored content.

CALCULATOR - the user wants a precise numeric result from a mathematical
             operation, even if described in plain language.

SEARCH - the user wants factual information about the world, current events,
         people, places, or general knowledge that would not exist in a
         private document collection.

Examples:
'Who invented the internet?' → SEARCH
'What does the document say about chunking?' → RAG
'How much is 15% of 240?' → CALCULATOR
'Summarize the architecture section' → RAG
'What is the capital of France?' → SEARCH
'Calculate compound interest on 1000 at 5% for 3 years' → CALCULATOR
'When was the Eiffel Tower built?' → SEARCH
'What retrieval method is described in the file?' → RAG
'If I have three dozen eggs and use half, how many remain?' → CALCULATOR
'Who is the current US president?' → SEARCH

Respond with exactly one word: RAG, CALCULATOR, or SEARCH.
No explanation. No punctuation. No other text whatsoever."""


def _classify_with_ollama(query: str) -> tuple[str, str]:
    """
    Call Ollama /api/generate with phi3.5 to classify intent.
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
        for label in ("CALCULATOR", "SEARCH", "RAG"):
            if label in raw:
                print(f"[coordinator] Ollama classified '{query[:60]}' => {label}")
                return label, "ollama_llm"
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
    """Extract the best available text from a sub-agent RunResponse."""
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


# ── Coordinator — pure Python dispatch, no LLM ────────────────────────────────

def run_coordinator(query: str) -> tuple[str, str, str, str]:
    """
    Classify intent with Ollama (phi3.5) then dispatch directly to the
    appropriate sub-agent using Python.  No coordinator LLM involved.

    Classification:
        - Ollama /api/generate is called first with a strict system prompt
          that returns exactly RAG, CALCULATOR, or SEARCH.
        - If Ollama is unreachable or returns an unexpected label, keyword
          regex is used as deterministic fallback (routing_method = "keyword_fallback").

    Sub-agents (rag_agent, calculator_agent, web_search_agent) are called
    directly via agent.run() — each agent handles its own LLM calls.

    Returns
    -------
    tuple[str, str, str, str]
        (routed_to_label, response_text, classification, routing_method)
    """
    classification, routing_method = _classify_with_ollama(query)

    if classification == "CALCULATOR":
        try:
            result = _response_text(calculator_agent.run(query))
        except Exception as exc:
            result = f"Calculator agent error: {exc}"
        return "Calculator Agent", result, classification, routing_method

    if classification == "SEARCH":
        try:
            result = _response_text(web_search_agent.run(query))
        except Exception as exc:
            result = f"Web search agent error: {exc}"
        return "Web Search Agent", result, classification, routing_method

    # Default: RAG
    try:
        result = _response_text(rag_agent.run(query))
    except Exception as exc:
        result = f"RAG agent error: {exc}"
    return "RAG Agent", result, classification, routing_method
