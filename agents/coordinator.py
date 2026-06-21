import sys
import os
import re
import json

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import config
import database
from agents.calculator_agent import safe_calculate
from agents.web_search_agent import web_search

# Shared sync httpx client (corporate proxy SSL bypass)
_http = httpx.Client(verify=False, timeout=30.0)


# ── Ollama: classify + rewrite in one call ─────────────────────────────────────

_OLLAMA_SYSTEM_PROMPT = """You are a query processor for an AI Research Assistant.
Given a user query, you must output a JSON object with exactly two fields:
- classification: exactly one of RAG, CALCULATOR, or SEARCH
- rewritten_query: the query rewritten for the target agent

Classification rules:
- RAG: the user asks about THIS specific system, project, or uploaded documents.
  Any question containing 'this system', 'this project', 'the architecture',
  'the document', 'what was said', 'what does the file say', 'in the paper',
  'used in this', or asking about internal technical details (embedding model,
  chunking strategy, retrieval method, vector database) → RAG.

- CALCULATOR: the user wants a numeric result from a math operation.

- SEARCH: the user wants general world knowledge (people, places, events,
  history) that is NOT about this specific project or system.

Rewriting rules:
- If RAG: rephrase as a concise search query. Remove conversational filler.
  Example: 'tell me about the architecture' -> 'system architecture components and design'
  Example: 'what was said about retrieval' -> 'retrieval pipeline design and components'
  Example: 'what embedding model is used in this system' -> 'embedding model used in system'
  Example: 'tell me more about the architecture' -> 'system architecture components and design'

- If CALCULATOR: extract ONLY the math expression asteval can evaluate. Convert word numbers.
  Example: 'three dozen eggs use half' -> '(3 * 12) / 2'
  Example: 'square root of 256' -> 'sqrt(256)'
  Example: '15 percent of 240' -> '0.15 * 240'

- If SEARCH: rephrase as a clean factual question.
  Example: 'when was eiffel tower built' -> 'When was the Eiffel Tower constructed?'
  Example: 'who invented internet' -> 'Who invented the internet?'

Respond with ONLY valid JSON. No explanation. No markdown. No extra text.
Example output:
{"classification": "CALCULATOR", "rewritten_query": "(3 * 12) / 2"}"""


def _classify_and_rewrite(query: str) -> tuple[str, str, str]:
    """
    Call Ollama phi3.5 once to both classify intent and rewrite the query for
    the target tool.

    Returns (classification, rewritten_query, routing_method).
    Falls back to (_keyword_classify, original query, 'keyword_fallback') on
    any failure.
    """
    try:
        payload = {
            "model": config.OLLAMA_ROUTING_MODEL,
            "prompt": f"System: {_OLLAMA_SYSTEM_PROMPT}\n\nUser query: {query}\n\nJSON output:",
            "stream": False,
        }
        resp = httpx.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=20.0,
            verify=False,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()

        # Strip <think>...</think> blocks (chain-of-thought models)
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        # Strip markdown code fences
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

        # Try direct JSON parse first
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Extract the first {...} block
            m = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
            if not m:
                raise ValueError(f"No JSON object in Ollama response: {raw[:200]}")
            parsed = json.loads(m.group())

        classification = str(parsed.get("classification", "")).strip().upper()
        rewritten_query = str(parsed.get("rewritten_query", query)).strip()

        if classification not in ("RAG", "CALCULATOR", "SEARCH"):
            raise ValueError(f"Invalid classification: {classification!r}")

        print(f"[coordinator] Ollama classified: {classification} | Rewritten: {rewritten_query}")
        return classification, rewritten_query, "ollama_llm"

    except Exception as exc:
        print(f"[coordinator] Ollama classify+rewrite failed ({exc}), using keyword fallback")
        return _keyword_classify(query), query, "keyword_fallback"


# ── Keyword fallback classifier ────────────────────────────────────────────────

_MATH_RE = re.compile(
    r"\b(calculat|comput|arithmeti|divid|multipl|percent|sqrt|logarithm|integrat|derivativ)\b"
    r"|what\s+is\s+\d"
    r"|\d+\s*[+\-*/]\s*\d"
    r"|\b(sum|minus|plus|times|divided\s+by|added|subtracted|multiplied|squared|dozen|remain|eggs)\b",
    re.IGNORECASE,
)
_WEB_RE = re.compile(
    r"\b(current|today|latest|news|recent|live|stock|weather|price|trending|2025|2026"
    r"|invented|who\s+is|who\s+was|who\s+created|who\s+made|who\s+built|history\s+of)\b",
    re.IGNORECASE,
)


def _keyword_classify(query: str) -> str:
    """Deterministic keyword-based fallback classification."""
    if _MATH_RE.search(query):
        return "CALCULATOR"
    if _WEB_RE.search(query):
        return "SEARCH"
    return "RAG"


# ── qwen3-32b synthesis ────────────────────────────────────────────────────────

_SYNTHESIS_SYSTEM = (
    "You are a synthesis engine. You receive raw tool results and write "
    "a clean, accurate, well-structured final answer for the user.\n"
    "Rules:\n"
    "- For RAG: cite which chunk number and source file your answer comes from\n"
    "- For CALCULATOR: state the expression and numeric result clearly\n"
    "- For SEARCH: cite the source URLs\n"
    "- Never make up information not present in the tool result\n"
    "- Be concise and direct"
)


def _synthesize(raw_result: str, classification: str, original_query: str) -> str:
    """
    Call qwen3-32b via the Groq OpenAI-compatible API to synthesize a final
    answer from the raw tool result.  Uses httpx directly (no Agno overhead).
    Falls back to returning raw_result if the API call fails.
    """
    try:
        resp = _http.post(
            f"{config.GROQ_BASE_URL}/chat/completions",
            json={
                "model": config.QWEN_MODEL,
                "messages": [
                    {"role": "system", "content": _SYNTHESIS_SYSTEM},
                    {
                        "role": "user",
                        "content": (
                            f"User question: {original_query}\n\n"
                            f"Tool result:\n{raw_result}\n\n"
                            "Write the final answer for the user."
                        ),
                    },
                ],
            },
            headers={
                "Authorization": f"Bearer {config.GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        # Strip any <think>...</think> blocks qwen3 might prepend
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return content
    except Exception as exc:
        print(f"[coordinator] qwen3-32b synthesis failed: {exc}")
        return raw_result


# ── Coordinator — Ollama classify+rewrite → direct tool call → synthesis ───────

def run_coordinator(query: str) -> tuple[str, str, str, str, str, str]:
    """
    Full pipeline:
      1. Ollama phi3.5 classifies intent AND rewrites the query in one call.
      2. Python dispatches to the correct tool function directly (no Agno Agent).
      3. Tool result is formatted as the raw_tool_result.
      4. qwen3-32b synthesizes a clean final_answer from raw_tool_result.

    Returns
    -------
    tuple[str, str, str, str, str, str]
        (routed_to, raw_tool_result, final_answer, classification,
         rewritten_query, routing_method)
    """
    classification, rewritten_query, routing_method = _classify_and_rewrite(query)

    # ── CALCULATOR ──────────────────────────────────────────────────────────────
    if classification == "CALCULATOR":
        try:
            raw = safe_calculate(rewritten_query)
        except Exception as exc:
            raw = f"Expression: {rewritten_query}\nError: {exc}"
        routed_to = "Calculator Agent"

    # ── SEARCH ──────────────────────────────────────────────────────────────────
    elif classification == "SEARCH":
        try:
            raw = web_search(rewritten_query)
        except Exception as exc:
            raw = f"=== WEB SEARCH RESULTS ===\n\nError: {exc}\n\n=== END SEARCH RESULTS ==="
        routed_to = "Web Search Agent"

    # ── RAG (default) ───────────────────────────────────────────────────────────
    else:
        try:
            results = database.search_chunks(rewritten_query, config.TOP_K)
            if not results:
                raw = "=== RETRIEVED CHUNKS ===\n\nNo chunks found for this query.\n\n=== END RETRIEVED CHUNKS ==="
            else:
                lines = ["=== RETRIEVED CHUNKS ===\n"]
                for i, r in enumerate(results, 1):
                    lines.append(
                        f"[Chunk {i}] Source: {r['source_file']} | Similarity: {r['similarity']:.6f}"
                    )
                    lines.append(f"Content: {r['content']}")
                    lines.append("")
                lines.append("=== END RETRIEVED CHUNKS ===")
                raw = "\n".join(lines)
        except Exception as exc:
            raw = f"=== RETRIEVED CHUNKS ===\n\nError: {exc}\n\n=== END RETRIEVED CHUNKS ==="
        routed_to = "RAG Agent"

    final_answer = _synthesize(raw, classification, query)

    return routed_to, raw, final_answer, classification, rewritten_query, routing_method
