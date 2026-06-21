import json
import re
import sys
import os

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

_SYSTEM_PROMPTS = {
    "CALCULATOR": """You are a query processor for a math tool.
Your job has two steps that happen as one transformation:
First: clean the input — fix typos, remove filler words like
'umm', 'can you', 'please', 'basically', normalise the text.
Second: extract the clean math expression for asteval evaluation.

Output ONLY the final math expression. No explanation. No text.
No equals sign. No units. Just the raw expression.

Rules:
- dozen = 12
- percent or % = /100
- half = /2, quarter = /4, third = /3
- Use operators: + - * / ** sqrt()
- Convert all word numbers to digits

Examples:
'umm what is three dozen eggs use half??' → (3 * 12) / 2
'can u calculate 15 percent of 240 pls' → 0.15 * 240
'whats the square root of 256' → sqrt(256)
'two dozen plus fifteen' → (2 * 12) + 15
'wat is 144 divided by 12 plus 37' → 144 / 12 + 37""",

    "RAG": """You are a search query processor for a vector database.
Your job has two steps that happen as one transformation:
First: clean the input — fix typos, remove filler words like
'umm', 'can you', 'tell me', 'basically', normalise the text.
Second: convert to a clean keyword search query for DuckDB
vector search. Keep all domain-specific technical terms intact.

Output ONLY the final search query. No explanation. No text.

Examples:
'umm tell me abt the chunking thing in this system??'
    → 'chunking strategy text splitting method'
'can u explain wat embedding model is used'
    → 'embedding model vector generation'
'wats the retrieval pipeline do basically'
    → 'retrieval pipeline components stages'
'tell me more about the architecture pls'
    → 'system architecture components design'
'what is the vss extension'
    → 'VSS extension vector similarity search DuckDB'
'what does the doc say about chunk overlap'
    → 'chunk overlap configuration value'""",

    "SEARCH": """You are a web search query processor.
Your job has two steps that happen as one transformation:
First: clean the input — fix typos, remove filler words,
normalise the text into a proper question.
Second: convert to a clean factual web search query.

Output ONLY the final search query. No explanation. No text.

Examples:
'umm when was eiffel tower built??'
    → 'Eiffel Tower construction date'
'who invented the internet basically'
    → 'who invented the internet'
'wats the current ceo of openai'
    → 'OpenAI CEO 2025'
'tell me about latest developments in ai'
    → 'latest AI developments 2025'""",

    "MULTI": """You are a multi-query processor.
Your job has two steps that happen as one transformation:
First: clean the input — fix typos, remove filler words,
normalise the text.
Second: split into exactly two clean sub-queries.

Return ONLY a valid JSON object with exactly this structure:
{
    "query_1": {"type": "RAG", "query": "clean search query"},
    "query_2": {"type": "CALCULATOR", "query": "math expression"}
}

The type field must be exactly RAG, CALCULATOR, or SEARCH.
No explanation. No markdown. No code blocks. Valid JSON only.

Examples:
Input: 'What does doc say about chunking AND what is 500 divided by 6?'
Output: {"query_1": {"type": "RAG", "query": "chunking strategy purpose"}, "query_2": {"type": "CALCULATOR", "query": "500 / 6"}}

Input: 'What is the embedding model and who invented the internet?'
Output: {"query_1": {"type": "RAG", "query": "embedding model vector generation"}, "query_2": {"type": "SEARCH", "query": "who invented the internet"}}""",
}

_http = httpx.Client(verify=False, timeout=30.0)


def rewrite_query(query: str, classification: str) -> "str | dict":
    """
    Node 3 — clean and rewrite the query for the target tool.

    For CALCULATOR / RAG / SEARCH: returns a rewritten string.
    For MULTI: returns a dict with query_1 and query_2 sub-queries.
    Falls back to the original query on any failure.
    """
    try:
        system = _SYSTEM_PROMPTS.get(classification, _SYSTEM_PROMPTS["RAG"])

        resp = _http.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": config.OLLAMA_ROUTING_MODEL,
                "prompt": f"Convert this query: {query}",
                "system": system,
                "keep_alive": config.OLLAMA_KEEP_ALIVE,
                "stream": False,
            },
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
        # Strip chain-of-thought blocks
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        if classification == "MULTI":
            # Extract the JSON object even if model adds surrounding text or code fences
            raw_clean = re.sub(r"```(?:json)?|```", "", raw).strip()
            json_match = re.search(r"\{.*\}", raw_clean, re.DOTALL)
            try:
                if not json_match:
                    raise ValueError("No JSON object found in response")
                parsed = json.loads(json_match.group(0))
                q1 = parsed.get("query_1", {})
                q2 = parsed.get("query_2", {})
                if q1.get("type") and q1.get("query") and q2.get("type") and q2.get("query"):
                    print(
                        f"[Query Rewriter] MULTI split: "
                        f"{q1['type']}: {q1['query']} | "
                        f"{q2['type']}: {q2['query']}"
                    )
                    return parsed
                raise ValueError("Missing required fields in parsed JSON")
            except (json.JSONDecodeError, AttributeError, ValueError) as exc:
                print(f"[Query Rewriter] MULTI JSON parse failed ({exc}), using fallback")
            # Fallback — return original query in both slots
            return {
                "query_1": {"type": "RAG", "query": query},
                "query_2": {"type": "SEARCH", "query": query},
            }

        # CALCULATOR / RAG / SEARCH — take first non-empty line
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        # Strip leading/trailing arrows or quote chars the model sometimes adds
        result = lines[0] if lines else query
        result = re.sub(r"^[→\->\s'\"]+", "", result).strip()
        result = re.sub(r"['\"\s]+$", "", result).strip()
        if not result:
            result = query

        print(
            f"[Query Rewriter] {classification} | "
            f"Original: {query} → Rewritten: {result}"
        )
        return result

    except Exception as exc:
        print(f"[Query Rewriter] Warning: rewrite failed ({exc}), returning original query")
        return query
