import sys
import os
import re

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import config
import database
from agents.calculator_agent import safe_calculate
from agents.web_search_agent import web_search

# Shared sync httpx client for Node 5 Groq synthesis call (SSL bypass)
_http = httpx.Client(verify=False, timeout=30.0)


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 1 — classify
# Single job: classify query as RAG, CALCULATOR, or SEARCH (one word output)
# ═══════════════════════════════════════════════════════════════════════════════

_NODE1_SYSTEM = """You are a query classifier for an AI Research Assistant.
Your ONLY job is to output exactly one word: RAG, CALCULATOR, SEARCH, or MULTI.
No explanation. No punctuation. No other text. One word only.

The internal Knowledge Base contains:
- AI system architecture documentation
- RAG pipeline design and components
- Agent orchestration and coordination logic
- Embedding models and vector storage details
- Project-specific technical implementation details

Classification rules:
RAG — query asks about anything in the internal knowledge base above.
      Technical architectures, pipeline design, project details,
      implementation specifics, any domain-specific AI system question.

CALCULATOR — query requires computing a precise numeric result
             from a mathematical operation, even in plain language.

SEARCH — query asks about general world knowledge, current events,
         people, places, or facts completely unrelated to AI systems
         and agents. ONLY choose SEARCH if clearly not in the KB.

MULTI — query clearly needs MORE THAN ONE tool to answer completely.
        Contains both a document question AND math, or RAG AND search.
        Only use MULTI when two distinct tool types are obviously needed.
        When in doubt about MULTI, classify as the dominant single intent.

When in doubt between RAG and SEARCH, always choose RAG.

Examples:
'What embedding model is used?' → RAG
'Tell me about the architecture' → RAG
'What are the pipeline components?' → RAG
'What is the retrieval layer?' → RAG
'three dozen eggs use half' → CALCULATOR
'15 percent of 240' → CALCULATOR
'square root of 256' → CALCULATOR
'When was the Eiffel Tower built?' → SEARCH
'Who is the current CEO of OpenAI?' → SEARCH
'What is the capital of France?' → SEARCH
'What does the doc say about chunking AND what is 500/6?' → MULTI
'Search for RAG systems and also calculate 15% of 240' → MULTI
'What is the embedding model and who invented the internet?' → MULTI"""


def _classify(query: str) -> tuple[str, str]:
    """
    Node 1 — Call Ollama phi3.5 to classify intent.
    Single job: return exactly one of RAG, CALCULATOR, SEARCH.
    Returns (classification, routing_method).
    """
    try:
        resp = httpx.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": config.OLLAMA_ROUTING_MODEL,
                "prompt": f"Query: {query}",
                "keep_alive": config.OLLAMA_KEEP_ALIVE,
                "stream": False,
                "system": _NODE1_SYSTEM,
            },
            timeout=20.0,
            verify=False,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
        # Strip chain-of-thought blocks
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        # Extract first word and uppercase it
        first_word = raw.split()[0].upper().rstrip(".,;:") if raw.split() else ""
        if first_word in ("RAG", "CALCULATOR", "SEARCH", "MULTI"):
            print(f"[Node 1] Classification: {first_word}")
            return first_word, "ollama_llm"
        # Not a valid label — default to RAG
        print(f"[Node 1] Unexpected response '{raw[:60]}', defaulting to RAG")
        return "RAG", "keyword_fallback"
    except Exception as exc:
        print(f"[Node 1] Ollama unreachable ({exc}), defaulting to RAG")
        return "RAG", "keyword_fallback"


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 2 — similarity safety net
# Only called when Node 1 returns SEARCH.
# Single job: check KB relevance and override to RAG if warranted.
# ═══════════════════════════════════════════════════════════════════════════════

def _similarity_check(query: str) -> bool:
    """
    Node 2 — Check whether the KB contains content relevant to this query.
    Returns True if the top similarity score exceeds SIMILARITY_THRESHOLD,
    indicating the query should be answered from the KB (override to RAG).
    """
    try:
        results = database.search_chunks(query, top_k=1)
        if not results:
            print("[Node 2] No KB results → Confirmed SEARCH")
            return False
        score = results[0]["similarity"]
        if score > config.SIMILARITY_THRESHOLD:
            print(f"[Node 2] Top similarity: {score:.4f} → Override to RAG")
            return True
        print(f"[Node 2] Top similarity: {score:.4f} → Confirmed SEARCH")
        return False
    except Exception as exc:
        print(f"[Node 2] Similarity check failed ({exc}) → Confirmed SEARCH")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 3 — rewrite
# Single job: rewrite query for the target tool (Ollama phi3.5, already hot)
# ═══════════════════════════════════════════════════════════════════════════════

_NODE3_SYSTEMS = {
    "CALCULATOR": """You are a mathematical expression extractor.
Your ONLY job is to convert the user's question into a clean math
expression that Python's asteval can evaluate directly.
Output ONLY the expression. No explanation. No text. Just the expression.

Rules:
- Convert word numbers to digits
- Use standard operators: + - * / ** sqrt()
- dozen = 12, percent = /100, half = /2, quarter = /4

Examples:
'three dozen eggs use half' → (3 * 12) / 2
'15 percent of 240' → 0.15 * 240
'square root of 256' → sqrt(256)
'two dozen plus fifteen' → (2 * 12) + 15""",

    "RAG": """You are a search query optimizer.
Your ONLY job is to convert the user's question into a clean
keyword search query for a vector database.
Output ONLY the search query. No explanation. No text.
Remove conversational filler. Keep domain-specific terms.

Examples:
'tell me about the architecture' → system architecture components design
'what embedding model is used' → embedding model vector generation
'summarize the retrieval pipeline' → retrieval pipeline components stages
'what was said about chunking' → chunking strategy text splitting method""",

    "SEARCH": """You are a web search query optimizer.
Your ONLY job is to convert the user's question into a clean
factual search query for web search.
Output ONLY the search query. No explanation. No text.

Examples:
'when was eiffel tower built' → Eiffel Tower construction date
'who invented the internet' → who invented the internet
'current CEO of OpenAI' → OpenAI CEO 2025""",
}


def _rewrite(query: str, classification: str) -> str:
    """
    Node 3 — Rewrite query for the target tool via Ollama phi3.5.
    Model is already hot from Node 1 (keep_alive=5m).
    Single job: return the rewritten query string only.
    """
    system = _NODE3_SYSTEMS.get(classification, _NODE3_SYSTEMS["RAG"])
    try:
        resp = httpx.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": config.OLLAMA_ROUTING_MODEL,
                "prompt": f"Convert this query: {query}",
                "keep_alive": config.OLLAMA_KEEP_ALIVE,
                "stream": False,
                "system": system,
            },
            timeout=20.0,
            verify=False,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
        # Strip chain-of-thought blocks
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        # Take first non-empty line only
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        first_line = lines[0] if lines else query
        # Strip leading arrows or punctuation sometimes prepended by the model
        first_line = re.sub(r"^[→\->\s'\"]+", "", first_line).strip()
        result = first_line if first_line else query
        print(f"[Node 3] Rewritten query: {result}")
        return result
    except Exception as exc:
        print(f"[Node 3] Rewrite failed ({exc}), using original query")
        return query


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 4 — execute tool
# Pure Python. Zero LLM. Single job: call the correct tool with rewritten query.
# ═══════════════════════════════════════════════════════════════════════════════

def _execute_tool(classification: str, rewritten_query: str) -> str:
    """
    Node 4 — Direct tool execution with no LLM involvement.
    Calls safe_calculate / search_chunks / web_search directly.
    """
    if classification == "CALCULATOR":
        try:
            result = safe_calculate(rewritten_query)
            print(f"[Node 4] Calculator result: {result.split(chr(10))[-1]}")
            return result
        except Exception as exc:
            return f"Expression: {rewritten_query}\nError: {exc}"

    if classification == "SEARCH":
        try:
            result = web_search(rewritten_query)
            n = result.count("[Result ")
            print(f"[Node 4] Web search returned {n} results")
            return result
        except Exception as exc:
            return f"=== WEB SEARCH RESULTS ===\n\nError: {exc}\n\n=== END SEARCH RESULTS ==="

    # RAG (default)
    try:
        results = database.search_chunks(rewritten_query, config.TOP_K)
        if not results:
            print("[Node 4] Retrieved 0 chunks")
            return "=== RETRIEVED CHUNKS ===\n\nNo chunks found for this query.\n\n=== END RETRIEVED CHUNKS ==="
        lines = ["=== RETRIEVED CHUNKS ===\n"]
        for i, r in enumerate(results, 1):
            lines.append(
                f"[Chunk {i}] Source: {r['source_file']} | Similarity: {r['similarity']:.6f}"
            )
            lines.append(f"Content: {r['content']}")
            lines.append("")
        lines.append("=== END RETRIEVED CHUNKS ===")
        raw = "\n".join(lines)
        print(
            f"[Node 4] Retrieved {len(results)} chunks, "
            f"top similarity: {results[0]['similarity']:.4f}"
        )
        return raw
    except Exception as exc:
        return f"=== RETRIEVED CHUNKS ===\n\nError: {exc}\n\n=== END RETRIEVED CHUNKS ==="


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 5 — synthesize
# Single job: write the final answer from the tool result only.
# Uses Groq qwen3-32b via direct httpx (config.QWEN_MODEL / get_synthesis_model).
# ═══════════════════════════════════════════════════════════════════════════════

_NODE5_SYSTEM = """You are a synthesis engine for an AI Research Assistant.
Your ONLY job is to write a clean, accurate final answer based
strictly on the tool result provided to you.

Rules:
- Use ONLY information from the tool result
- Never add information from your own training data
- For RAG: cite which chunk number and source file
- For CALCULATOR: state the expression and result clearly
- For SEARCH: cite source URLs
- Be concise and direct
- If tool result is empty or irrelevant, say so honestly"""


def _synthesize(query: str, classification: str, tool_result: str) -> str:
    """
    Node 5 — Synthesize final answer via Groq qwen3-32b.
    Called from coordinator.py; uses direct httpx POST to Groq API.
    get_synthesis_model() in llm_client.py defines this model configuration.
    """
    try:
        resp = _http.post(
            f"{config.GROQ_BASE_URL}/chat/completions",
            json={
                "model": config.QWEN_MODEL,
                "messages": [
                    {"role": "system", "content": _NODE5_SYSTEM},
                    {
                        "role": "user",
                        "content": (
                            f"Original query: {query}\n\n"
                            f"Tool result:\n{tool_result}\n\n"
                            "Write the final answer."
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
        # Strip any chain-of-thought blocks qwen3 might prepend
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        print("[Node 5] Synthesis complete")
        return content
    except Exception as exc:
        print(f"[Node 5] Synthesis failed ({exc}), returning raw result")
        return tool_result


# ═══════════════════════════════════════════════════════════════════════════════
# run_coordinator — orchestrate all five nodes
# ═══════════════════════════════════════════════════════════════════════════════

_ROUTED_TO = {
    "CALCULATOR": "Calculator Agent",
    "SEARCH": "Web Search Agent",
    "RAG": "RAG Agent",
}


def run_coordinator(query: str) -> tuple[str, str, str, str, str, str]:
    """
    Orchestrate the five-node pipeline:

    Node 1: classify (one Ollama call, keep_alive=5m)
    Node 2: similarity safety net (only when Node 1 → SEARCH)
    Node 3: rewrite (one Ollama call, model already hot)
    Node 4: direct tool execution (zero LLM)
    Node 5: qwen3-32b synthesis (one Groq call)

    Returns
    -------
    tuple[str, str, str, str, str, str]
        (routed_to, raw_tool_result, final_answer,
         classification, rewritten_query, routing_method)
    """
    # Node 1: classify
    classification, routing_method = _classify(query)

    # Node 2: similarity safety net (only if SEARCH, skip for MULTI)
    if classification == "SEARCH":
        override = _similarity_check(query)
        if override:
            classification = "RAG"
            routing_method = "similarity_override"
            print("[Coordinator] SEARCH overridden to RAG by similarity check")
    elif classification == "MULTI":
        pass  # No similarity check for MULTI queries

    # MULTI placeholder — tracker agent built in Prompt 9D
    if classification == "MULTI":
        print("[Coordinator] MULTI query detected — tracker agent pending")
        rewritten_query = query
        tool_result = "MULTI routing pending tracker agent implementation"
        final_answer = tool_result
        return "Multi-Agent (pending)", tool_result, final_answer, classification, rewritten_query, routing_method

    # Node 3: rewrite
    rewritten_query = _rewrite(query, classification)

    # Node 4: execute tool
    tool_result = _execute_tool(classification, rewritten_query)

    # Node 5: synthesize
    final_answer = _synthesize(query, classification, tool_result)

    routed_to = _ROUTED_TO.get(classification, "RAG Agent")
    return routed_to, tool_result, final_answer, classification, rewritten_query, routing_method
