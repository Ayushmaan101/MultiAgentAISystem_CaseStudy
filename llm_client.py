"""
LLM client factory for the AI Research Assistant.

Primary provider : OpenRouter (Llama 3.3 70B via meta-llama/llama-3.3-70b-instruct)
Fallback provider: Groq    (Llama 4 Scout via meta-llama/llama-4-scout-17b-16e-instruct)

Usage
-----
    from llm_client import get_model, get_fallback_model, _is_retryable

    # In agent constructors:
    Agent(model=get_model(), ...)

    # In routing functions when primary fails:
    except Exception as exc:
        if _is_retryable(exc):
            Agent(model=get_fallback_model(), ...)
"""

import httpx
from openai import AsyncOpenAI as _AsyncOpenAIClient
from groq import AsyncGroq as _AsyncGroqClient
from agno.models.openai.like import OpenAILike
from agno.models.groq import Groq as GroqModel

import config

# Sync httpx client — used by Agno's synchronous code path (coordinator fallback, direct tool calls).
_http_client = httpx.Client(verify=False)

# Async httpx client — shared by the SDK async clients below.
# A single AsyncClient instance is safe to share across requests.
_async_http_client = httpx.AsyncClient(verify=False)

# Error substrings that indicate a transient provider failure worth retrying.
_RETRYABLE_PATTERNS = (
    "429",
    "402",
    "502",
    "503",
    "rate limit",
    "rate-limit",
    "insufficient credits",
    "provider returned error",
    "decommissioned",
    "tool call validation failed",
    "not in request.tools",
    "failed to call a function",
    "connecttimeout",
    "connection",
    "timeout",
)


def _is_retryable(exc: Exception) -> bool:
    """Return True if *exc* looks like a transient provider failure worth retrying on Groq."""
    msg = str(exc).lower()
    return any(p in msg for p in _RETRYABLE_PATTERNS)


def _is_error_response(text: str) -> bool:
    """
    Return True if *text* is an error string returned by Agno instead of a real answer.

    Agno catches API exceptions internally and surfaces them as the agent's content
    string rather than re-raising.  This function detects those cases so routing
    functions can still trigger the Groq fallback.

    Also catches raw JSON tool-call debris that some Groq models emit when they
    generate a function-call in a non-standard format (e.g. ``{"name":"lookup",...}``).
    """
    t = text.lower().strip()
    if any(p in t for p in _RETRYABLE_PATTERNS):
        return True
    # Raw JSON tool-call fragment — not a real answer
    if t.startswith("{") and ('"name"' in t or "'name'" in t) and ('"parameters"' in t or '"arguments"' in t):
        return True
    return False


def get_model() -> OpenAILike:
    """
    Return an OpenAILike model instance pointed at OpenRouter (primary LLM provider).

    Passes both a sync and async httpx client so SSL verify=False works in both
    Agno's synchronous coordinator path and the async AgentOS / uvicorn path.
    Logs "Using OpenRouter" to stdout so the server logs show which provider is active.
    """
    print("[llm_client] Using OpenRouter")
    async_client = _AsyncOpenAIClient(
        api_key=config.OPENROUTER_API_KEY,
        base_url=config.OPENROUTER_BASE_URL,
        http_client=_async_http_client,
    )
    return OpenAILike(
        id=config.LLM_MODEL,
        api_key=config.OPENROUTER_API_KEY,
        base_url=config.OPENROUTER_BASE_URL,
        http_client=_http_client,
        async_client=async_client,
    )


def get_fallback_model() -> GroqModel:
    """
    Return a Groq model for single-tool agents (RAG, Calculator, Web Search).

    Uses llama-3.3-70b-versatile — strong instruction-following, reliable
    single-tool calling.  llama-4-scout uses Hermes XML format for multi-tool
    schemas, which Agno does not parse correctly.
    Logs "Falling back to Groq" to stdout.
    """
    print("[llm_client] Falling back to Groq")
    async_client = _AsyncGroqClient(
        api_key=config.GROQ_API_KEY,
        http_client=_async_http_client,
    )
    return GroqModel(
        id=config.GROQ_MODEL,
        api_key=config.GROQ_API_KEY,
        http_client=_http_client,
        async_client=async_client,
    )


def get_router_model() -> GroqModel:
    """
    Return a Groq model for the coordinator (multi-tool routing).

    Uses llama-4-scout-17b-16e-instruct — emits JSON tool-call format that
    Agno parses correctly when there are 3+ tools.  llama-3.3-70b-versatile
    switches to Hermes XML format for multi-tool schemas, breaking Agno's parser.
    Logs "Router model (Groq Scout)" to stdout.
    """
    print("[llm_client] Router model (Groq Scout)")
    async_client = _AsyncGroqClient(
        api_key=config.GROQ_API_KEY,
        http_client=_async_http_client,
    )
    return GroqModel(
        id=config.GROQ_ROUTER_MODEL,
        api_key=config.GROQ_API_KEY,
        http_client=_http_client,
        async_client=async_client,
    )
