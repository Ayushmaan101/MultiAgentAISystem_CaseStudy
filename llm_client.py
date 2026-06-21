"""
LLM client factory for the AI Research Assistant.

Synthesis provider: Groq (qwen/qwen3-32b) — used ONLY for final answer synthesis
                    inside coordinator.py after tool execution.
Primary provider  : OpenRouter (Llama 3.3 70B) — available for general use.

Usage
-----
    from llm_client import get_synthesis_model

    # In coordinator Agent (app.py):
    Agent(model=get_synthesis_model(), ...)
"""

import httpx
from openai import AsyncOpenAI as _AsyncOpenAIClient
from agno.models.openai.like import OpenAILike

import config

# Sync httpx client — used by coordinator.py direct Groq API calls and Agno sync paths.
_http_client = httpx.Client(verify=False)

# Async httpx client — shared by SDK async clients (AgentOS / uvicorn path).
_async_http_client = httpx.AsyncClient(verify=False)


def get_model() -> OpenAILike:
    """
    Return an OpenAILike model instance pointed at OpenRouter (primary LLM provider).

    Passes both a sync and async httpx client so SSL verify=False works in both
    Agno's synchronous coordinator path and the async AgentOS / uvicorn path.
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


def get_synthesis_model() -> OpenAILike:
    """
    Return openai/gpt-oss-20b via Groq for the AgentOS coordinator shell.

    gpt-oss-20b is an OpenAI-architecture model that always emits JSON
    function-call format — never Hermes XML — regardless of whether the model
    knows the answer from training data.  qwen3-32b was found to answer
    "known" queries (e.g. 15% of 240) directly without calling route_query.

    qwen/qwen3-32b is used for synthesis via a direct httpx call inside
    coordinator.py (_synthesize).  This function provides the coordinator
    shell model only.
    """
    print("[llm_client] Coordinator shell model (gpt-oss-20b)")
    async_client = _AsyncOpenAIClient(
        api_key=config.GROQ_API_KEY,
        base_url=config.GROQ_BASE_URL,
        http_client=_async_http_client,
    )
    return OpenAILike(
        id=config.GROQ_MODEL,
        api_key=config.GROQ_API_KEY,
        base_url=config.GROQ_BASE_URL,
        http_client=_http_client,
        async_client=async_client,
    )
