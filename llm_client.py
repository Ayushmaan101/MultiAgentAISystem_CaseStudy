"""
LLM client factory for the AI Research Assistant.

Synthesis provider: Groq qwen/qwen3-32b — used exclusively by Node 5 in
                    coordinator.py for final answer synthesis after tool
                    execution.  No tool calling involved — pure text generation.

Usage
-----
    from llm_client import get_synthesis_model

    # Coordinator shell in app.py:
    Agent(model=get_synthesis_model(), ...)
"""

import httpx
from openai import AsyncOpenAI as _AsyncOpenAIClient
from agno.models.openai.like import OpenAILike

import config

# Sync httpx client — used by coordinator.py direct Groq API calls (Node 5).
_http_client = httpx.Client(verify=False)

# Async httpx client — shared by SDK async clients (AgentOS / uvicorn path).
_async_http_client = httpx.AsyncClient(verify=False)


def get_synthesis_model() -> OpenAILike:
    """
    Return qwen/qwen3-32b via Groq, used exclusively by Node 5 synthesis.

    This model is called via direct httpx in coordinator.py (not as an Agno
    Agent tool) so tool-calling reliability is not a concern here — qwen3-32b
    is only used for text synthesis from raw tool results.

    NOTE: The AgentOS coordinator shell in app.py uses openai/gpt-oss-20b
    directly (hardcoded) because qwen3-32b answers 'known' queries without
    calling tools (same Hermes XML issue as Llama models).  gpt-oss-20b is
    the only Groq model confirmed to always emit JSON tool calls.
    """
    print("[llm_client] Synthesis model (qwen3-32b)")
    async_client = _AsyncOpenAIClient(
        api_key=config.GROQ_API_KEY,
        base_url=config.GROQ_BASE_URL,
        http_client=_async_http_client,
    )
    return OpenAILike(
        id=config.QWEN_MODEL,
        api_key=config.GROQ_API_KEY,
        base_url=config.GROQ_BASE_URL,
        http_client=_http_client,
        async_client=async_client,
    )
