"""
Agno UI entry point for the AI Research Assistant.

Run with:
    python app.py

Then open https://app.agno.com and point the endpoint at http://localhost:7777

Architecture
------------
Single coordinator Agent registered in AgentOS.

The full five-node pipeline executes inside the route_query tool:
    Node 1: Ollama phi3.5 classify (keep_alive=5m)
    Node 2: Python similarity safety net (SEARCH override)
    Node 3: Ollama phi3.5 rewrite (already hot)
    Node 4: Python direct tool execution (zero LLM)
    Node 5: qwen3-32b synthesis (Groq, httpx)

The coordinator Agent shell uses openai/gpt-oss-20b — the only Groq model
confirmed to always emit JSON function calls without answering from memory.
qwen3-32b is used exclusively for Node 5 synthesis (no tool calling needed).

route_query returns a structured string:
    === RAW TOOL RESULT ===     ← visible in AgentOS tools dropdown
    ...
    === SYNTHESIZED ANSWER ===  ← coordinator Agent extracts and returns this
"""

import httpx
from openai import AsyncOpenAI as _AsyncOpenAI

from agno.agent import Agent
from agno.models.openai.like import OpenAILike
from agno.os.app import AgentOS

from agents.coordinator import run_coordinator
from database import init_db
import config

# Initialize the local DuckDB on startup (idempotent).
init_db()

# ── Coordinator shell model ────────────────────────────────────────────────────
# gpt-oss-20b: OpenAI-architecture, always emits JSON tool calls.
# get_synthesis_model() returns qwen3-32b (for Node 5 only, not for tool calling).

_async_http = httpx.AsyncClient(verify=False)
_sync_http = httpx.Client(verify=False)

_coordinator_model = OpenAILike(
    id=config.GROQ_MODEL,                    # "openai/gpt-oss-20b"
    api_key=config.GROQ_API_KEY,
    base_url=config.GROQ_BASE_URL,
    http_client=_sync_http,
    async_client=_AsyncOpenAI(
        api_key=config.GROQ_API_KEY,
        base_url=config.GROQ_BASE_URL,
        http_client=_async_http,
    ),
)


# ── Coordinator tool ───────────────────────────────────────────────────────────

def route_query(query: str) -> str:
    """
    Run the full five-node pipeline and return structured output.

    The return string has two labelled sections:
    - === RAW TOOL RESULT ===      shown in the AgentOS tools dropdown
    - === SYNTHESIZED ANSWER ===   extracted by the coordinator Agent for chat
    """
    routed_to, raw_tool_result, final_answer, classification, rewritten_query, routing_method = (
        run_coordinator(query)
    )
    return (
        f"[Routed to {routed_to} via {routing_method} | {classification}]\n"
        f"Rewritten query: \"{rewritten_query}\"\n\n"
        f"=== RAW TOOL RESULT ===\n"
        f"{raw_tool_result}\n"
        f"=== END RAW TOOL RESULT ===\n\n"
        f"=== SYNTHESIZED ANSWER ===\n"
        f"{final_answer}\n"
        f"=== END SYNTHESIZED ANSWER ==="
    )


# ── Coordinator Agent ──────────────────────────────────────────────────────────

coordinator = Agent(
    name="Coordinator",
    model=_coordinator_model,
    tools=[route_query],
    tool_call_limit=1,
    instructions=[
        "You are a stateless query dispatcher. You have no knowledge and no memory.",
        "You CANNOT answer questions. You MUST call route_query for EVERY query — no exceptions.",
        "Step 1: Call route_query with the user's query exactly as given.",
        "Step 2: From the tool result, find the section between "
        "'=== SYNTHESIZED ANSWER ===' and '=== END SYNTHESIZED ANSWER ==='.",
        "Step 3: Return ONLY that section content, verbatim. Do not add any other text.",
    ],
    markdown=True,
)


# ── AgentOS app ────────────────────────────────────────────────────────────────

agent_os = AgentOS(
    name="AI Research Assistant",
    description="Multi-agent assistant with five-node pipeline: classify, safety-net, rewrite, execute, synthesize",
    agents=[coordinator],
    auto_provision_dbs=False,
)

app = agent_os.get_app()

if __name__ == "__main__":
    agent_os.serve("app:app", reload=False)
