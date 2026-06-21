"""
Agno UI entry point for the AI Research Assistant.

Run with:
    python app.py

Then open https://app.agno.com and point the endpoint at http://localhost:7777

Architecture
------------
Single coordinator agent registered in AgentOS.  The full pipeline runs inside
the route_query tool function:

    Ollama phi3.5 (classify + rewrite)
        → Python direct tool call (safe_calculate / web_search / search_chunks)
        → qwen3-32b synthesis (Groq, called directly from coordinator.py)

route_query returns a structured string containing two labelled sections:

    === RAW TOOL RESULT ===        ← visible in the AgentOS tools dropdown
    ...
    === SYNTHESIZED ANSWER ===     ← coordinator Agent extracts and returns this

The coordinator Agent (qwen3-32b) is instructed to return ONLY the content
of the SYNTHESIZED ANSWER section, so the chat shows only the clean answer
while the dropdown shows the raw data.
"""

from agno.agent import Agent
from agno.os.app import AgentOS

from agents.coordinator import run_coordinator
from llm_client import get_synthesis_model
from database import init_db

# Initialize the local DuckDB on startup (idempotent).
init_db()


# ── Coordinator tool ───────────────────────────────────────────────────────────

def route_query(query: str) -> str:
    """
    Execute the full pipeline for this query and return structured output.

    The return value is intentionally structured with two labelled sections so
    that the AgentOS tools dropdown shows the raw tool result while the chat
    shows only the synthesized answer (extracted by the coordinator Agent).
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
    model=get_synthesis_model(),   # qwen3-32b
    tools=[route_query],
    tool_call_limit=1,
    instructions=[
        "You are a stateless query dispatcher. You have no knowledge and no memory.",
        "You CANNOT answer questions directly. You MUST call route_query for every query.",
        "Step 1: Call route_query with the user's query exactly as given.",
        "Step 2: From the tool result, find the section between "
        "'=== SYNTHESIZED ANSWER ===' and '=== END SYNTHESIZED ANSWER ==='.",
        "Step 3: Return ONLY that section content. Do not add any other text.",
    ],
    markdown=True,
)


# ── AgentOS app ────────────────────────────────────────────────────────────────

agent_os = AgentOS(
    name="AI Research Assistant",
    description="Multi-agent assistant: RAG, Calculator, Web Search via single coordinator",
    agents=[coordinator],
    auto_provision_dbs=False,
)

app = agent_os.get_app()

if __name__ == "__main__":
    agent_os.serve("app:app", reload=False)
