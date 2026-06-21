"""
Agno UI entry point for the AI Research Assistant.

agno.playground.Playground does not exist in Agno 2.6.18.
The correct equivalent is AgentOS from agno.os.app, which:
  - Serves a FastAPI backend on port 7777
  - Exposes all registered agents via the Agno REST protocol
  - Is compatible with https://app.agno.com (CORS already allows that origin by default)

Run with:
    python app.py

Then open https://app.agno.com and point the endpoint at http://localhost:7777

Architecture note
-----------------
Routing is single-layer: Ollama phi3.5 classifies, Python dispatches to the
correct sub-agent directly.  There is no coordinator LLM.

Sub-agents (rag_agent, calculator_agent, web_search_agent) are defined in
agents/*.py and use llama-3.3-70b-versatile (Groq) via get_fallback_model().
They are imported here and registered directly in AgentOS — no duplicate
instances.

The coordinator in AgentOS wraps run_coordinator() as a single Python tool.
The only LLM call in the coordinator path is one tool-dispatch call in Agno
(llama-3.3-70b-versatile, single-tool → always calls route_query).
"""

from agno.agent import Agent
from agno.os.app import AgentOS

from agents.rag_agent import rag_agent
from agents.calculator_agent import calculator_agent
from agents.web_search_agent import web_search_agent
from agents.coordinator import run_coordinator
from llm_client import get_fallback_model, get_coordinator_model
from database import init_db

# Initialize the local DuckDB on startup (idempotent).
init_db()


# ── Coordinator tool ───────────────────────────────────────────────────────────
# Ollama phi3.5 classifies the query, Python dispatches to the right sub-agent.
# The sub-agent's full response is returned verbatim.

def route_query(query: str) -> str:
    """Retrieve the answer for this query from the backend pipeline."""
    routed_to, response, classification, routing_method = run_coordinator(query)
    return f"[Routed to {routed_to} via {routing_method} | {classification}]\n\n{response}"


coordinator = Agent(
    name="Coordinator",
    model=get_coordinator_model(),
    tools=[route_query],
    tool_call_limit=1,
    instructions=[
        "You are a stateless dispatcher. You have no knowledge and no memory.",
        "You CANNOT answer questions. You can ONLY call route_query to retrieve answers.",
        "You MUST call route_query for every single query — math, facts, anything.",
        "Step 1: Call route_query with the user query.",
        "Step 2: Return the tool result exactly as-is.",
    ],
    markdown=True,
)


# ── AgentOS app ────────────────────────────────────────────────────────────────

agent_os = AgentOS(
    name="AI Research Assistant",
    description="Multi-agent assistant: RAG, Calculator, Web Search, and Coordinator",
    agents=[coordinator, rag_agent, calculator_agent, web_search_agent],
    # Disable cloud DB auto-provisioning — this project uses its own DuckDB.
    auto_provision_dbs=False,
)

# FastAPI application instance (used by uvicorn when imported as a module).
app = agent_os.get_app()

if __name__ == "__main__":
    agent_os.serve("app:app", reload=False)
