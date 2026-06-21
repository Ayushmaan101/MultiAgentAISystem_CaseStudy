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
Agents here use Groq as primary model because OpenRouter free-tier is
rate-limited.  The main.py REST API continues to use OpenRouter+fallback
via run_coordinator().  Both paths share the same tool functions.
"""

from agno.agent import Agent
from agno.os.app import AgentOS

from agents.rag_agent import document_lookup
from agents.calculator_agent import safe_calculate
from agents.web_search_agent import web_search
from llm_client import get_fallback_model
from database import init_db

# Initialize the local DuckDB on startup (idempotent).
init_db()

# ── Routing helpers used by the coordinator ────────────────────────────────────
# These mirror the logic in coordinator.py but are defined here as simple
# pass-through functions so the coordinator agent can call them without
# triggering the full 3-tier fallback chain at agent-construction time.

def call_rag(query: str) -> str:
    """Look up the knowledge base and return retrieved document chunks."""
    from agents.coordinator import route_to_rag
    return route_to_rag(query)


def call_calculator(query: str) -> str:
    """Evaluate a mathematical expression and return the numeric result."""
    from agents.coordinator import route_to_calculator
    return route_to_calculator(query)


def call_web_search(query: str) -> str:
    """Search the web and return the top results with citations."""
    from agents.coordinator import route_to_web_search
    return route_to_web_search(query)


# ── Register all four agents ───────────────────────────────────────────────────

coordinator = Agent(
    name="Coordinator",
    model=get_fallback_model(),
    tools=[call_rag, call_calculator, call_web_search],
    tool_call_limit=1,
    instructions=[
        "You are a routing coordinator. ALWAYS call one of the routing tools.",
        "  - Math/numbers/calculations => call call_calculator",
        "  - Documents/knowledge base => call call_rag",
        "  - Current events/web => call call_web_search",
        "Never answer directly. Call a tool immediately.",
    ],
    markdown=True,
)

rag_agent = Agent(
    name="RAG Agent",
    model=get_fallback_model(),
    tools=[document_lookup],
    instructions=[
        "You are a document retrieval assistant.",
        "Always call document_lookup before answering — never rely on memory.",
        "Show each retrieved chunk (source file, similarity score, content excerpt) before giving your answer.",
    ],
    markdown=True,
)

calculator_agent = Agent(
    name="Calculator Agent",
    model=get_fallback_model(),
    tools=[safe_calculate],
    instructions=[
        "You are a mathematical computation assistant.",
        "Always call safe_calculate for every math query — never compute yourself.",
        "After the tool returns, clearly state the expression and its numeric result.",
    ],
    markdown=True,
)

web_search_agent = Agent(
    name="Web Search Agent",
    model=get_fallback_model(),
    tools=[web_search],
    instructions=[
        "You are a web research assistant.",
        "Always call web_search before answering any question that requires current or external information.",
        "Never answer from memory when fresh web data is available.",
        "Cite all sources (title and URL) in your response.",
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
