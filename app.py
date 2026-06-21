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

from agents.rag_agent import document_lookup as _document_lookup
from agents.calculator_agent import safe_calculate
from agents.web_search_agent import web_search
from llm_client import get_fallback_model, get_router_model


def document_lookup(query: str) -> str:
    """Search the local knowledge base for chunks relevant to the query."""
    return _document_lookup(query)


# Groq llama-4-scout-17b abbreviates "document_lookup" → "lookup" when calling tools.
# Register the same function under both names so either call succeeds.
def lookup(query: str) -> str:
    """Search the local knowledge base for chunks relevant to the query."""
    return _document_lookup(query)
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
    model=get_router_model(),
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
        "You MUST call the document_lookup tool for every query. Do NOT answer from memory.",
        "Step 1: Call document_lookup with the user's query.",
        "Step 2: Copy the entire === RETRIEVED CHUNKS === block verbatim into your response.",
        "Step 3: Write a synthesized answer citing chunk number and source file.",
    ],
    markdown=True,
)

calculator_agent = Agent(
    name="Calculator Agent",
    model=get_fallback_model(),
    tools=[safe_calculate],
    instructions=[
        "You are a mathematical computation assistant.",
        "You MUST call the safe_calculate tool for every math query. Do NOT compute yourself.",
        "Step 1: Call safe_calculate with the mathematical expression.",
        "Step 2: Copy the tool result verbatim into your response.",
        "Step 3: State the expression and its numeric result clearly.",
    ],
    markdown=True,
)

web_search_agent = Agent(
    name="Web Search Agent",
    model=get_fallback_model(),
    tools=[web_search],
    instructions=[
        "You are a web research assistant.",
        "You MUST call the web_search tool for every query. Do NOT answer from memory.",
        "Step 1: Call web_search with the user's query.",
        "Step 2: Copy the entire === WEB SEARCH RESULTS === block verbatim into your response.",
        "Step 3: Write a synthesized answer citing the title and URL from each relevant source.",
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
