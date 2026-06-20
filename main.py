from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

import database
from agents.coordinator import coordinator

# ── Tool-name → human-readable agent label ────────────────────────────────────
_TOOL_TO_AGENT = {
    "_route_to_rag": "RAG Agent",
    "_route_to_calculator": "Calculator Agent",
    "_route_to_web_search": "Web Search Agent",
}


def _detect_routed_agent(run_output) -> str:
    """Inspect tool calls on the RunOutput to identify which sub-agent was used."""
    tools = getattr(run_output, "tools", None) or []
    for t in tools:
        label = _TOOL_TO_AGENT.get(getattr(t, "tool_name", ""), None)
        if label:
            return label
    return "Unknown"


# ── Lifespan (startup) ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    database.init_db()
    print("Database initialized and ready")
    yield


app = FastAPI(
    title="AI Research Assistant",
    description="Multi-agent assistant: RAG, Calculator, Web Search via Agno + OpenRouter",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request / response schemas ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    query: str
    routed_to: str
    response: str
    status: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "message": "AI Research Assistant is running"}


@app.get("/chunks")
def get_chunks(limit: int = Query(default=10, ge=1, le=500)):
    conn = database.get_connection()
    rows = conn.execute(
        """
        SELECT id, source_file, chunk_index, timestamp
        FROM   document_chunks
        ORDER  BY timestamp DESC
        LIMIT  ?
        """,
        [limit],
    ).fetchall()
    return [
        {
            "id": r[0],
            "source_file": r[1],
            "chunk_index": r[2],
            "timestamp": str(r[3]),
        }
        for r in rows
    ]


@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    try:
        run_output = coordinator.run(request.query)
        routed_to = _detect_routed_agent(run_output)
        content = run_output.content or ""
        if not isinstance(content, str):
            content = str(content)
        return QueryResponse(
            query=request.query,
            routed_to=routed_to,
            response=content,
            status="success",
        )
    except Exception as exc:
        return QueryResponse(
            query=request.query,
            routed_to="Unknown",
            response=str(exc),
            status="error",
        )
