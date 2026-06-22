from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

import database
from agents.coordinator import run_coordinator


# ── Lifespan (startup) ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    database.init_db()
    print("Database initialized and ready")
    yield


app = FastAPI(
    title="AI Research Assistant",
    description="Multi-agent assistant: RAG, Calculator, Web Search via Agno + Groq",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request / response schemas ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    query: str
    rewritten_query: str
    classification: str
    routing_method: str
    routed_to: str
    raw_tool_result: str
    final_answer: str
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
        result = run_coordinator(request.query)
        return QueryResponse(
            query=request.query,
            rewritten_query=result["rewritten_query"],
            classification=result["classification"],
            routing_method=result["routing_method"],
            routed_to=result["routed_to"],
            raw_tool_result=result["raw_tool_result"],
            final_answer=result["final_answer"],
            status=result["status"],
        )
    except Exception as exc:
        return QueryResponse(
            query=request.query,
            rewritten_query="",
            classification="",
            routing_method="",
            routed_to="Unknown",
            raw_tool_result="",
            final_answer=str(exc),
            status="error",
        )
