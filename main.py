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
    classification: str = ""
    routing_method: str = ""


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
        routed_to, content, classification, routing_method = run_coordinator(request.query)
        return QueryResponse(
            query=request.query,
            routed_to=routed_to,
            response=content,
            status="success",
            classification=classification,
            routing_method=routing_method,
        )
    except Exception as exc:
        return QueryResponse(
            query=request.query,
            routed_to="Unknown",
            response=str(exc),
            status="error",
            classification="",
            routing_method="",
        )
