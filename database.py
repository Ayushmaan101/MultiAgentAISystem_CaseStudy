import os
import logging
import duckdb
from sentence_transformers import SentenceTransformer

import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_conn: duckdb.DuckDBPyConnection | None = None
_model: SentenceTransformer | None = None


def get_connection() -> duckdb.DuckDBPyConnection:
    global _conn
    if _conn is None:
        os.makedirs("./db", exist_ok=True)
        _conn = duckdb.connect(config.DB_PATH)
    return _conn


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
        logger.info("Embedding model loaded.")
    return _model


# ── F1_DB_INIT ──────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create DB directory, connect, load vss, create document_chunks schema."""
    conn = get_connection()

    try:
        conn.execute("INSTALL vss;")
        conn.execute("LOAD vss;")
        logger.info("VSS extension installed and loaded.")
    except Exception as e:
        logger.warning(f"VSS extension unavailable ({e}). Pure SQL cosine fallback will be used.")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id          VARCHAR PRIMARY KEY,
            content     TEXT,
            embedding   FLOAT[384],
            source_file VARCHAR,
            file_type   VARCHAR,
            chunk_index INTEGER,
            chunk_type  VARCHAR DEFAULT 'child',
            parent_id   VARCHAR DEFAULT NULL,
            timestamp   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    logger.info("Table 'document_chunks' ready.")

    try:
        # Required for persisted (on-disk) DuckDB databases
        conn.execute("SET hnsw_enable_experimental_persistence = true;")
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_hnsw
            ON document_chunks USING HNSW (embedding)
        """)
        logger.info("HNSW index created on 'embedding' column.")
    except Exception as e:
        logger.warning(f"Could not create HNSW index ({e}). Falling back to pure SQL cosine similarity on queries.")

    conn.commit()
    logger.info(f"Database initialized at {config.DB_PATH}")


# ── F2_EMBED_FALLBACK part 1 ─────────────────────────────────────────────────

def generate_embedding(text: str) -> list[float]:
    """Return a 384-dimensional normalized embedding vector for the given text."""
    model = get_model()
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()


# ── F2_EMBED_FALLBACK parts 2 & 3 ────────────────────────────────────────────

def search_chunks(query: str, top_k: int = config.TOP_K) -> list[tuple]:
    """
    Parent-child retrieval: search child chunks by semantic similarity,
    then fetch the parent section for each match to return full context.

    Step 1: Generate query embedding.
    Step 2: Search ONLY child chunks (chunk_type = 'child').
            Over-fetch (top_k * 4) on HNSW path to account for index
            returning mixed parent/child rows before WHERE filtering.
    Step 3: For each child, fetch its parent chunk.
    Step 4: Return list of tuples:
            (parent_content, source_file, similarity_score, child_content)
            Falls back to child_content if parent fetch fails.

    Primary path  : HNSW index via vss extension (array_distance).
    Fallback path : Pure SQL cosine similarity using list_cosine_similarity,
                    triggered when the vss search raises any exception.
    """
    conn = get_connection()
    query_embedding = generate_embedding(query)

    # Step 2: search child chunks
    try:
        # Over-fetch so HNSW filtering for chunk_type='child' still yields top_k results
        rows = conn.execute("""
            SELECT id, content, source_file, parent_id,
                   array_distance(embedding, ?::FLOAT[384]) AS distance
            FROM   document_chunks
            WHERE  chunk_type = 'child'
            ORDER  BY distance
            LIMIT  ?
        """, [query_embedding, top_k * 4]).fetchall()

        child_rows = [
            (r[0], r[1], r[2], r[3], round(1.0 - r[4], 6))
            for r in rows
        ][:top_k]

    except Exception as vss_err:
        logger.warning(
            f"VSS search failed ({vss_err}). "
            "Triggering pure SQL cosine similarity fallback."
        )

        try:
            rows = conn.execute("""
                SELECT id, content, source_file, parent_id,
                       list_cosine_similarity(embedding, ?::FLOAT[384]) AS similarity
                FROM   document_chunks
                WHERE  chunk_type = 'child'
                ORDER  BY similarity DESC
                LIMIT  ?
            """, [query_embedding, top_k]).fetchall()

        except Exception:
            rows = conn.execute("""
                SELECT id, content, source_file, parent_id,
                       list_dot_product(embedding, ?::FLOAT[384])
                       / (
                           sqrt(list_dot_product(embedding, embedding))
                           * sqrt(list_dot_product(?::FLOAT[384], ?::FLOAT[384]))
                       ) AS similarity
                FROM   document_chunks
                WHERE  chunk_type = 'child'
                ORDER  BY similarity DESC
                LIMIT  ?
            """, [query_embedding, query_embedding, query_embedding, top_k]).fetchall()

        child_rows = [
            (r[0], r[1], r[2], r[3], round(r[4], 6) if r[4] is not None else 0.0)
            for r in rows
        ]

    # Steps 3 & 4: fetch parent for each child result
    results: list[tuple] = []
    for _, child_content, source_file, parent_id, similarity in child_rows:
        if parent_id:
            try:
                parent_row = conn.execute("""
                    SELECT content FROM document_chunks
                    WHERE  id = ? AND chunk_type = 'parent'
                """, [parent_id]).fetchone()
                parent_content = parent_row[0] if parent_row else child_content
            except Exception:
                parent_content = child_content
        else:
            parent_content = child_content

        results.append((parent_content, source_file, similarity, child_content))

    return results


# ── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    init_db()

    logger.info("Running embedding smoke test...")
    vec = generate_embedding("The quick brown fox jumps over the lazy dog.")
    assert len(vec) == 384, f"Unexpected embedding dimension: {len(vec)}"
    logger.info(f"Embedding verified: 384 dimensions. First 5 values: {vec[:5]}")

    logger.info("All checks passed. F1_DB_INIT and F2_EMBED_FALLBACK are operational.")
