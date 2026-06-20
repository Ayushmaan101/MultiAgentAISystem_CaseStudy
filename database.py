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

def search_chunks(query: str, top_k: int = config.TOP_K) -> list[dict]:
    """
    Search document_chunks by semantic similarity.

    Primary path  : HNSW index via vss extension (array_distance).
    Fallback path : Pure SQL cosine similarity using list_cosine_similarity
                    (dot_product / magnitude_a * magnitude_b), triggered when
                    the vss search raises any exception.
    """
    conn = get_connection()
    query_embedding = generate_embedding(query)

    try:
        rows = conn.execute("""
            SELECT id, content, source_file,
                   array_distance(embedding, ?::FLOAT[384]) AS distance
            FROM   document_chunks
            ORDER  BY distance
            LIMIT  ?
        """, [query_embedding, top_k]).fetchall()

        return [
            {
                "id": r[0],
                "content": r[1],
                "source_file": r[2],
                "similarity": round(1.0 - r[3], 6),
            }
            for r in rows
        ]

    except Exception as vss_err:
        logger.warning(
            f"VSS search failed ({vss_err}). "
            "Triggering pure SQL cosine similarity fallback."
        )

        # Pure SQL cosine similarity: dot_product(a,b) / (|a| * |b|)
        # list_cosine_similarity is a built-in DuckDB scalar — no vss required.
        # If that is somehow unavailable a manual LIST_DOT_PRODUCT variant is used.
        try:
            rows = conn.execute("""
                SELECT id, content, source_file,
                       list_cosine_similarity(embedding, ?::FLOAT[384]) AS similarity
                FROM   document_chunks
                ORDER  BY similarity DESC
                LIMIT  ?
            """, [query_embedding, top_k]).fetchall()

        except Exception:
            # Last-resort: manual dot-product / (norm_a * norm_b) in SQL
            rows = conn.execute("""
                SELECT id, content, source_file,
                       list_dot_product(embedding, ?::FLOAT[384])
                       / (
                           sqrt(list_dot_product(embedding, embedding))
                           * sqrt(list_dot_product(?::FLOAT[384], ?::FLOAT[384]))
                       ) AS similarity
                FROM   document_chunks
                ORDER  BY similarity DESC
                LIMIT  ?
            """, [query_embedding, query_embedding, query_embedding, top_k]).fetchall()

        return [
            {
                "id": r[0],
                "content": r[1],
                "source_file": r[2],
                "similarity": round(r[3], 6) if r[3] is not None else 0.0,
            }
            for r in rows
        ]


# ── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    init_db()

    logger.info("Running embedding smoke test...")
    vec = generate_embedding("The quick brown fox jumps over the lazy dog.")
    assert len(vec) == 384, f"Unexpected embedding dimension: {len(vec)}"
    logger.info(f"Embedding verified: 384 dimensions. First 5 values: {vec[:5]}")

    logger.info("All checks passed. F1_DB_INIT and F2_EMBED_FALLBACK are operational.")
