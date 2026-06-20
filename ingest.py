import hashlib
import logging
import os
import re
import time

import config
import database

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt"}

# ── 1. FILE TYPE PARSERS ──────────────────────────────────────────────────────

def parse_pdf(path: str) -> str:
    from pypdf import PdfReader
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def parse_markdown(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def parse_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def parse_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return parse_pdf(path)
    elif ext == ".md":
        return parse_markdown(path)
    elif ext == ".txt":
        return parse_txt(path)
    raise ValueError(f"Unsupported file type: {ext}")


# ── 2. STRUCTURAL/RECURSIVE CHUNKER ──────────────────────────────────────────

_HEADER_RE = re.compile(r"(?=^#{1,3} )", re.MULTILINE)


def _split_by_priority(text: str) -> list[str]:
    """Split text by: markdown headers > double newlines > single newlines > char limit."""
    segments = [s.strip() for s in _HEADER_RE.split(text) if s.strip()]
    if not segments:
        return [text.strip()] if text.strip() else []

    result: list[str] = []
    for seg in segments:
        if len(seg) <= config.CHUNK_SIZE:
            result.append(seg)
            continue

        # Priority 2: double newlines
        for para in seg.split("\n\n"):
            para = para.strip()
            if not para:
                continue
            if len(para) <= config.CHUNK_SIZE:
                result.append(para)
                continue

            # Priority 3: single newlines — accumulate lines up to CHUNK_SIZE
            current = ""
            for line in para.split("\n"):
                line = line.strip()
                if not line:
                    continue
                candidate = (current + "\n" + line).strip() if current else line
                if len(candidate) <= config.CHUNK_SIZE:
                    current = candidate
                else:
                    if current:
                        result.append(current)
                    if len(line) <= config.CHUNK_SIZE:
                        current = line
                    else:
                        # Priority 4: fixed character limit
                        for i in range(0, len(line), config.CHUNK_SIZE):
                            result.append(line[i : i + config.CHUNK_SIZE])
                        current = ""
            if current:
                result.append(current)

    return result


def chunk_text(text: str) -> list[str]:
    """Return non-empty chunks ≤ CHUNK_SIZE, each carrying CHUNK_OVERLAP context prefix."""
    raw = _split_by_priority(text)
    chunks: list[str] = []
    for i, chunk in enumerate(raw):
        if i > 0 and config.CHUNK_OVERLAP > 0:
            overlap = raw[i - 1][-config.CHUNK_OVERLAP:]
            chunk = (overlap + " " + chunk).strip()[: config.CHUNK_SIZE]
        chunks.append(chunk.strip())
    return [c for c in chunks if c]


# ── 3. UPSERT TO DUCKDB ───────────────────────────────────────────────────────

def _chunk_id(source_file: str, chunk_index: int) -> str:
    return hashlib.md5((source_file + str(chunk_index)).encode()).hexdigest()


def upsert_chunks(source_file: str, file_type: str, chunks: list[str]) -> None:
    conn = database.get_connection()
    for idx, content in enumerate(chunks):
        cid = _chunk_id(source_file, idx)
        embedding = database.generate_embedding(content)
        conn.execute(
            """
            INSERT OR REPLACE INTO document_chunks
                (id, content, embedding, source_file, file_type, chunk_index)
            VALUES (?, ?, ?::FLOAT[384], ?, ?, ?)
            """,
            [cid, content, embedding, source_file, file_type, idx],
        )
    conn.commit()


# ── 4. MAIN INGEST FUNCTION ───────────────────────────────────────────────────

def ingest_directory(directory: str) -> None:
    database.init_db()
    os.makedirs(directory, exist_ok=True)

    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ]

    if not files:
        print(f"No supported files found in {directory!r}.")
        return

    for path in sorted(files):
        filename = os.path.basename(path)
        ext = os.path.splitext(filename)[1].lower().lstrip(".")
        start = time.perf_counter()
        try:
            text = parse_file(path)
            chunks = chunk_text(text)
            upsert_chunks(filename, ext, chunks)
            elapsed = time.perf_counter() - start
            print(f"[OK] {filename} | {len(chunks)} chunks | {elapsed:.2f}s")
        except Exception as exc:
            print(f"[ERROR] {filename}: {exc}")


# ── 5. ENTRY POINT ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ingest_directory("./documents")
