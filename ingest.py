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
    with open(path, "r", encoding="utf-8-sig") as f:  # utf-8-sig strips BOM if present
        return f.read()


def parse_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8-sig") as f:
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


# ── 2. CHUNKER DISPATCHER ────────────────────────────────────────────────────

def chunk_document(text: str, file_type: str, source_file: str) -> list[dict]:
    """Route to the correct per-file-type chunker."""
    if file_type == "md":
        return _chunk_markdown(text, source_file)
    elif file_type == "pdf":
        return _chunk_pdf(text, source_file)
    else:
        return _chunk_txt(text, source_file)


# ── 3. MARKDOWN CHUNKER ──────────────────────────────────────────────────────

_HEADER_RE = re.compile(r"^(#{1,3} .+)$", re.MULTILINE)


def _chunk_markdown(text: str, source_file: str) -> list[dict]:
    """
    Split markdown by headers. Each header + its body = one parent chunk.
    Paragraphs within the body = child chunks, each pointing back to their parent.
    Header is never orphaned: it is always stored with its body in the parent,
    and every child carries parent_id so full context can be fetched at retrieval time.
    """
    header_matches = list(_HEADER_RE.finditer(text))

    # Build (header, body) sections
    sections: list[tuple[str, str]] = []
    if not header_matches:
        sections.append(("", text.strip()))
    else:
        pre = text[: header_matches[0].start()].strip()
        if pre:
            sections.append(("", pre))
        for i, m in enumerate(header_matches):
            header = m.group(1)
            body_start = m.end()
            body_end = header_matches[i + 1].start() if i + 1 < len(header_matches) else len(text)
            body = text[body_start:body_end].strip()
            sections.append((header, body))

    chunks: list[dict] = []
    global_child_index = 0

    for section_index, (header, body) in enumerate(sections):
        parent_content = (header + "\n" + body).strip() if header else body
        if not parent_content:
            continue

        parent_id = hashlib.md5(
            (source_file + (header or "pre") + str(section_index)).encode()
        ).hexdigest()

        chunks.append({
            "id": parent_id,
            "content": parent_content,
            "source_file": source_file,
            "file_type": "md",
            "chunk_type": "parent",
            "parent_id": None,
            "chunk_index": section_index,
        })

        if not body:
            continue

        # Build child chunks from paragraphs inside the body
        child_paras: list[str] = []
        for para in body.split("\n\n"):
            para = para.strip()
            if not para or len(para) < 30:
                continue
            if len(para) <= config.CHUNK_SIZE:
                child_paras.append(para)
            else:
                # Long paragraph: accumulate lines up to CHUNK_SIZE
                current = ""
                for line in para.split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    candidate = (current + "\n" + line).strip() if current else line
                    if len(candidate) <= config.CHUNK_SIZE:
                        current = candidate
                    else:
                        if current and len(current) >= 30:
                            child_paras.append(current)
                        current = line if len(line) <= config.CHUNK_SIZE else line[: config.CHUNK_SIZE]
                if current and len(current) >= 30:
                    child_paras.append(current)

        for child_index, para in enumerate(child_paras):
            child_id = hashlib.md5(
                (source_file + parent_id + str(child_index)).encode()
            ).hexdigest()
            chunks.append({
                "id": child_id,
                "content": para,
                "source_file": source_file,
                "file_type": "md",
                "chunk_type": "child",
                "parent_id": parent_id,
                "chunk_index": global_child_index,
            })
            global_child_index += 1

    return chunks


# ── 4. PDF CHUNKER ───────────────────────────────────────────────────────────

def _chunk_pdf(text: str, source_file: str) -> list[dict]:
    """Group every 3 consecutive paragraphs into a parent. Each paragraph is a child."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip() and len(p.strip()) >= 30]
    return _group_into_parent_child(paras, source_file, "pdf", "pdf_parent")


# ── 5. TXT CHUNKER ───────────────────────────────────────────────────────────

def _chunk_txt(text: str, source_file: str) -> list[dict]:
    """Group every 3 consecutive paragraphs into a parent. Each paragraph is a child."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip() and len(p.strip()) >= 30]
    return _group_into_parent_child(paras, source_file, "txt", "txt_parent")


def _group_into_parent_child(
    paras: list[str], source_file: str, file_type: str, parent_key: str
) -> list[dict]:
    """Shared logic for PDF and TXT: group every 3 paragraphs into one parent."""
    chunks: list[dict] = []
    global_child_index = 0

    for group_index in range(0, len(paras), 3):
        group = paras[group_index : group_index + 3]
        parent_content = "\n\n".join(group)
        parent_id = hashlib.md5(
            (source_file + parent_key + str(group_index // 3)).encode()
        ).hexdigest()

        chunks.append({
            "id": parent_id,
            "content": parent_content,
            "source_file": source_file,
            "file_type": file_type,
            "chunk_type": "parent",
            "parent_id": None,
            "chunk_index": group_index // 3,
        })

        for para_index, para in enumerate(group):
            child_id = hashlib.md5(
                (source_file + parent_id + str(para_index)).encode()
            ).hexdigest()
            chunks.append({
                "id": child_id,
                "content": para,
                "source_file": source_file,
                "file_type": file_type,
                "chunk_type": "child",
                "parent_id": parent_id,
                "chunk_index": global_child_index,
            })
            global_child_index += 1

    return chunks


# ── 6. UPSERT TO DUCKDB ──────────────────────────────────────────────────────

def upsert_chunks(source_file: str, file_type: str, chunks: list[dict]) -> tuple[int, int]:
    conn = database.get_connection()
    for chunk in chunks:
        embedding = database.generate_embedding(chunk["content"])
        conn.execute(
            """
            INSERT OR REPLACE INTO document_chunks
                (id, content, embedding, source_file, file_type,
                 chunk_index, chunk_type, parent_id)
            VALUES (?, ?, ?::FLOAT[384], ?, ?, ?, ?, ?)
            """,
            [
                chunk["id"],
                chunk["content"],
                embedding,
                chunk["source_file"],
                chunk["file_type"],
                chunk["chunk_index"],
                chunk["chunk_type"],
                chunk["parent_id"],
            ],
        )
    conn.commit()
    parents = sum(1 for c in chunks if c["chunk_type"] == "parent")
    children = sum(1 for c in chunks if c["chunk_type"] == "child")
    return parents, children


# ── 7. MAIN INGEST FUNCTION ───────────────────────────────────────────────────

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
            chunks = chunk_document(text, ext, filename)
            parents, children = upsert_chunks(filename, ext, chunks)
            elapsed = time.perf_counter() - start
            print(f"[OK] {filename} | {parents} parents | {children} children | {elapsed:.2f}s")
        except Exception as exc:
            print(f"[ERROR] {filename}: {exc}")


# ── 8. ENTRY POINT ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ingest_directory("./documents")
