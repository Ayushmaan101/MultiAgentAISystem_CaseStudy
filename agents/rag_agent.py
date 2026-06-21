import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import config
import database


def document_lookup(query: str) -> str:
    """Search the local knowledge base for chunks relevant to the query."""
    results = database.search_chunks(query, config.TOP_K)
    if not results:
        return "=== RETRIEVED CHUNKS ===\n\nNo chunks found for this query.\n\n=== END RETRIEVED CHUNKS ==="

    lines = ["=== RETRIEVED CHUNKS ===\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"[Chunk {i}]")
        lines.append(f"Source: {r['source_file']}")
        lines.append(f"Similarity: {r['similarity']:.6f}")
        lines.append(f"Content: {r['content']}")
        lines.append("")
    lines.append("=== END RETRIEVED CHUNKS ===")
    return "\n".join(lines)
