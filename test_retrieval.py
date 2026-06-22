from database import search_chunks
results = search_chunks("Phase 2 advanced pipeline optimizations", 3)
for parent_content, source, score, child_content in results:
    print(f"Score: {score:.4f}")
    print(f"Source: {source}")
    print(f"Matched: {child_content[:200]}")
    print(f"Context (first 400): {parent_content[:400].encode('ascii', errors='replace').decode()}")
    print("---")
