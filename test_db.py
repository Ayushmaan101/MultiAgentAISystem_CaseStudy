import duckdb
con = duckdb.connect("./db/embeddings.db")
parents = con.execute("SELECT COUNT(*) FROM document_chunks WHERE chunk_type='parent'").fetchone()[0]
children = con.execute("SELECT COUNT(*) FROM document_chunks WHERE chunk_type='child'").fetchone()[0]
print(f"Parents: {parents}")
print(f"Children: {children}")
files = con.execute(
    "SELECT source_file, chunk_type, COUNT(*) as n FROM document_chunks "
    "GROUP BY source_file, chunk_type ORDER BY source_file"
).fetchall()
for row in files:
    print(row)
con.close()
