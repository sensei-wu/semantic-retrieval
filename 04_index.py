from common import connect, embed

with connect() as conn:
    conn.execute("DROP INDEX IF EXISTS documents_embedding_hnsw")
    conn.execute(
        "CREATE INDEX documents_embedding_hnsw ON documents "
        "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
    )
    qvec = embed(["database"])[0]
    # With only 10 rows the planner prefers a seq scan; force the index to see it work.
    conn.execute("SET enable_seqscan = off")
    plan = conn.execute(
        "EXPLAIN SELECT id FROM documents ORDER BY embedding <=> %s LIMIT 3", (qvec,)
    ).fetchall()
    print("\n".join(r[0] for r in plan))
