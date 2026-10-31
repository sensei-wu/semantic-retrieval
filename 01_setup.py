from common import DIMS, connect

with connect() as conn:
    conn.execute("DROP TABLE IF EXISTS documents")
    conn.execute(
        f"""
        CREATE TABLE documents (
            id        bigserial PRIMARY KEY,
            content   text NOT NULL,
            category  text,
            embedding vector({DIMS})
        )
        """
    )
    print(conn.execute("SELECT extversion FROM pg_extension WHERE extname='vector'").fetchone())
    print("documents table ready")
