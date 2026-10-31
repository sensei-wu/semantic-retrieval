"""Chunk every .md/.txt file in docs/, embed the chunks and store them in `chunks`."""
from pathlib import Path

from common import DIMS, connect, embed

CHUNK_CHARS = 800  # all-MiniLM-L6-v2 truncates input at ~256 tokens (~1000 chars)


def chunk(text: str) -> list[str]:
    """Pack paragraphs into ~CHUNK_CHARS pieces, each prefixed with its nearest heading."""
    chunks, current, heading = [], "", ""

    def flush():
        nonlocal current
        if current.strip():
            chunks.append(f"{heading}\n{current.strip()}" if heading else current.strip())
        current = ""

    for para in text.split("\n\n"):
        if para.startswith("#"):
            flush()
            heading = para.splitlines()[0].lstrip("# ").strip()
        if len(current) + len(para) > CHUNK_CHARS:
            flush()
        current += para + "\n\n"
    flush()
    return chunks


rows = []
for path in sorted(Path("docs").glob("*")):
    if path.suffix in {".md", ".txt"}:
        for i, c in enumerate(chunk(path.read_text())):
            rows.append((path.name, i, c))
print(f"{len(rows)} chunks from docs/")

vectors = embed([c for _, _, c in rows])

with connect() as conn:
    conn.execute("DROP TABLE IF EXISTS chunks")
    conn.execute(
        f"""
        CREATE TABLE chunks (
            id        bigserial PRIMARY KEY,
            source    text NOT NULL,
            chunk_idx int  NOT NULL,
            content   text NOT NULL,
            embedding vector({DIMS})
        )
        """
    )
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO chunks (source, chunk_idx, content, embedding) VALUES (%s, %s, %s, %s)",
            [(s, i, c, v) for (s, i, c), v in zip(rows, vectors)],
        )
    conn.execute(
        "CREATE INDEX chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops)"
    )
    print(conn.execute("SELECT count(*) FROM chunks").fetchone()[0], "chunks stored, HNSW index built")
