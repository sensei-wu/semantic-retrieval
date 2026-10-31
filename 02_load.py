from common import connect, embed

DOCS = [
    ("The cat sat on the warm windowsill.", "animals"),
    ("Dogs are loyal companions and love long walks.", "animals"),
    ("Postgres is a powerful open-source relational database.", "tech"),
    ("pgvector adds vector similarity search to Postgres.", "tech"),
    ("Neural networks learn representations from data.", "tech"),
    ("The recipe calls for two cups of flour and a pinch of salt.", "food"),
    ("Sourdough bread needs a long, slow fermentation.", "food"),
    ("The stock market fell sharply on inflation fears.", "finance"),
    ("Interest rates affect mortgage payments.", "finance"),
    ("A thunderstorm rolled in over the mountains.", "weather"),
]

vectors = embed([d for d, _ in DOCS])

with connect() as conn:
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO documents (content, category, embedding) VALUES (%s, %s, %s)",
            [(text, cat, vec) for (text, cat), vec in zip(DOCS, vectors)],
        )
    print(conn.execute("SELECT count(*) FROM documents").fetchone()[0], "rows loaded")
