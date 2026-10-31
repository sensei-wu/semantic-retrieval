import sys

from common import connect, embed

query = " ".join(sys.argv[1:]) or "database for storing embeddings"
qvec = embed([query])[0]

with connect() as conn:
    print(f"Query: {query!r}\n")
    for op, name in [("<=>", "cosine distance"), ("<->", "L2 distance"), ("<#>", "negative inner product")]:
        print(f"-- {name} ({op})")
        rows = conn.execute(
            f"SELECT content, category, embedding {op} %s AS dist "
            f"FROM documents ORDER BY dist LIMIT 3",
            (qvec,),
        ).fetchall()
        for content, cat, dist in rows:
            print(f"  {dist:7.4f}  [{cat}] {content}")
        print()

    print("-- filtered: category = 'tech', cosine")
    for content, dist in conn.execute(
        "SELECT content, embedding <=> %s AS dist FROM documents "
        "WHERE category = 'tech' ORDER BY dist LIMIT 3",
        (qvec,),
    ):
        print(f"  {dist:7.4f}  {content}")
