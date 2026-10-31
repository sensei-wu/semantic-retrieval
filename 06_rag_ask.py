"""Answer a question from the stored chunks using a local Ollama model.

usage: python 06_rag_ask.py "your question" [--k 4] [--max-dist 0.65] [--model llama3.2:3b]
"""
import argparse
import json

import requests

from common import connect, embed

parser = argparse.ArgumentParser()
parser.add_argument("question")
parser.add_argument("--k", type=int, default=4)
parser.add_argument("--max-dist", type=float, default=0.65)
parser.add_argument("--model", default="llama3.2:3b")
args = parser.parse_args()

# 1. Retrieve: nearest chunks to the question, dropping anything too far away
qvec = embed([args.question])[0]
with connect() as conn:
    hits = conn.execute(
        "SELECT source, chunk_idx, content, embedding <=> %s AS dist FROM chunks "
        "ORDER BY dist LIMIT %s",
        (qvec, args.k),
    ).fetchall()

print("Retrieved chunks:")
for source, idx, content, dist in hits:
    kept = "keep" if dist <= args.max_dist else "drop"
    print(f"  {dist:.3f} [{kept}] {source}#{idx}: {content[:70]!r}")
hits = [h for h in hits if h[3] <= args.max_dist]

if not hits:
    raise SystemExit("\nNo chunk is close enough to the question; not calling the LLM.")

# 2. Augment: put the chunks in the prompt
context = "\n\n---\n\n".join(h[2] for h in hits)
prompt = (
    "Answer the question using only the context below. "
    "If the context does not contain the answer, say you don't know.\n\n"
    f"Context:\n{context}\n\nQuestion: {args.question}"
)

# 3. Generate: stream the answer from Ollama
print("\nAnswer:")
with requests.post(
    "http://localhost:11434/api/chat",
    json={"model": args.model, "messages": [{"role": "user", "content": prompt}], "stream": True},
    stream=True,
    timeout=300,
) as r:
    r.raise_for_status()
    for line in r.iter_lines():
        if line:
            print(json.loads(line)["message"]["content"], end="", flush=True)
print()
