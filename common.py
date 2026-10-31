import os

import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/pgvector_playground"
)
MODEL_NAME = "all-MiniLM-L6-v2"  # 384 dimensions
DIMS = 384

_model = None


def embed(texts: list[str]):
    """Return a (len(texts), 384) numpy array of normalized embeddings."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model.encode(texts, normalize_embeddings=True)


def connect() -> psycopg.Connection:
    conn = psycopg.connect(DATABASE_URL, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn
