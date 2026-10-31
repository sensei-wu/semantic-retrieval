# pgvector-poc

Playground for learning pgvector with local `sentence-transformers` embeddings.

## Setup
1. Start Docker (Colima): `colima start`
2. Start Postgres: `docker compose -f infra/docker-compose.yml up -d`
3. Python env (Intel Mac: torch 2.2.2 is the last wheel available):
   ```
   conda create -n pgvector python=3.11
   conda activate pgvector
   pip install "torch==2.2.2" "numpy<2" "sentence-transformers<4" "psycopg[binary]" pgvector jupyterlab
   ```
4. Run: `python 01_setup.py && python 02_load.py && python 03_search.py "your query"`
5. Then `python 04_index.py`; for experiments: `jupyter lab`

Other projects can use the same server: `CREATE DATABASE other_project;` and connect on `localhost:5432`.

## Learning (human generated)

pgvector is a classic postgre db with vector capabilities enabled -- there is just another column type `vector(dim)` that stores a vector array. But embedding itself is a capability that stays outside of the vector db! You need an embedding model to transform natural language sentences to a high dimensional vector space. 

In this project, we use a 384 dimensional space. The transformer model comes with a capability to convert sentences to vectors because it is pretrained on a large corpus. The model is a transformer with many layers, each with its own weight matrices. It's trained and then fine-tuned with a contrastive objective on large numbers of sentence pairs, pulling similar sentences together in vector space and pushing unrelated ones apart. That's why it's good at similarity, and why a raw language model's output is often worse for search.. Once transformed vectors are stored in pgvector, then one can search for sentences based on similarity to the stored sentences.
