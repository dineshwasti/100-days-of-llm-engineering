"""
Mini RAG: ask questions over local .txt documents.

Pipeline: chunk -> embed (local, no extra API key needed) -> retrieve (cosine
similarity) -> generate (Claude, grounded in retrieved context).

Usage:
    python rag.py "your question here"

Setup:
    pip install -r requirements.txt
    set ANTHROPIC_API_KEY=your_key_here
"""

import os
import re
import sys
import glob
import hashlib
from collections import Counter

import numpy as np
from anthropic import Anthropic

DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")
CHUNK_SIZE = 400        # characters per chunk
CHUNK_OVERLAP = 80      # overlap between consecutive chunks
TOP_K = 3               # number of chunks to retrieve
EMBED_DIM = 512         # dimensionality of our simple hashed embeddings


# ---------------------------------------------------------------------------
# 1. Chunking
# ---------------------------------------------------------------------------

def load_documents(docs_dir: str) -> list[tuple[str, str]]:
    """Return [(filepath, text), ...] for every .txt file in docs_dir."""
    paths = sorted(glob.glob(os.path.join(docs_dir, "*.txt")))
    return [(p, open(p, encoding="utf-8").read()) for p in paths]


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping fixed-size chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end].strip())
        start += size - overlap
    return [c for c in chunks if c]


# ---------------------------------------------------------------------------
# 2. Embedding (simple hashed bag-of-words -- no extra API/key required)
# ---------------------------------------------------------------------------
# This is intentionally simple so the RAG mechanics stay visible. Swap this
# function for a real embedding model (Voyage, OpenAI, sentence-transformers)
# to improve retrieval quality -- see README "Ideas to extend".

def embed(text: str, dim: int = EMBED_DIM) -> np.ndarray:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    vec = np.zeros(dim, dtype=np.float32)
    counts = Counter(words)
    for word, count in counts.items():
        idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % dim
        vec[idx] += count
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


# ---------------------------------------------------------------------------
# 3. Retrieval
# ---------------------------------------------------------------------------

def build_index(docs_dir: str = DOCS_DIR):
    records = []  # list of dicts: {source, chunk_id, text, vector}
    for path, text in load_documents(docs_dir):
        for i, chunk in enumerate(chunk_text(text)):
            records.append({
                "source": path,
                "chunk_id": i,
                "text": chunk,
                "vector": embed(chunk),
            })
    return records


def retrieve(query: str, records: list[dict], top_k: int = TOP_K) -> list[dict]:
    q_vec = embed(query)
    scored = [(cosine_similarity(q_vec, r["vector"]), r) for r in records]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:top_k]]


# ---------------------------------------------------------------------------
# 4. Generation
# ---------------------------------------------------------------------------

def generate_answer(query: str, context_chunks: list[dict]) -> str:
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    context = "\n\n".join(
        f"[{i+1}] (from {os.path.basename(c['source'])})\n{c['text']}"
        for i, c in enumerate(context_chunks)
    )
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the answer isn't in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}"
    )
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print('Usage: python rag.py "your question"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])

    records = build_index()
    if not records:
        print(f"No documents found in {DOCS_DIR}. Add some .txt files first.")
        sys.exit(1)

    top_chunks = retrieve(query, records)
    answer = generate_answer(query, top_chunks)

    print("Answer:")
    print(answer)
    print("\nSources used:")
    for i, c in enumerate(top_chunks):
        print(f"[{i+1}] {c['source']} (chunk {c['chunk_id']})")


if __name__ == "__main__":
    main()
