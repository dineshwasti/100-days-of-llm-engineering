# Mini RAG — Ask Questions Over Your Own Documents

A small, self-contained **Retrieval-Augmented Generation (RAG)** command-line tool.
Drop `.txt` files into `docs/`, ask a question, and it retrieves the most relevant
chunks and asks Claude to answer using only that context — with the source
snippets shown so you can verify the answer.

## Why this project

RAG is one of the most common real-world LLM engineering patterns: instead of
hoping the model "knows" your data, you retrieve the relevant pieces yourself
and hand them to the model as context. This project implements the full loop
end-to-end, in plain Python, so the mechanics are visible (no framework magic):

1. **Chunk** documents into overlapping text windows
2. **Embed** each chunk (and the query) into vectors
3. **Retrieve** the top-k most similar chunks via cosine similarity
4. **Generate** an answer, grounded only in the retrieved context

## Setup

```bash
pip install -r requirements.txt
set ANTHROPIC_API_KEY=your_key_here      # PowerShell: $env:ANTHROPIC_API_KEY="your_key_here"
```

## Usage

Put some `.txt` files in `docs/` (a couple of samples are included), then:

```bash
python rag.py "What does the onboarding doc say about laptop setup?"
```

Example output:

```
Answer:
New hires get their laptop shipped before day one; IT sets up accounts
during the first morning.

Sources used:
[1] docs/onboarding.txt (chunk 2)
[2] docs/onboarding.txt (chunk 1)
```

## How it works (files)

| File | Purpose |
|------|---------|
| `rag.py` | Main CLI: chunk → embed → retrieve → generate |
| `docs/` | Your knowledge base (plain `.txt` files) |
| `requirements.txt` | Dependencies |

## Ideas to extend (good "Day 03+" follow-ups)
- Swap the naive cosine-similarity search for a real vector DB (Chroma, FAISS, Qdrant)
- Add re-ranking of retrieved chunks before generation
- Support PDFs / markdown, not just `.txt`
- Add an eval script that checks answer groundedness against sources
- Turn it into a small web app (Streamlit/FastAPI)
