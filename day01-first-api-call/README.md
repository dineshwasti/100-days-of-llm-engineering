# Day 01 — Calling an LLM API

**Goal:** make my first programmatic call to an LLM API and understand the request/response shape (messages, roles, tokens).

## What I did
- Wrote `call_llm.py`, a minimal script that sends a prompt to an LLM API and prints the response.
- Learned the basic anatomy of a chat completion request: `system` / `user` / `assistant` roles, `max_tokens`, temperature.

## Next steps
- Try streaming responses.
- Try passing conversation history (multi-turn).
