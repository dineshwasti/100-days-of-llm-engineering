# LLM Engineering Project Ladder

A set of projects to build **yourself**, ordered easy → medium → hard. Each one
teaches a specific skill that shows up in real LLM engineering roles. Don't
skip levels — each project reuses skills from the ones before it.

Rule of thumb: **read the spec, then close this file and build it from
memory/docs.** Come back only when stuck. That's how the skill actually sticks.

---

## 🟢 Easy — fundamentals

### 1. Prompt Playground CLI
**Learn:** basic API calls, system prompts, temperature/max_tokens, streaming.
**Build:** a CLI that takes a prompt from the terminal, lets you set
`--system`, `--temperature`, `--stream`, and prints the response. Bonus:
save every prompt+response to a local JSON log file.
**Done when:** you can run `python playground.py "explain recursion" --stream`
and watch tokens print live.

### 2. Multi-Turn Chatbot (with memory)
**Learn:** conversation state, message history, roles.
**Build:** a terminal chatbot that remembers the conversation across turns
(keeps appending to a `messages` list) until you type `exit`. Add a
`/reset` command to clear history and a `/save` command to dump the
conversation to a `.md` transcript.
**Done when:** the bot correctly answers "what did I just ask you?"

### 3. Structured Data Extractor
**Learn:** getting reliable structured output (JSON) from an LLM.
**Build:** feed it messy text (e.g. a paragraph describing a person) and
have it return valid JSON matching a schema you define (name, age, job,
etc). Validate the output with `pydantic` or `jsonschema` and retry on
failure.
**Done when:** it handles 10 different messy inputs without ever returning
invalid JSON.

---

## 🟡 Medium — real components

### 4. Mini RAG over your own docs *(already scaffolded → [day02-mini-rag](./day02-mini-rag/))*
**Learn:** chunking, embeddings, cosine similarity retrieval, grounded generation.
**Build:** you already have a working version — try rebuilding it yourself
from scratch without looking, or swap the naive hashed embedding for a real
embedding model.
**Stretch:** add source citations with exact quotes, not just filenames.

### 5. Tool-Calling Agent
**Learn:** function/tool calling, the model deciding *when* to call a tool.
**Build:** give the model 2-3 tools (e.g. `get_weather(city)`,
`calculate(expr)`, `search_docs(query)` reusing project 4). Let it decide
which tool to call based on the user's question, execute it, and feed the
result back for a final answer.
**Done when:** it correctly picks the right tool for 5 different question
types, and correctly picks *no tool* for a question that doesn't need one.

### 6. LLM Output Evaluator
**Learn:** how to actually measure LLM quality (not vibes).
**Build:** a script that takes a set of (question, expected_answer) pairs,
runs your model on each, and scores the answers — one automated check
(e.g. keyword/exact match) AND one "LLM-as-judge" check (ask a second LLM
call to rate correctness 1-5 with reasoning). Output a report: pass rate,
average score, worst 3 examples.
**Done when:** you can run it against project 4's RAG system and get a
real pass/fail number.

---

## 🔴 Hard — production-shaped

### 7. Guarded Chat API (prompt-injection resistant)
**Learn:** input/output guardrails, prompt injection defense, safe tool use.
**Build:** wrap a chatbot behind a small FastAPI server. Add: input
sanitization/limits, a system prompt hardened against injection, output
filtering (block certain content), and rate limiting per user. Write 5
adversarial test prompts (e.g. "ignore previous instructions...") and prove
your guardrails hold.
**Done when:** your 5 adversarial prompts all fail to break the intended
behavior, and you can explain *why* each defense works.

### 8. Production RAG Service
**Learn:** real vector DB, caching, latency/cost tradeoffs, observability.
**Build:** rebuild project 4 as a proper service: real vector DB (Chroma/
Qdrant/FAISS), response caching for repeated queries, request logging with
latency + token cost per call, and a `/health` endpoint. Load-test it with
20 concurrent fake requests and report p50/p95 latency.
**Done when:** you have a dashboard or log file showing latency, token
cost, and cache hit rate per request.

### 9. Multi-Agent Task Orchestrator
**Learn:** agent-to-agent coordination, planning, error recovery.
**Build:** a "planner" agent that breaks a complex task (e.g. "research X
and write a summary report") into subtasks, dispatches them to specialist
agents (researcher, writer, fact-checker), and combines results. Handle at
least one failure case gracefully (a subtask errors — retry or replan).
**Done when:** it completes a genuinely multi-step task end-to-end and you
can show the trace of what each agent did.

---

## How to use this with the repo

- Each project gets its own `dayNN-project-name/` folder when you start it.
- Write the README **first** (what you're building and why) before code —
  it forces you to actually understand the spec.
- Commit in small steps as you go, not one giant dump at the end — the
  commit history should show your actual thinking/progress.
- When you finish one, ping me and I'll do a quick code review pass before
  you commit, and we'll pick/scaffold the next one together.
