# Tool-Calling Agent

The model is given three tools and decides for itself which one to call — or
whether to call any at all. We execute what it asks for, hand the result back,
and let it continue until it has an answer.

## Why this project

This is the jump from "chatbot" to "agent", and it's smaller than it sounds.
There's no magic: the model returns a `tool_use` block instead of text, you run
the function, you send the result back, you call the model again. That loop is
the whole idea behind every agent framework.

The genuinely interesting parts are the ones that aren't in the happy path:

- **Knowing when *not* to call a tool.** "Who wrote Hamlet?" should be answered
  directly. An agent that reaches for a tool every time is worse than no agent.
- **Errors are messages, not exceptions.** When a tool fails, the model needs to
  *see* the failure so it can explain or recover. Crashing the loop wastes the
  work already done.
- **Tool input is untrusted input.** The arguments come from a language model,
  not from you. See the calculator note below.

## The tools

| Tool | What it does |
|------|--------------|
| `calculate` | Evaluates arithmetic via a restricted AST walker |
| `get_weather` | Canned weather for five cities (no network needed) |
| `search_docs` | Keyword search over a small employee handbook |

### A note on the calculator

The obvious implementation is `eval(expression)`. Don't. The expression is
written by the model, and a prompt-injected or simply confused model can emit
`__import__('os').system(...)`. This version parses to an AST and walks only
number literals and arithmetic operators — anything else raises before it runs:

```
"__import__('os').system('echo PWNED')"  ->  BLOCKED: expression element not allowed: Call
"2 + 2"                                  ->  4
```

There's also an exponent cap, because `9 ** 999999999` is a denial-of-service
in one line even without any code execution.

## Setup

```bash
pip install -r requirements.txt
```

```powershell
$env:ANTHROPIC_API_KEY="your_key_here"
```

## Usage

```bash
python agent.py "What is 1847 * 23?"
```

```
Q: What is 1847 * 23?
A: 1847 × 23 = 42,481.
   (tools used: calculate; 2 model call(s))
```

See the tool calls as they happen:

```bash
python agent.py "What's the weather in Tokyo?" --verbose
```

Run the full set, including a question needing no tool and one where the tool
deliberately fails:

```bash
python agent.py --demo
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `question` | — | What to ask (omit if using `--demo`) |
| `--demo` | off | Run built-in questions covering each tool |
| `--max-steps` | `6` | Loop limit before giving up |
| `--verbose` | off | Print each tool call and result |
| `--model` | `claude-sonnet-5` | Model to use |

## Tests

```bash
python test_agent.py
```

19 tests, covering:

- **The sandbox** — that `calculate` refuses `__import__`, `open`, `exec`,
  `lambda`, subclass traversal, and oversized exponents
- **Tool behaviour** — success, unknown city, no search match, bad arguments
- **The agent loop** — answering with no tool, calling one tool then answering,
  handling several tool calls in a single turn, and stopping at `max_steps`
- **Protocol correctness** — that every `tool_use` id gets exactly one matching
  `tool_result`, in order, with the `is_error` flag set

## What I learned

- **The loop is the agent.** Call model → got `tool_use`? → run it → append
  result → call again. Everything else is ergonomics.
- **`stop_reason` drives control flow.** `tool_use` means keep going;
  anything else means the model is done.
- **Every `tool_use` needs exactly one `tool_result` with a matching id**, sent
  in a single user turn. The model can request several tools at once, and
  missing one is an API error rather than a graceful degradation.
- **Append the assistant's request verbatim.** The tool result only makes sense
  relative to the request that produced it, so the raw content blocks go back
  into the history unmodified.
- **Bound the loop.** A model that keeps requesting tools will happily spend
  your money forever. `max_steps` is not optional.
- **Tool descriptions are prompt engineering.** "Use this for any math rather
  than working it out yourself" changes behaviour more than any code does.

## Ideas to extend

- Add a tool that hits a real API and deal with timeouts and rate limits
- Let the agent hold a conversation instead of answering one question
- Reuse Day 02's RAG retriever as a `search_docs` backend over real documents
- Log every tool call with latency and token cost
- Add a tool requiring confirmation before it runs (a write, a purchase)
- Try prompt-injecting the handbook text and see whether the agent obeys it
