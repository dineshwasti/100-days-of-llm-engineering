# Prompt Playground CLI

A command-line tool for experimenting with LLM API calls. Change the system
prompt, temperature, and streaming behavior from the terminal, and every call
is logged so you can compare results later.

## Why this project

Before building anything complex with LLMs, you need a fast feedback loop for
the fundamentals: how the system prompt shapes behavior, what temperature
actually changes, and how streaming differs from a blocking call. This tool
makes those knobs easy to turn and keeps a record of every experiment.

## Setup

```bash
pip install -r requirements.txt
```

```powershell
$env:ANTHROPIC_API_KEY="your_key_here"
```

## Usage

Basic call:

```bash
python playground.py "Explain recursion in one sentence"
```

With a system prompt:

```bash
python playground.py "Say hello" --system "Respond only in pirate speak."
```

Deterministic vs creative — run each of these a few times and compare:

```bash
python playground.py "Name a color" --temperature 0
```

```bash
python playground.py "Name a color" --temperature 1
```

Stream the response as it's generated:

```bash
python playground.py "Write a short story about a lighthouse" --stream
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `prompt` | *(required)* | The user prompt to send |
| `--system` | `"You are a helpful assistant."` | System prompt that sets the model's behavior |
| `--temperature` | `1.0` | Sampling temperature, `0.0`–`1.0`. Lower = more deterministic |
| `--max-tokens` | `500` | Maximum tokens in the response |
| `--stream` | off | Print tokens as they arrive instead of waiting for the full response |
| `--model` | `claude-sonnet-5` | Model to use |
| `--no-log` | off | Don't write this call to `log.jsonl` |

## Logging

Every call (unless `--no-log` is passed) appends one JSON object per line to
`log.jsonl`, recording the timestamp, model, system prompt, temperature,
prompt, and response. This makes it easy to diff two runs or review past
experiments:

```json
{"timestamp": "2026-08-24T06:12:44+00:00", "model": "claude-sonnet-5", "system": "You are a helpful assistant.", "temperature": 0.0, "max_tokens": 500, "stream": false, "prompt": "Name a color", "response": "Blue"}
```

## What I learned

- **System prompts** are a separate parameter from the message list — they set
  persistent behavior rather than being part of the conversation.
- **Temperature** near `0` makes output nearly deterministic; near `1` the same
  prompt produces visibly different answers each run.
- **Streaming** uses a different call (`client.messages.stream(...)` as a
  context manager) and yields text incrementally via `stream.text_stream`,
  which matters a lot for perceived latency in real apps.

## Ideas to extend

- Add `--file` to read the prompt from a text file
- Add a `--compare` mode that runs the same prompt at several temperatures side by side
- Print token usage and estimated cost per call
- Add multi-turn support so the CLI keeps conversation history (that's Project 2)
