# Multi-Turn Chatbot (with memory)

A terminal chatbot that remembers the conversation, so follow-ups like
"what did I just ask you?" actually work. Includes slash commands for
clearing history and saving a transcript.

## Why this project

The API is **stateless** — it has no idea a previous call happened. What we
call "memory" is just the client resending every prior message on every
request. Building that loop by hand makes the cost model obvious: each turn
sends the entire history again, so tokens (and latency, and cost) grow with
conversation length rather than with the length of what you just typed.

That's also the foundation for everything later — an agent loop is this same
message-history pattern with tool results appended into it.

## Setup

```bash
pip install -r requirements.txt
```

```powershell
$env:ANTHROPIC_API_KEY="your_key_here"
```

## Usage

```bash
python chat.py
```

With a custom personality:

```bash
python chat.py --system "You are a terse assistant. Answer in one line."
```

Example session:

```
You: my favourite colour is green
Assistant: Noted! Green is a great choice.

You: what's my favourite colour?
Assistant: Green — you just told me.

You: /history
2 turn(s), 4 message(s) in memory.

You: /save chat.md
Saved transcript to chat.md

You: /exit
Bye.
```

## Commands

| Command | Effect |
|---------|--------|
| `/reset` | Clear the conversation history |
| `/save [filename]` | Write the conversation to a markdown transcript |
| `/history` | Show how many turns are currently in memory |
| `/help` | List the commands |
| `/exit` (or `/quit`) | Quit |

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--system` | `"You are a helpful assistant."` | System prompt setting the assistant's behavior |
| `--model` | `claude-sonnet-5` | Model to use |
| `--no-stream` | off | Wait for the full reply instead of streaming it |

## Tests

```bash
python test_chat.py
```

Covers the logic that doesn't need the API: history accumulation, turn
counting, `/reset`, transcript rendering, `/save` with and without a filename,
unknown commands, and case-insensitive commands.

## What I learned

- **The API is stateless.** Memory is entirely client-side — you rebuild it on
  every call by resending `messages`. Nothing is stored server-side between
  requests.
- **The system prompt is not part of the history.** It's a separate parameter,
  so it doesn't get repeated in the message list and doesn't grow over time.
- **Message order and roles matter.** The list has to alternate user/assistant
  correctly; append the assistant's reply *after* it comes back, or the next
  turn sends a malformed history.
- **Watch out for aliasing.** While testing I captured `conversation.messages`
  by reference and every snapshot looked identical, because they were all the
  same list object. It made my test look like history was broken when it
  wasn't — a good reminder that Python lists are passed by reference.

## Ideas to extend

- Trim or summarize old turns once the history passes a token budget
- Print token usage and running cost per turn
- Persist history to disk so a session survives restarting the program
- Add `/system <prompt>` to change the personality mid-conversation
- Support loading a previous transcript back in as starting context
