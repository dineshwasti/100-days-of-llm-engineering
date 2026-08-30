"""
Multi-Turn Chatbot with memory.

A terminal chatbot that remembers the whole conversation, so follow-up
questions like "what did I just ask you?" work. Supports slash commands for
resetting the conversation and saving a transcript.

Commands:
    /reset            clear the conversation history
    /save [filename]  write the conversation to a markdown transcript
    /history          show how many turns are in memory
    /help             list commands
    /exit             quit

Usage:
    python chat.py
    python chat.py --system "You are a terse assistant. Answer in one line."
    python chat.py --no-stream

Setup:
    pip install -r requirements.txt
    set ANTHROPIC_API_KEY=your_key_here   (PowerShell: $env:ANTHROPIC_API_KEY="your_key_here")
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from anthropic import Anthropic

DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_SYSTEM = "You are a helpful assistant."


class Conversation:
    """Holds the message history that gets replayed to the model each turn.

    This is the whole point of the project: the API itself is stateless, so
    "memory" is just us resending every previous message on every call.
    """

    def __init__(self, system: str = DEFAULT_SYSTEM):
        self.system = system
        self.messages: list[dict] = []
        self.started_at = datetime.now()

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_assistant(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text})

    def reset(self) -> None:
        self.messages = []
        self.started_at = datetime.now()

    @property
    def turns(self) -> int:
        """A turn is one user message plus its reply."""
        return len([m for m in self.messages if m["role"] == "user"])

    def to_markdown(self) -> str:
        lines = [
            "# Conversation transcript",
            "",
            f"- **Started:** {self.started_at:%Y-%m-%d %H:%M:%S}",
            f"- **System prompt:** {self.system}",
            f"- **Turns:** {self.turns}",
            "",
            "---",
            "",
        ]
        for message in self.messages:
            speaker = "You" if message["role"] == "user" else "Assistant"
            lines.append(f"**{speaker}:**")
            lines.append("")
            lines.append(message["content"])
            lines.append("")
        return "\n".join(lines)


def default_transcript_name() -> str:
    return f"transcript-{datetime.now():%Y%m%d-%H%M%S}.md"


def save_transcript(conversation: Conversation, filename: str | None = None) -> Path:
    path = Path(filename or default_transcript_name())
    path.write_text(conversation.to_markdown(), encoding="utf-8")
    return path


HELP_TEXT = """Commands:
  /reset            clear the conversation history
  /save [filename]  save the conversation as a markdown transcript
  /history          show how many turns are in memory
  /help             show this message
  /exit             quit"""


def handle_command(line: str, conversation: Conversation) -> str:
    """Run a slash command. Returns 'exit' to quit, otherwise 'handled'."""
    parts = line.strip().split(maxsplit=1)
    command = parts[0].lower()
    argument = parts[1].strip() if len(parts) > 1 else None

    if command in ("/exit", "/quit"):
        return "exit"

    if command == "/reset":
        conversation.reset()
        print("History cleared.")

    elif command == "/save":
        if not conversation.messages:
            print("Nothing to save yet.")
        else:
            path = save_transcript(conversation, argument)
            print(f"Saved transcript to {path}")

    elif command == "/history":
        print(f"{conversation.turns} turn(s), "
              f"{len(conversation.messages)} message(s) in memory.")

    elif command == "/help":
        print(HELP_TEXT)

    else:
        print(f"Unknown command: {command}. Try /help")

    return "handled"


def reply_streaming(client: Anthropic, conversation: Conversation, model: str) -> str:
    chunks = []
    with client.messages.stream(
        model=model,
        max_tokens=1000,
        system=conversation.system,
        messages=conversation.messages,
    ) as stream:
        for chunk in stream.text_stream:
            print(chunk, end="", flush=True)
            chunks.append(chunk)
    print()
    return "".join(chunks)


def reply_blocking(client: Anthropic, conversation: Conversation, model: str) -> str:
    response = client.messages.create(
        model=model,
        max_tokens=1000,
        system=conversation.system,
        messages=conversation.messages,
    )
    text = response.content[0].text
    print(text)
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A terminal chatbot that remembers the conversation."
    )
    parser.add_argument("--system", default=DEFAULT_SYSTEM,
                        help="System prompt setting the assistant's behavior")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model to use")
    parser.add_argument("--no-stream", action="store_true",
                        help="Wait for the full reply instead of streaming it")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    client = Anthropic(api_key=api_key)
    conversation = Conversation(system=args.system)

    print("Chatbot ready. Type /help for commands, /exit to quit.\n")

    while True:
        try:
            line = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        if line.startswith("/"):
            if handle_command(line, conversation) == "exit":
                break
            continue

        conversation.add_user(line)

        print("Assistant: ", end="", flush=True)
        if args.no_stream:
            reply = reply_blocking(client, conversation, args.model)
        else:
            reply = reply_streaming(client, conversation, args.model)

        conversation.add_assistant(reply)
        print()

    print("Bye.")


if __name__ == "__main__":
    main()
