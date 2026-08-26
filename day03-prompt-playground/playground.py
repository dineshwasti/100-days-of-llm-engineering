"""
Prompt Playground CLI

A small command-line tool for experimenting with LLM API calls:
system prompts, temperature, streaming, and automatic call logging.

Usage:
    python playground.py "Explain recursion in one sentence"
    python playground.py "Write a haiku about the ocean" --temperature 1.0
    python playground.py "Count to 20 slowly" --stream
    python playground.py "You are a pirate. Say hi." --system "Respond only in pirate speak."

Setup:
    pip install -r requirements.txt
    set ANTHROPIC_API_KEY=your_key_here   (PowerShell: $env:ANTHROPIC_API_KEY="your_key_here")
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from anthropic import Anthropic

DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_SYSTEM = "You are a helpful assistant."
LOG_PATH = Path(__file__).parent / "log.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send a prompt to Claude with configurable system prompt, "
                     "temperature, and streaming."
    )
    parser.add_argument("prompt", help="The user prompt to send")
    parser.add_argument(
        "--system", default=DEFAULT_SYSTEM,
        help=f"System prompt (default: {DEFAULT_SYSTEM!r})",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Sampling temperature, 0.0-1.0 (default: 1.0)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=500,
        help="Max tokens in the response (default: 500)",
    )
    parser.add_argument(
        "--stream", action="store_true",
        help="Stream the response token-by-token instead of waiting for it all",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--no-log", action="store_true",
        help="Skip writing this call to log.jsonl",
    )
    return parser.parse_args()


def call_streaming(client: Anthropic, args: argparse.Namespace) -> str:
    """Stream the response to stdout as it arrives; return the full text."""
    full_text = []
    with client.messages.stream(
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        system=args.system,
        messages=[{"role": "user", "content": args.prompt}],
    ) as stream:
        for chunk in stream.text_stream:
            print(chunk, end="", flush=True)
            full_text.append(chunk)
    print()  # trailing newline after the stream finishes
    return "".join(full_text)


def call_blocking(client: Anthropic, args: argparse.Namespace) -> str:
    """Make a normal (non-streaming) call and return the full text."""
    response = client.messages.create(
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        system=args.system,
        messages=[{"role": "user", "content": args.prompt}],
    )
    return response.content[0].text


def log_call(args: argparse.Namespace, response_text: str) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "system": args.system,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "stream": args.stream,
        "prompt": args.prompt,
        "response": response_text,
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    client = Anthropic(api_key=api_key)

    if args.stream:
        response_text = call_streaming(client, args)
    else:
        response_text = call_blocking(client, args)
        print(response_text)

    if not args.no_log:
        log_call(args, response_text)


if __name__ == "__main__":
    main()
