"""
Day 01: minimal script to call an LLM API (Anthropic Claude example).

Setup:
    pip install anthropic
    set ANTHROPIC_API_KEY=your_key_here   (PowerShell: $env:ANTHROPIC_API_KEY="your_key_here")
"""

import os
from anthropic import Anthropic

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def ask(prompt: str) -> str:
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


if __name__ == "__main__":
    question = "Explain what a token is in the context of LLMs, in two sentences."
    print(ask(question))
