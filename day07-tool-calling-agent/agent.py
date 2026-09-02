"""
Tool-Calling Agent.

The model is given three tools and decides for itself which (if any) to call.
We run whatever it asks for, hand the result back, and let it continue until
it has a final answer.

Usage:
    python agent.py "What's 1847 * 23?"
    python agent.py "What's the weather in Tokyo?"
    python agent.py "How many days off do I get?"
    python agent.py "Who wrote Hamlet?"          # needs no tool at all
    python agent.py --demo
    python agent.py "..." --verbose              # show the tool calls

Setup:
    pip install -r requirements.txt
    set ANTHROPIC_API_KEY=your_key_here   (PowerShell: $env:ANTHROPIC_API_KEY="your_key_here")
"""

import argparse
import os
import sys

from anthropic import Anthropic

from tools import TOOL_SCHEMAS, run_tool

DEFAULT_MODEL = "claude-sonnet-5"
MAX_STEPS = 6

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools. "
    "Use a tool when it genuinely helps -- for arithmetic, weather lookups, or "
    "questions about company policy. For general knowledge you already have, "
    "just answer directly instead of forcing a tool call. "
    "If a tool returns an error, explain the problem to the user rather than "
    "retrying the same call."
)


class AgentResult:
    def __init__(self, answer: str, tool_calls: list[dict], steps: int):
        self.answer = answer
        self.tool_calls = tool_calls
        self.steps = steps


def run_agent(client, question: str, model: str = DEFAULT_MODEL,
              max_steps: int = MAX_STEPS, verbose: bool = False) -> AgentResult:
    """Run the agent loop until the model stops asking for tools.

    The loop is the whole idea: call the model, and if it came back asking for
    a tool, run it, append the result to the conversation, and call again. The
    model drives; we just execute and report back.
    """
    messages = [{"role": "user", "content": question}]
    tool_calls = []

    for step in range(1, max_steps + 1):
        response = client.messages.create(
            model=model,
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            answer = "".join(block.text for block in response.content
                             if getattr(block, "type", None) == "text")
            return AgentResult(answer.strip(), tool_calls, step)

        # The model asked for one or more tools. Record its request verbatim,
        # then reply with a result block for every tool_use id it sent -- the
        # API requires one result per call, in the same turn.
        messages.append({"role": "assistant", "content": response.content})

        results = []
        for block in response.content:
            if getattr(block, "type", None) != "tool_use":
                continue

            output, is_error = run_tool(block.name, block.input)
            tool_calls.append({"name": block.name, "input": block.input,
                               "output": output, "is_error": is_error})

            if verbose:
                marker = "!" if is_error else "-"
                print(f"  [{marker}] {block.name}({block.input}) -> {output}",
                      file=sys.stderr)

            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": output,
                "is_error": is_error,
            })

        messages.append({"role": "user", "content": results})

    raise RuntimeError(f"agent did not finish within {max_steps} steps")


DEMO_QUESTIONS = [
    "What is 1847 * 23?",
    "What's the weather in Tokyo?",
    "How many days of paid time off do I get?",
    "Who wrote Hamlet?",                      # should use no tool
    "What's the weather on Mars?",            # tool will error; agent should cope
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="An agent that decides which tool to call, if any."
    )
    parser.add_argument("question", nargs="?", help="What to ask")
    parser.add_argument("--demo", action="store_true",
                        help="Run a set of questions covering each tool")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--verbose", action="store_true",
                        help="Print each tool call and its result")
    args = parser.parse_args()

    if not args.question and not args.demo:
        parser.error("ask a question, or use --demo")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    client = Anthropic(api_key=api_key)
    questions = DEMO_QUESTIONS if args.demo else [args.question]

    for question in questions:
        print(f"\nQ: {question}")
        result = run_agent(client, question, args.model, args.max_steps, args.verbose)
        used = ", ".join(call["name"] for call in result.tool_calls) or "none"
        print(f"A: {result.answer}")
        print(f"   (tools used: {used}; {result.steps} model call(s))")


if __name__ == "__main__":
    main()
