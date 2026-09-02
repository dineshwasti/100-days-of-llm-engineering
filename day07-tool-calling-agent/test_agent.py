"""Tests for the tools and the agent loop.

The agent loop is tested against a fake client that returns scripted
tool_use / text responses, so the whole multi-step flow is exercised without
touching the API.

Run with:
    python test_agent.py
"""

import sys

from agent import run_agent
from tools import ToolError, calculate, get_weather, run_tool, search_docs

failures = []


def check(label, condition):
    if not condition:
        failures.append(label)
        print("FAIL", label)


# --- fakes -----------------------------------------------------------------

class TextBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class ToolUseBlock:
    type = "tool_use"

    def __init__(self, name, input, id="tu_1"):
        self.name = name
        self.input = input
        self.id = id


class FakeResponse:
    def __init__(self, content, stop_reason):
        self.content = content
        self.stop_reason = stop_reason


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("fake client called more times than scripted")
        return self.responses.pop(0)


# --- calculate -------------------------------------------------------------

def test_calculate_basic():
    check("addition", calculate("2 + 3") == "5")
    check("precedence", calculate("2 + 3 * 4") == "14")
    check("parens", calculate("(2 + 3) * 4") == "20")
    check("negative", calculate("-5 + 2") == "-3")
    check("float division", calculate("7 / 2") == "3.5")
    check("floor division", calculate("7 // 2") == "3")
    check("power", calculate("2 ** 10") == "1024")


def test_calculate_rejects_code_execution():
    """The important one: model output is untrusted input."""
    for dangerous in ["__import__('os').system('echo pwned')",
                      "open('/etc/passwd').read()",
                      "[].__class__.__base__.__subclasses__()",
                      "exec('x=1')",
                      "lambda: 1"]:
        try:
            calculate(dangerous)
            check(f"rejects {dangerous[:25]}", False)
        except ToolError:
            check(f"rejects {dangerous[:25]}", True)


def test_calculate_rejects_huge_exponent():
    try:
        calculate("9 ** 999999999")
        check("rejects huge exponent", False)
    except ToolError:
        check("rejects huge exponent", True)


def test_calculate_handles_bad_input():
    for bad in ["2 +", "hello", ""]:
        try:
            calculate(bad)
            check(f"rejects malformed {bad!r}", False)
        except ToolError:
            check(f"rejects malformed {bad!r}", True)


def test_calculate_division_by_zero():
    try:
        calculate("1 / 0")
        check("division by zero raises ToolError", False)
    except ToolError:
        check("division by zero raises ToolError", True)


# --- get_weather / search_docs ---------------------------------------------

def test_get_weather():
    check("known city", "Tokyo" in get_weather("tokyo"))
    check("case insensitive", "Tokyo" in get_weather("  TOKYO "))


def test_get_weather_unknown_city():
    try:
        get_weather("Mars")
        check("unknown city raises", False)
    except ToolError as exc:
        check("unknown city raises", True)
        check("error lists known cities", "tokyo" in str(exc))


def test_search_docs():
    check("finds pto", "20 days" in search_docs("how much pto do I get"))
    check("finds insurance", "insurance" in search_docs("insurance").lower())
    check("no match is graceful", "No matching" in search_docs("zzzz"))


# --- run_tool dispatch -----------------------------------------------------

def test_run_tool_success():
    output, is_error = run_tool("calculate", {"expression": "6 * 7"})
    check("dispatch works", output == "42" and is_error is False)


def test_run_tool_unknown_name():
    output, is_error = run_tool("nonexistent", {})
    check("unknown tool flagged", is_error is True and "Unknown tool" in output)


def test_run_tool_returns_errors_not_raises():
    output, is_error = run_tool("get_weather", {"city": "Atlantis"})
    check("tool error returned not raised", is_error is True)
    check("error text is useful", "no weather data" in output)


def test_run_tool_bad_arguments():
    output, is_error = run_tool("calculate", {"wrong_arg": "1"})
    check("bad kwargs handled", is_error is True and "bad arguments" in output)


# --- the agent loop --------------------------------------------------------

def test_answers_without_tools():
    client = FakeClient([FakeResponse([TextBlock("Shakespeare wrote Hamlet.")], "end_turn")])
    result = run_agent(client, "Who wrote Hamlet?")
    check("direct answer returned", result.answer == "Shakespeare wrote Hamlet.")
    check("no tools used", result.tool_calls == [])
    check("one model call", result.steps == 1)


def test_calls_a_tool_then_answers():
    client = FakeClient([
        FakeResponse([ToolUseBlock("calculate", {"expression": "1847 * 23"})], "tool_use"),
        FakeResponse([TextBlock("1847 * 23 = 42481.")], "end_turn"),
    ])
    result = run_agent(client, "What is 1847 * 23?")
    check("final answer returned", "42481" in result.answer)
    check("one tool call recorded", len(result.tool_calls) == 1)
    check("tool actually executed", result.tool_calls[0]["output"] == "42481")
    check("two model calls", result.steps == 2)


def test_tool_result_is_sent_back():
    client = FakeClient([
        FakeResponse([ToolUseBlock("calculate", {"expression": "2 + 2"}, id="tu_abc")], "tool_use"),
        FakeResponse([TextBlock("4")], "end_turn"),
    ])
    run_agent(client, "2+2?")

    second_call_messages = client.calls[1]["messages"]
    check("history has 3 messages", len(second_call_messages) == 3)
    check("assistant turn recorded", second_call_messages[1]["role"] == "assistant")

    result_block = second_call_messages[2]["content"][0]
    check("result block type", result_block["type"] == "tool_result")
    check("tool_use_id matches", result_block["tool_use_id"] == "tu_abc")
    check("result content correct", result_block["content"] == "4")


def test_handles_multiple_tools_in_one_turn():
    client = FakeClient([
        FakeResponse([ToolUseBlock("get_weather", {"city": "Tokyo"}, id="a"),
                      ToolUseBlock("get_weather", {"city": "London"}, id="b")], "tool_use"),
        FakeResponse([TextBlock("Tokyo is clear, London is raining.")], "end_turn"),
    ])
    result = run_agent(client, "Weather in Tokyo and London?")
    check("both tools ran", len(result.tool_calls) == 2)

    results_sent = client.calls[1]["messages"][2]["content"]
    check("one result per call", len(results_sent) == 2)
    check("ids preserved in order",
          [r["tool_use_id"] for r in results_sent] == ["a", "b"])


def test_tool_error_is_reported_to_model():
    client = FakeClient([
        FakeResponse([ToolUseBlock("get_weather", {"city": "Mars"})], "tool_use"),
        FakeResponse([TextBlock("I don't have weather data for Mars.")], "end_turn"),
    ])
    result = run_agent(client, "Weather on Mars?")
    check("error recorded", result.tool_calls[0]["is_error"] is True)

    result_block = client.calls[1]["messages"][2]["content"][0]
    check("is_error flag sent to model", result_block["is_error"] is True)
    check("agent still answers", "Mars" in result.answer)


def test_stops_at_max_steps():
    looping = [FakeResponse([ToolUseBlock("calculate", {"expression": "1+1"})], "tool_use")] * 3
    client = FakeClient(looping)
    try:
        run_agent(client, "loop forever", max_steps=3)
        check("raises when it never finishes", False)
    except RuntimeError:
        check("raises when it never finishes", True)


def test_tools_are_offered_to_the_model():
    client = FakeClient([FakeResponse([TextBlock("hi")], "end_turn")])
    run_agent(client, "hi")
    names = [t["name"] for t in client.calls[0]["tools"]]
    check("all three tools offered",
          sorted(names) == ["calculate", "get_weather", "search_docs"])


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()

    print()
    if failures:
        print(f"{len(failures)} FAILURE(S):", failures)
        sys.exit(1)
    print("All tests passed.")
