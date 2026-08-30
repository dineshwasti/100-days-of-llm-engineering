"""Tests for the conversation memory and slash commands.

These cover everything that doesn't need the API, which is most of the
interesting logic: history tracking, reset, transcript rendering, and
command dispatch.

Run with:
    python test_chat.py
"""

import sys
import tempfile
from pathlib import Path

from chat import Conversation, handle_command, save_transcript

failures = []


def check(label, condition):
    if not condition:
        failures.append(label)
        print("FAIL", label)


def test_history_accumulates():
    c = Conversation()
    c.add_user("hello")
    c.add_assistant("hi there")
    c.add_user("what did I just say?")
    check("messages accumulate", len(c.messages) == 3)
    check("turn count counts user messages", c.turns == 2)
    check("roles alternate correctly",
          [m["role"] for m in c.messages] == ["user", "assistant", "user"])


def test_reset_clears():
    c = Conversation()
    c.add_user("hello")
    c.add_assistant("hi")
    c.reset()
    check("reset clears messages", c.messages == [])
    check("reset zeroes turns", c.turns == 0)


def test_transcript_contains_content():
    c = Conversation(system="Be terse.")
    c.add_user("ping")
    c.add_assistant("pong")
    md = c.to_markdown()
    check("transcript has user text", "ping" in md)
    check("transcript has assistant text", "pong" in md)
    check("transcript records system prompt", "Be terse." in md)
    check("transcript labels speakers", "**You:**" in md and "**Assistant:**" in md)


def test_save_writes_file():
    c = Conversation()
    c.add_user("save me")
    c.add_assistant("saved")
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "out.md"
        path = save_transcript(c, str(target))
        check("save returns the path", path == target)
        check("file exists", target.exists())
        check("file has content", "save me" in target.read_text(encoding="utf-8"))


def test_commands():
    c = Conversation()
    c.add_user("hello")
    c.add_assistant("hi")

    check("/exit signals exit", handle_command("/exit", c) == "exit")
    check("/quit signals exit", handle_command("/quit", c) == "exit")
    check("/help is handled", handle_command("/help", c) == "handled")
    check("/history is handled", handle_command("/history", c) == "handled")
    check("unknown command is handled, not crashing",
          handle_command("/nonsense", c) == "handled")

    check("history intact before reset", len(c.messages) == 2)
    handle_command("/reset", c)
    check("/reset empties history", c.messages == [])


def test_save_command_with_filename():
    c = Conversation()
    c.add_user("hello")
    c.add_assistant("hi")
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "named.md"
        handle_command(f"/save {target}", c)
        check("/save honours the filename argument", target.exists())


def test_save_command_with_empty_history():
    c = Conversation()
    check("/save on empty history doesn't crash",
          handle_command("/save", c) == "handled")


def test_commands_are_case_insensitive():
    c = Conversation()
    check("/EXIT works uppercase", handle_command("/EXIT", c) == "exit")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()

    print()
    if failures:
        print(f"{len(failures)} FAILURE(S):", failures)
        sys.exit(1)
    print("All tests passed.")
