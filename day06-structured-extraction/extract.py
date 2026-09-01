"""
Structured Data Extractor.

Turns messy free text into validated JSON matching a schema. The interesting
part isn't the prompt -- it's what happens when the model returns something
that doesn't validate: we feed the error back and let it try again.

Usage:
    python extract.py "Sarah Chen, 34, works as a data engineer in Berlin."
    python extract.py --demo
    python extract.py "..." --max-retries 5 --verbose

Setup:
    pip install -r requirements.txt
    set ANTHROPIC_API_KEY=your_key_here   (PowerShell: $env:ANTHROPIC_API_KEY="your_key_here")
"""

import argparse
import json
import os
import re
import sys
from typing import Optional

from anthropic import Anthropic
from pydantic import BaseModel, Field, ValidationError

DEFAULT_MODEL = "claude-sonnet-5"


# ---------------------------------------------------------------------------
# The schema we want back. Pydantic gives us both the JSON Schema to put in
# the prompt AND the validator to check the response against.
# ---------------------------------------------------------------------------

class Person(BaseModel):
    name: str = Field(description="Full name of the person")
    age: Optional[int] = Field(default=None, ge=0, le=130,
                               description="Age in years, null if not stated")
    job: Optional[str] = Field(default=None,
                               description="Job title, null if not stated")
    city: Optional[str] = Field(default=None,
                                description="City they live in, null if not stated")
    email: Optional[str] = Field(default=None,
                                 description="Email address, null if not stated")


class ExtractionError(RuntimeError):
    """Raised when the model never produced valid output within the retry budget."""


# ---------------------------------------------------------------------------
# Parsing: models like to wrap JSON in prose or markdown fences
# ---------------------------------------------------------------------------

FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json_block(text: str) -> str:
    """Pull the JSON object out of a model response.

    Handles three common shapes: a bare object, an object inside a ```json
    fence, and an object with chatty text around it.
    """
    text = text.strip()

    fenced = FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()

    # Fall back to the outermost {...} span if there's still prose around it.
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            text = text[start:end + 1]

    return text


def parse_and_validate(raw: str) -> Person:
    """Parse model output into a validated Person, or raise."""
    payload = json.loads(extract_json_block(raw))
    return Person.model_validate(payload)


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------

def build_prompt(text: str) -> str:
    schema = json.dumps(Person.model_json_schema(), indent=2)
    return (
        "Extract structured information about the person described below.\n\n"
        f"Text:\n{text}\n\n"
        f"Return a single JSON object matching this schema:\n{schema}\n\n"
        "Rules:\n"
        "- Respond with JSON only, no commentary and no markdown fences.\n"
        "- Use null for any field the text does not state. Do not guess.\n"
    )


def build_retry_prompt(text: str, bad_output: str, error: str) -> str:
    return (
        f"{build_prompt(text)}\n"
        f"Your previous response was rejected.\n\n"
        f"Previous response:\n{bad_output}\n\n"
        f"Error:\n{error}\n\n"
        "Return corrected JSON only."
    )


# ---------------------------------------------------------------------------
# The extraction loop
# ---------------------------------------------------------------------------

def extract_person(client, text: str, model: str = DEFAULT_MODEL,
                   max_retries: int = 3, verbose: bool = False) -> Person:
    """Ask the model for structured data, retrying with the error on failure."""
    prompt = build_prompt(text)
    last_output = ""

    for attempt in range(1, max_retries + 1):
        response = client.messages.create(
            model=model,
            max_tokens=500,
            temperature=0,          # deterministic: we want the schema, not creativity
            messages=[{"role": "user", "content": prompt}],
        )
        last_output = response.content[0].text

        try:
            person = parse_and_validate(last_output)
            if verbose:
                print(f"[attempt {attempt}] valid", file=sys.stderr)
            return person
        except (json.JSONDecodeError, ValidationError) as exc:
            if verbose:
                print(f"[attempt {attempt}] rejected: {exc}", file=sys.stderr)
            if attempt == max_retries:
                raise ExtractionError(
                    f"No valid output after {max_retries} attempts. "
                    f"Last error: {exc}"
                ) from exc
            prompt = build_retry_prompt(text, last_output, str(exc))

    raise ExtractionError("unreachable")


DEMO_INPUTS = [
    "Sarah Chen, 34, works as a data engineer in Berlin. Reach her at sarah.c@example.com.",
    "my brother tom is a nurse, he's 29",
    "Dr. Amara Okafor (cardiologist) — based in Lagos.",
    "just some text with no person in it at all",
    "ANNA KOWALSKI // 41 // ARCHITECT // WARSAW // a.kowalski@example.pl",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract validated structured data from messy text."
    )
    parser.add_argument("text", nargs="?", help="Text to extract a person from")
    parser.add_argument("--demo", action="store_true",
                        help="Run against a set of built-in messy examples")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--verbose", action="store_true",
                        help="Show each attempt and why it was rejected")
    args = parser.parse_args()

    if not args.text and not args.demo:
        parser.error("provide some text, or use --demo")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    client = Anthropic(api_key=api_key)
    inputs = DEMO_INPUTS if args.demo else [args.text]

    failures = 0
    for text in inputs:
        print(f"\nInput: {text}")
        try:
            person = extract_person(client, text, args.model,
                                    args.max_retries, args.verbose)
            print(person.model_dump_json(indent=2))
        except ExtractionError as exc:
            failures += 1
            print(f"FAILED: {exc}", file=sys.stderr)

    if args.demo:
        print(f"\n{len(inputs) - failures}/{len(inputs)} extracted successfully.")

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
