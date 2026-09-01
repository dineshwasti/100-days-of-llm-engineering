"""Tests for JSON parsing, schema validation, and the retry loop.

The retry behaviour is tested with a fake client that returns a scripted
sequence of responses, so we can prove the loop recovers from bad output
without touching the API.

Run with:
    python test_extract.py
"""

import sys

from pydantic import ValidationError

from extract import (
    ExtractionError,
    Person,
    build_prompt,
    build_retry_prompt,
    extract_json_block,
    extract_person,
    parse_and_validate,
)

failures = []


def check(label, condition):
    if not condition:
        failures.append(label)
        print("FAIL", label)


class FakeResponse:
    def __init__(self, text):
        self.content = [type("Block", (), {"text": text})()]


class FakeClient:
    """Returns a scripted list of responses, one per call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []
        self.messages = self

    def create(self, **kwargs):
        self.prompts.append(kwargs["messages"][0]["content"])
        if not self.responses:
            raise AssertionError("fake client called more times than scripted")
        return FakeResponse(self.responses.pop(0))


# --- parsing ---------------------------------------------------------------

def test_parses_bare_json():
    check("bare json", extract_json_block('{"name": "Ann"}') == '{"name": "Ann"}')


def test_parses_fenced_json():
    raw = '```json\n{"name": "Ann"}\n```'
    check("fenced json", extract_json_block(raw) == '{"name": "Ann"}')


def test_parses_unlabelled_fence():
    raw = '```\n{"name": "Ann"}\n```'
    check("unlabelled fence", extract_json_block(raw) == '{"name": "Ann"}')


def test_strips_surrounding_prose():
    raw = 'Sure! Here is the JSON:\n{"name": "Ann"}\nHope that helps.'
    check("prose stripped", extract_json_block(raw) == '{"name": "Ann"}')


def test_handles_nested_braces():
    raw = 'text {"name": "Ann", "meta": {"x": 1}} more text'
    check("nested braces", extract_json_block(raw) == '{"name": "Ann", "meta": {"x": 1}}')


# --- validation ------------------------------------------------------------

def test_valid_payload():
    p = parse_and_validate('{"name": "Ann", "age": 30, "job": "chef", '
                           '"city": "Oslo", "email": "a@example.com"}')
    check("name parsed", p.name == "Ann")
    check("age parsed", p.age == 30)


def test_optional_fields_default_to_none():
    p = parse_and_validate('{"name": "Ann"}')
    check("age defaults null", p.age is None)
    check("job defaults null", p.job is None)


def test_rejects_missing_required_field():
    try:
        parse_and_validate('{"age": 30}')
        check("missing name rejected", False)
    except ValidationError:
        check("missing name rejected", True)


def test_rejects_out_of_range_age():
    try:
        parse_and_validate('{"name": "Ann", "age": 999}')
        check("age 999 rejected", False)
    except ValidationError:
        check("age 999 rejected", True)


def test_rejects_malformed_json():
    try:
        parse_and_validate("{not json at all")
        check("malformed json rejected", False)
    except Exception:
        check("malformed json rejected", True)


# --- prompting -------------------------------------------------------------

def test_prompt_includes_schema_and_text():
    prompt = build_prompt("Ann is 30.")
    check("prompt has the text", "Ann is 30." in prompt)
    check("prompt has schema fields", '"name"' in prompt and '"email"' in prompt)


def test_retry_prompt_includes_the_error():
    retry = build_retry_prompt("Ann is 30.", "oops", "age must be <= 130")
    check("retry shows bad output", "oops" in retry)
    check("retry shows the error", "age must be <= 130" in retry)


# --- the retry loop --------------------------------------------------------

def test_succeeds_first_try():
    client = FakeClient(['{"name": "Ann", "age": 30}'])
    person = extract_person(client, "Ann is 30", max_retries=3)
    check("first try succeeds", person.name == "Ann")
    check("only one call made", len(client.prompts) == 1)


def test_recovers_from_malformed_json():
    client = FakeClient(["I'm not going to give you JSON",
                         '{"name": "Ann", "age": 30}'])
    person = extract_person(client, "Ann is 30", max_retries=3)
    check("recovered after bad json", person.name == "Ann")
    check("took two calls", len(client.prompts) == 2)
    check("second prompt contains the error feedback",
          "rejected" in client.prompts[1])


def test_recovers_from_validation_error():
    client = FakeClient(['{"name": "Ann", "age": 500}',
                         '{"name": "Ann", "age": 50}'])
    person = extract_person(client, "Ann is 50", max_retries=3)
    check("recovered after validation error", person.age == 50)
    check("bad value shown in retry prompt", "500" in client.prompts[1])


def test_gives_up_after_max_retries():
    client = FakeClient(["nope", "still nope"])
    try:
        extract_person(client, "Ann", max_retries=2)
        check("raises after budget exhausted", False)
    except ExtractionError:
        check("raises after budget exhausted", True)
    check("used exactly the retry budget", len(client.prompts) == 2)


def test_uses_temperature_zero():
    class RecordingClient(FakeClient):
        def create(self, **kwargs):
            self.last_kwargs = kwargs
            return super().create(**kwargs)

    client = RecordingClient(['{"name": "Ann"}'])
    extract_person(client, "Ann")
    check("temperature is 0", client.last_kwargs["temperature"] == 0)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()

    print()
    if failures:
        print(f"{len(failures)} FAILURE(S):", failures)
        sys.exit(1)
    print("All tests passed.")
