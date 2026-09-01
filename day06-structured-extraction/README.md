# Structured Data Extractor

Turns messy free text into **validated** JSON matching a schema — and when the
model returns something invalid, feeds the error back and asks it to try again.

## Why this project

Getting an LLM to produce prose is easy. Getting it to produce data your
program can actually consume is the part that breaks in production. Models
wrap JSON in markdown fences, add a friendly "Sure, here you go!", invent
fields, or return an age of 500.

So the useful pattern isn't "write a better prompt" — it's **never trust the
output**: parse defensively, validate against a real schema, and treat a
failure as a retry with the error attached rather than a crash. That loop is
the same one that later shows up around tool calls and agent steps.

## How it works

1. A `pydantic` model defines the target schema (`Person`)
2. Its JSON Schema goes straight into the prompt — one source of truth
3. The response is parsed defensively (handles fences and surrounding prose)
4. It's validated against the same model
5. On failure, the bad output **and the validation error** go back to the model
6. Give up after a retry budget rather than looping forever

## Setup

```bash
pip install -r requirements.txt
```

```powershell
$env:ANTHROPIC_API_KEY="your_key_here"
```

## Usage

```bash
python extract.py "Sarah Chen, 34, works as a data engineer in Berlin. Reach her at sarah.c@example.com."
```

```json
{
  "name": "Sarah Chen",
  "age": 34,
  "job": "data engineer",
  "city": "Berlin",
  "email": "sarah.c@example.com"
}
```

Run against a set of deliberately messy inputs — including one with no person
in it at all:

```bash
python extract.py --demo
```

Watch the retry loop work:

```bash
python extract.py "my brother tom is a nurse, he's 29" --verbose
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `text` | — | The text to extract from (omit if using `--demo`) |
| `--demo` | off | Run against built-in messy examples |
| `--max-retries` | `3` | How many attempts before giving up |
| `--verbose` | off | Print each attempt and why it was rejected |
| `--model` | `claude-sonnet-5` | Model to use |

## Tests

```bash
python test_extract.py
```

24 checks covering the parts that don't need the API: fence/prose stripping,
nested braces, schema validation (missing required fields, out-of-range age,
malformed JSON), prompt construction, and the retry loop itself — including
that it recovers from both malformed JSON *and* a validation error, and that
it gives up cleanly once the budget is spent.

## What I learned

- **`temperature=0` for extraction.** You want the schema followed, not
  creative variation. Creativity is the enemy here.
- **Pydantic pays for itself twice.** `model_json_schema()` generates the
  schema for the prompt, and the same class validates the response — so the
  prompt and the validator can never drift apart.
- **Parse defensively.** Even told "no markdown fences", models sometimes add
  them. Stripping fences and falling back to the outermost `{...}` costs a few
  lines and removes a whole class of failure.
- **The error message is the best retry prompt.** Just handing back
  `"age must be <= 130"` is far more effective than re-asking politely,
  because it says precisely what to fix.
- **Bound your retries.** An unbounded "keep trying until valid" loop is a way
  to spend real money on an input that will never work.

## Ideas to extend

- Extract a `list[Person]` when the text mentions several people
- Return `None` cleanly for text containing no person, instead of forcing a guess
- Swap the hand-rolled loop for the API's native tool-use / structured output
- Track how often each attempt number succeeds, to see if retries actually help
- Add a confidence field and check whether the model is calibrated
