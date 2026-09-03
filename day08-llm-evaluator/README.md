# LLM Output Evaluator

Runs a model over a set of test cases and scores the answers two ways —
a deterministic automated check and an LLM-as-judge — then reports pass rate,
average score, and the worst cases.

## Why this project

"Did that prompt change help?" is unanswerable by vibes. You need a number you
can compare before and after. This is the smallest honest version of that:
a fixed dataset, two scorers with different failure modes, and a report.

The reason for two scorers is that each is wrong in a way the other catches:

- **Keyword matching** is free, instant, and reproducible — but shallow. It
  can't tell a paraphrase from a miss, and it will happily pass
  *"the capital is definitely **not** Paris"*.
- **LLM-as-judge** understands meaning — but costs an API call per case, isn't
  deterministic, and is itself a model that can be wrong.

A real eval suite uses cheap checks for the obvious regressions and a judge for
the fuzzy cases. Neither alone is enough.

## The report

Here's the output from a run (a deliberately imperfect system under test):

```
============================================================
EVALUATION REPORT
============================================================
Cases:            10
Automated pass:   80% (8/10)
Avg judge score:  4.70 / 5

Worst 3 case(s):

  Q: In git, what does 'git fetch' do that 'git pull' does not?
  Expected: Fetch downloads remote changes without merging them; pull fetches and then merges.
  Got:      Fetch only downloads changes.
  Keywords: 0%   Judge: 3/5
  Judge says: Incomplete: does not mention merging.

  Q: What is the boiling point of water at sea level in Celsius?
  Expected: 100 degrees Celsius
  Got:      One hundred degrees.
  Keywords: 0%   Judge: 5/5
  Judge says: Correct, just spelled out.
...
```

That second case is the whole argument in miniature. The answer is **correct** —
"One hundred degrees" — but it fails the keyword check because the dataset asked
for the string `100`. The judge catches what the automated check can't. If you
only had the keyword score, you'd be chasing a bug that doesn't exist.

## Setup

```bash
pip install -r requirements.txt
```

```powershell
$env:ANTHROPIC_API_KEY="your_key_here"
```

## Usage

```bash
python evaluate.py
```

Automated checks only — no judge calls, so half the cost:

```bash
python evaluate.py --no-judge
```

Compare two system prompts by running each and diffing the numbers:

```bash
python evaluate.py --system "Answer in one short sentence."
```

```bash
python evaluate.py --system "Answer thoroughly, with detail."
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `dataset.json` | Test cases to run |
| `--system` | concise/factual | System prompt for the system under test |
| `--no-judge` | off | Skip the LLM judge |
| `--worst` | `3` | How many failing cases to show |
| `--verbose` | off | Print each case as it runs |
| `--model` | `claude-sonnet-5` | Model to use |

Exits non-zero if any case fails the automated check, so it drops straight into CI.

## The dataset

`dataset.json` holds 10 cases, each with a question, a reference answer, and the
keywords the automated check looks for:

```json
{
  "question": "What is the capital of France?",
  "expected": "Paris",
  "keywords": ["paris"]
}
```

Add your own cases here — especially ones your system has gotten wrong before.
That's how an eval set earns its keep: it becomes a regression test.

## Tests

```bash
python test_evaluate.py
```

29 checks covering scoring, judge parsing (clean JSON, fenced JSON, regex
fallback, out-of-range scores, unparsable output), aggregation, worst-case
ranking, and the evaluation loop. One test deliberately asserts the *negation
bug* in keyword scoring — documenting the limitation rather than pretending
it isn't there.

## What I learned

- **`run_evaluation` takes the system and the judge as arguments.** That one
  choice makes the whole thing testable without an API key, and means you can
  point it at a different system (say, Day 02's RAG) without touching it.
- **Judges need a rubric, not a vibe.** "Rate 1-5" gives mush. Spelling out what
  each number means, and saying "judge meaning, not wording", makes scores far
  more stable.
- **`temperature=0` for the judge.** An evaluation that returns different
  numbers on reruns is not an evaluation.
- **Parse the judge defensively too.** It's just another model call, with the
  same tendency to wrap JSON in prose — hence the regex fallback.
- **Aggregate scores hide the useful information.** "4.7/5" tells you nothing
  actionable. The three worst cases tell you exactly what to fix, which is why
  the report leads with them.

## Ideas to extend

- Point it at the Day 02 RAG system and score retrieval quality
- Add a groundedness check: is every claim supported by the retrieved context?
- Track results over time so you can see regressions between prompt versions
- Report token cost and latency per case alongside quality
- Add a second judge and measure how often the two agree
