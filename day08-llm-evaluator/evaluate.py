"""
LLM Output Evaluator.

Runs a model over a set of test cases and scores the answers two ways:

  1. an automated check  -- deterministic, free, but shallow
  2. an LLM-as-judge     -- catches paraphrase, but costs money and can be wrong

Then reports pass rate, average judge score, and the worst cases, so you can
tell whether a prompt change actually helped instead of guessing.

Usage:
    python evaluate.py
    python evaluate.py --dataset dataset.json --verbose
    python evaluate.py --no-judge          # automated checks only, no extra calls
    python evaluate.py --system "Answer in one short sentence."

Setup:
    pip install -r requirements.txt
    set ANTHROPIC_API_KEY=your_key_here   (PowerShell: $env:ANTHROPIC_API_KEY="your_key_here")
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from anthropic import Anthropic

DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_DATASET = Path(__file__).parent / "dataset.json"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    question: str
    expected: str
    keywords: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "TestCase":
        return cls(question=data["question"],
                   expected=data["expected"],
                   keywords=data.get("keywords", []))


@dataclass
class Result:
    case: TestCase
    answer: str
    keyword_score: float                 # 0.0 - 1.0
    judge_score: Optional[int] = None    # 1 - 5
    judge_reason: str = ""

    @property
    def passed(self) -> bool:
        """Automated pass: every required keyword showed up."""
        return self.keyword_score == 1.0

    @property
    def combined(self) -> float:
        """Used only for ranking the worst cases."""
        if self.judge_score is None:
            return self.keyword_score
        return (self.keyword_score + (self.judge_score - 1) / 4) / 2


def load_dataset(path: Path) -> list[TestCase]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [TestCase.from_dict(item) for item in data]


# ---------------------------------------------------------------------------
# Scorer 1: automated keyword check
# ---------------------------------------------------------------------------

def keyword_score(answer: str, keywords: list[str]) -> float:
    """Fraction of required keywords present in the answer.

    Crude on purpose. It can't tell "Paris" from "not Paris", and it misses
    correct paraphrases -- which is exactly why the judge exists too. What it
    does give you is a deterministic, free signal that never drifts.
    """
    if not keywords:
        return 1.0
    lowered = answer.lower()
    hits = sum(1 for keyword in keywords if keyword.lower() in lowered)
    return hits / len(keywords)


# ---------------------------------------------------------------------------
# Scorer 2: LLM as judge
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """You are grading an answer against a reference answer.

Question:
{question}

Reference answer:
{expected}

Answer to grade:
{answer}

Grade correctness from 1 to 5:
5 = fully correct, matches the reference
4 = correct but with minor omissions
3 = partially correct
2 = mostly wrong
1 = completely wrong or off-topic

Judge meaning, not wording -- a correct paraphrase scores 5.
Respond with JSON only: {{"score": <1-5>, "reason": "<one sentence>"}}"""

SCORE_RE = re.compile(r'"score"\s*:\s*([1-5])')


def parse_judge_response(raw: str) -> tuple[Optional[int], str]:
    """Pull a score and reason out of the judge's reply.

    Falls back to a regex when the JSON is malformed, because a judge that
    occasionally wraps its output in prose shouldn't sink the whole run.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()

    try:
        payload = json.loads(text)
        score = int(payload["score"])
        if 1 <= score <= 5:
            return score, str(payload.get("reason", ""))
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        pass

    match = SCORE_RE.search(raw)
    if match:
        return int(match.group(1)), "(reason unparsed)"

    return None, "(judge response unparsable)"


def judge(client, case: TestCase, answer: str, model: str) -> tuple[Optional[int], str]:
    response = client.messages.create(
        model=model,
        max_tokens=200,
        temperature=0,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            question=case.question, expected=case.expected, answer=answer)}],
    )
    return parse_judge_response(response.content[0].text)


# ---------------------------------------------------------------------------
# Running the evaluation
# ---------------------------------------------------------------------------

def make_answerer(client, model: str, system: str) -> Callable[[str], str]:
    """The system under test. Swap this out to evaluate something else."""
    def answer(question: str) -> str:
        response = client.messages.create(
            model=model,
            max_tokens=300,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": question}],
        )
        return response.content[0].text.strip()
    return answer


def run_evaluation(cases: list[TestCase], answer_fn: Callable[[str], str],
                   judge_fn: Optional[Callable[[TestCase, str], tuple]] = None,
                   verbose: bool = False) -> list[Result]:
    results = []
    for index, case in enumerate(cases, 1):
        answer = answer_fn(case.question)
        result = Result(case=case, answer=answer,
                        keyword_score=keyword_score(answer, case.keywords))

        if judge_fn is not None:
            result.judge_score, result.judge_reason = judge_fn(case, answer)

        results.append(result)

        if verbose:
            mark = "PASS" if result.passed else "FAIL"
            score = result.judge_score if result.judge_score is not None else "-"
            print(f"  [{index}/{len(cases)}] {mark} judge={score}  {case.question}",
                  file=sys.stderr)

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def pass_rate(results: list[Result]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.passed) / len(results)


def average_judge_score(results: list[Result]) -> Optional[float]:
    scored = [r.judge_score for r in results if r.judge_score is not None]
    if not scored:
        return None
    return sum(scored) / len(scored)


def worst(results: list[Result], n: int = 3) -> list[Result]:
    return sorted(results, key=lambda r: r.combined)[:n]


def format_report(results: list[Result], worst_n: int = 3) -> str:
    lines = ["", "=" * 60, "EVALUATION REPORT", "=" * 60,
             f"Cases:            {len(results)}",
             f"Automated pass:   {pass_rate(results):.0%} "
             f"({sum(1 for r in results if r.passed)}/{len(results)})"]

    average = average_judge_score(results)
    if average is not None:
        lines.append(f"Avg judge score:  {average:.2f} / 5")

    failures = [r for r in results if not r.passed]
    lines.append("")
    lines.append(f"Worst {min(worst_n, len(results))} case(s):")
    for result in worst(results, worst_n):
        lines.append("")
        lines.append(f"  Q: {result.case.question}")
        lines.append(f"  Expected: {result.case.expected}")
        lines.append(f"  Got:      {result.answer[:160]}")
        lines.append(f"  Keywords: {result.keyword_score:.0%}"
                     + (f"   Judge: {result.judge_score}/5" if result.judge_score else ""))
        if result.judge_reason:
            lines.append(f"  Judge says: {result.judge_reason}")

    lines.append("")
    lines.append(f"{len(failures)} case(s) failed the automated check.")
    lines.append("=" * 60)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate LLM answers with automated checks and an LLM judge."
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--system", default="Answer the question concisely and factually.")
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip the LLM judge (halves the API calls)")
    parser.add_argument("--worst", type=int, default=3, help="How many bad cases to show")
    parser.add_argument("--verbose", action="store_true", help="Print each case as it runs")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    cases = load_dataset(Path(args.dataset))
    client = Anthropic(api_key=api_key)

    answer_fn = make_answerer(client, args.model, args.system)
    judge_fn = None if args.no_judge else (
        lambda case, answer: judge(client, case, answer, args.model))

    results = run_evaluation(cases, answer_fn, judge_fn, args.verbose)
    print(format_report(results, args.worst))

    sys.exit(0 if pass_rate(results) == 1.0 else 1)


if __name__ == "__main__":
    main()
