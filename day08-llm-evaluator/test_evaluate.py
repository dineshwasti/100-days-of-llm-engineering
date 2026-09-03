"""Tests for scoring, judge parsing, aggregation, and the evaluation run.

Everything here runs without the API: the system under test and the judge are
both plain functions, which is exactly why `run_evaluation` takes them as
arguments instead of building them itself.

Run with:
    python test_evaluate.py
"""

import json
import sys
import tempfile
from pathlib import Path

from evaluate import (
    DEFAULT_DATASET,
    Result,
    TestCase,
    average_judge_score,
    format_report,
    keyword_score,
    load_dataset,
    parse_judge_response,
    pass_rate,
    run_evaluation,
    worst,
)

failures = []


def check(label, condition):
    if not condition:
        failures.append(label)
        print("FAIL", label)


def make_result(keyword=1.0, judge=None, question="q", answer="a"):
    return Result(case=TestCase(question=question, expected="e"),
                  answer=answer, keyword_score=keyword, judge_score=judge)


# --- keyword scoring -------------------------------------------------------

def test_keyword_all_present():
    check("all keywords", keyword_score("The capital is Paris.", ["paris"]) == 1.0)


def test_keyword_partial():
    score = keyword_score("Lists are mutable.", ["mutable", "immutable"])
    check("partial credit", score == 0.5)


def test_keyword_none_present():
    check("no keywords matched", keyword_score("no idea", ["paris"]) == 0.0)


def test_keyword_case_insensitive():
    check("case insensitive", keyword_score("PARIS", ["paris"]) == 1.0)


def test_keyword_empty_list_passes():
    check("no keywords required", keyword_score("anything", []) == 1.0)


def test_keyword_scorer_is_fooled_by_negation():
    """Documents a real limitation rather than pretending it isn't there."""
    check("negation not detected",
          keyword_score("The capital is definitely not Paris.", ["paris"]) == 1.0)


# --- judge parsing ---------------------------------------------------------

def test_parse_clean_json():
    score, reason = parse_judge_response('{"score": 5, "reason": "correct"}')
    check("clean json score", score == 5)
    check("clean json reason", reason == "correct")


def test_parse_fenced_json():
    score, _ = parse_judge_response('```json\n{"score": 4, "reason": "close"}\n```')
    check("fenced json", score == 4)


def test_parse_falls_back_to_regex():
    raw = 'Sure! Here is my grade: {"score": 2, "reason": mostly wrong}'
    score, reason = parse_judge_response(raw)
    check("regex fallback score", score == 2)
    check("regex fallback flags reason", "unparsed" in reason)


def test_parse_rejects_out_of_range():
    score, _ = parse_judge_response('{"score": 9, "reason": "nonsense"}')
    check("score 9 rejected", score is None)


def test_parse_unparsable():
    score, reason = parse_judge_response("I refuse to grade this.")
    check("unparsable returns None", score is None)
    check("unparsable explains itself", "unparsable" in reason)


# --- aggregation -----------------------------------------------------------

def test_pass_rate():
    results = [make_result(1.0), make_result(1.0), make_result(0.5), make_result(0.0)]
    check("pass rate", pass_rate(results) == 0.5)


def test_pass_rate_empty():
    check("empty pass rate is 0", pass_rate([]) == 0.0)


def test_average_judge_score():
    results = [make_result(judge=5), make_result(judge=3), make_result(judge=4)]
    check("average judge", average_judge_score(results) == 4.0)


def test_average_ignores_unscored():
    results = [make_result(judge=5), make_result(judge=None), make_result(judge=3)]
    check("None judge scores ignored", average_judge_score(results) == 4.0)


def test_average_none_when_no_judge():
    check("no judge scores -> None",
          average_judge_score([make_result(judge=None)]) is None)


def test_worst_orders_by_combined_score():
    good = make_result(1.0, judge=5, question="good")
    middling = make_result(1.0, judge=2, question="middling")
    bad = make_result(0.0, judge=1, question="bad")
    ranked = worst([good, middling, bad], n=3)
    check("worst first", ranked[0].case.question == "bad")
    check("best last", ranked[-1].case.question == "good")


def test_worst_respects_n():
    results = [make_result(0.0) for _ in range(5)]
    check("worst limited to n", len(worst(results, n=2)) == 2)


# --- running ---------------------------------------------------------------

def test_run_evaluation_scores_each_case():
    cases = [TestCase("capital of France?", "Paris", ["paris"]),
             TestCase("2+2?", "4", ["4"])]

    answers = {"capital of France?": "It is Paris.", "2+2?": "It is five."}
    results = run_evaluation(cases, lambda q: answers[q])

    check("one result per case", len(results) == 2)
    check("correct answer passes", results[0].passed is True)
    check("wrong answer fails", results[1].passed is False)


def test_run_evaluation_uses_judge():
    cases = [TestCase("q", "e", ["missing"])]
    seen = []

    def fake_judge(case, answer):
        seen.append((case.question, answer))
        return 4, "close enough"

    results = run_evaluation(cases, lambda q: "an answer", fake_judge)
    check("judge was called", seen == [("q", "an answer")])
    check("judge score recorded", results[0].judge_score == 4)
    check("judge reason recorded", results[0].judge_reason == "close enough")


def test_judge_can_rescue_a_keyword_failure():
    """The point of having both scorers: a correct paraphrase fails the
    keyword check but the judge still recognises it."""
    cases = [TestCase("capital of France?", "Paris", ["paris"])]
    results = run_evaluation(cases,
                             lambda q: "The French capital city.",
                             lambda case, answer: (2, "vague but not wrong"))
    check("keyword check failed", results[0].passed is False)
    check("judge still scored it", results[0].judge_score == 2)


def test_run_evaluation_without_judge_leaves_scores_none():
    results = run_evaluation([TestCase("q", "e", [])], lambda q: "a")
    check("no judge -> None", results[0].judge_score is None)


# --- dataset + report ------------------------------------------------------

def test_load_dataset_roundtrip():
    data = [{"question": "q1", "expected": "e1", "keywords": ["k"]},
            {"question": "q2", "expected": "e2"}]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "d.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        cases = load_dataset(path)
    check("both cases loaded", len(cases) == 2)
    check("keywords parsed", cases[0].keywords == ["k"])
    check("missing keywords defaults empty", cases[1].keywords == [])


def test_bundled_dataset_is_valid():
    cases = load_dataset(DEFAULT_DATASET)
    check("bundled dataset loads", len(cases) == 10)
    check("every case has a question", all(c.question for c in cases))
    check("every case has an expected answer", all(c.expected for c in cases))
    check("every case has keywords", all(c.keywords for c in cases))


def test_report_contains_the_numbers():
    results = [make_result(1.0, judge=5, question="good one"),
               make_result(0.0, judge=1, question="bad one", answer="wrong")]
    report = format_report(results)
    check("report has pass rate", "50%" in report)
    check("report has avg judge", "3.00" in report)
    check("report shows worst case first", report.index("bad one") < report.index("good one"))
    check("report counts failures", "1 case(s) failed" in report)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()

    print()
    if failures:
        print(f"{len(failures)} FAILURE(S):", failures)
        sys.exit(1)
    print("All tests passed.")
