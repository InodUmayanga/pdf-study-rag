"""End-to-end retrieval evaluation (pytest marker: ``eval``).

Generates the fixture PDF, runs the real OCR -> embed -> ChromaDB pipeline
on it and checks that retrieval finds the expected page for the questions
in ``evals/questions.json``. Needs the full ``requirements.txt`` installed
and downloads the embedding model on first run, so it is excluded from the
default ``pytest`` run (see ``addopts`` in pyproject.toml):

    pytest -m eval tests/test_eval.py -v -s

Gates (override with environment variables):
    EVAL_MIN_HIT3  minimum hit@3, default 0.9
    EVAL_MIN_HIT1  minimum hit@1, default 0.75

The LLM answer checks are deliberately not part of the gate: they cost API
calls and vary from run to run. ``EVAL_WITH_LLM=1`` runs and prints them
anyway (needs GROQ_API_KEY). Results go to evals/results-pytest.json so a
``python -m evals.run_eval`` report in evals/results.json is not overwritten.
"""

import os

import pytest

pytestmark = pytest.mark.eval

RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "evals",
    "results-pytest.json",
)


def test_retrieval_hits_expected_pages():
    # Imported here so collecting this file never needs the heavy deps.
    from evals.run_eval import print_report, run

    with_llm = os.getenv("EVAL_WITH_LLM") == "1"
    results = run(with_llm=with_llm, results_path=RESULTS_PATH)
    print_report(results)

    retrieval = results["retrieval"]
    assert retrieval["n"] >= 10, "question set unexpectedly small"
    empty = [r["id"] for r in retrieval["rows"] if not r["retrieved_pages"]]
    assert not empty, f"nothing retrieved for {empty}"

    min_hit3 = float(os.getenv("EVAL_MIN_HIT3", "0.9"))
    min_hit1 = float(os.getenv("EVAL_MIN_HIT1", "0.75"))
    assert retrieval["hit_at_3"] >= min_hit3, (
        f"hit@3 {retrieval['hit_at_3']:.2f} is below {min_hit3:.2f}; "
        f"see {RESULTS_PATH}"
    )
    assert retrieval["hit_at_1"] >= min_hit1, (
        f"hit@1 {retrieval['hit_at_1']:.2f} is below {min_hit1:.2f}; "
        f"see {RESULTS_PATH}"
    )
