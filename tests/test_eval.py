"""End-to-end retrieval evaluation (pytest marker: ``eval``).

Generates the fixture PDF, runs the real OCR -> embed -> ChromaDB pipeline
on it and checks that retrieval finds the expected page for the questions
in ``evals/questions.json``. Needs the full ``requirements.txt`` installed
and downloads the embedding model on first run, so it is excluded from the
default ``pytest`` run (see ``addopts`` in pyproject.toml):

    pytest -m eval tests/test_eval.py -v -s

The pass threshold is ``EVAL_MIN_HIT3`` (default 0.7).
"""

import os

import pytest

pytestmark = pytest.mark.eval


def test_retrieval_hits_expected_pages():
    # Imported here so collecting this file never needs the heavy deps.
    from evals.run_eval import print_report, run

    results = run(with_llm=False)
    print_report(results)

    retrieval = results["retrieval"]
    assert retrieval["n"] >= 10, "question set unexpectedly small"
    empty = [r["id"] for r in retrieval["rows"] if not r["retrieved_pages"]]
    assert not empty, f"nothing retrieved for {empty}"

    min_hit3 = float(os.getenv("EVAL_MIN_HIT3", "0.7"))
    assert retrieval["hit_at_3"] >= min_hit3, (
        f"hit@3 {retrieval['hit_at_3']:.2f} is below {min_hit3:.2f}; "
        "see evals/results.json"
    )
