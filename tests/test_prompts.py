"""Tests for the grounded prompt text and TOP_K config — no heavy imports."""

import importlib

import config
from prompts import (
    GROUNDED_QA_PROMPT,
    GROUNDED_REFINE_PROMPT,
    NOT_COVERED,
    is_not_covered,
    normalize_citations,
)


def test_qa_prompt_answers_only_from_context():
    assert "ONLY the context passages" in GROUNDED_QA_PROMPT
    assert "Do not use outside knowledge" in GROUNDED_QA_PROMPT


def test_qa_prompt_requires_page_citations():
    assert "[<file_name> p.<page>]" in GROUNDED_QA_PROMPT
    assert "page_label" in GROUNDED_QA_PROMPT


def test_qa_prompt_contains_fixed_refusal_sentence():
    assert NOT_COVERED in GROUNDED_QA_PROMPT
    # The placeholder must have been baked in, not left for llama-index.
    assert "{not_covered}" not in GROUNDED_QA_PROMPT


def test_qa_prompt_exposes_llamaindex_variables():
    assert "{context_str}" in GROUNDED_QA_PROMPT
    assert "{query_str}" in GROUNDED_QA_PROMPT


def test_refine_prompt_is_grounded_and_cites():
    assert "Do not use outside knowledge" in GROUNDED_REFINE_PROMPT
    assert "[<file_name> p.<page>]" in GROUNDED_REFINE_PROMPT
    assert NOT_COVERED in GROUNDED_REFINE_PROMPT
    for var in ("{query_str}", "{existing_answer}", "{context_msg}"):
        assert var in GROUNDED_REFINE_PROMPT


def test_is_not_covered_matches_fixed_sentence_only():
    assert is_not_covered(NOT_COVERED)
    assert is_not_covered(f"  {NOT_COVERED}\n")
    assert is_not_covered("the notes don't cover this")
    assert not is_not_covered("The notes cover this on page 3.")
    assert not is_not_covered("")
    assert not is_not_covered(None)


def test_top_k_default_and_override(monkeypatch):
    assert config.TOP_K == 3
    monkeypatch.setenv("TOP_K", "5")
    reloaded = importlib.reload(config)
    try:
        assert reloaded.TOP_K == 5
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_normalize_citations_maps_fullwidth_brackets():
    # Seen from gpt-oss-120b in evals/run_eval.py output.
    assert (
        normalize_citations("A 2 by 4 matrix【sample_notes.pdf p.2】")
        == "A 2 by 4 matrix[sample_notes.pdf p.2]"
    )
    assert normalize_citations("［notes.pdf p.7］") == "[notes.pdf p.7]"
    assert normalize_citations("plain [notes.pdf p.1]") == (
        "plain [notes.pdf p.1]"
    )
    assert normalize_citations(None) == ""
