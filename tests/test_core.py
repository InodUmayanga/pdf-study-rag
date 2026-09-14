"""Unit tests for pure helpers — no streamlit/llama-index needed."""

import importlib
import os
from types import SimpleNamespace

import config
from utils import format_sources


def _node(file_name="notes.pdf", page="3", score=0.85):
    inner = SimpleNamespace(
        metadata={"file_name": file_name, "page_label": page}
    )
    return SimpleNamespace(node=inner, score=score)


def test_format_sources_empty():
    assert format_sources([]) == ""
    assert format_sources(None) == ""


def test_format_sources_renders_citation():
    out = format_sources([_node()])
    assert "Sources:" in out
    assert "**notes.pdf**" in out
    assert "Page 3" in out
    assert "0.85" in out


def test_format_sources_missing_metadata():
    node = SimpleNamespace(node=SimpleNamespace(metadata=None), score=None)
    out = format_sources([node])
    assert "Unknown" in out
    assert "Page ?" in out
    assert "0.00" in out


def test_config_defaults():
    assert config.DB_DIR == "./chroma_db"
    assert config.COLLECTION_NAME == "pdf_study_collection"
    assert config.EMBED_MODEL == "BAAI/bge-small-en-v1.5"
    assert config.LLM_MODEL == "openai/gpt-oss-120b"
    assert config.RENDER_SCALE == 2


def test_config_env_override(monkeypatch):
    monkeypatch.setenv("RENDER_SCALE", "3")
    monkeypatch.setenv("DB_DIR", "./other_db")
    reloaded = importlib.reload(config)
    try:
        assert reloaded.RENDER_SCALE == 3
        assert reloaded.DB_DIR == "./other_db"
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_env_is_gitignored():
    gitignore = os.path.join(os.path.dirname(__file__), "..", ".gitignore")
    with open(gitignore) as f:
        assert ".env" in f.read()
