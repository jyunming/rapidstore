"""Python-boundary validation tests for the hybrid search kwarg.

The Rust side (`parse_hybrid` / `parse_hybrid_batch` in src/python/mod.rs) must
turn malformed input into clean `ValueError`s, never a panic. This file covers
the realistic mistakes a user can make from Python.
"""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

from tqdb import Database


@pytest.fixture
def db():
    """A tiny database with one vector + doc, just enough for the boundary tests
    to make `search`/`query` calls that actually touch the hybrid parser."""
    with tempfile.TemporaryDirectory() as tmp:
        d = 8
        instance = Database.open(tmp, dimension=d, bits=2)
        instance.insert("d0", np.zeros(d, dtype=np.float32), document="hello world")
        yield instance
        instance.close()


def _q(d: int = 8) -> np.ndarray:
    return np.zeros(d, dtype=np.float32)


# ── parse_hybrid (single search) ─────────────────────────────────────────────


def test_unknown_key_rejected(db):
    with pytest.raises(ValueError, match="unknown key"):
        db.search(_q(), top_k=1, hybrid={"text": "x", "bogus": 1})


def test_text_missing_rejected(db):
    with pytest.raises(ValueError, match="text"):
        db.search(_q(), top_k=1, hybrid={"weight": 0.5})


def test_text_wrong_type_rejected(db):
    # `text` must be str; passing int is rejected by PyO3 type extraction.
    with pytest.raises((ValueError, TypeError)):
        db.search(_q(), top_k=1, hybrid={"text": 123})


def test_weight_out_of_range(db):
    with pytest.raises(ValueError, match="weight"):
        db.search(_q(), top_k=1, hybrid={"text": "x", "weight": 1.5})
    with pytest.raises(ValueError, match="weight"):
        db.search(_q(), top_k=1, hybrid={"text": "x", "weight": -0.1})


def test_rrf_k_below_one_rejected(db):
    with pytest.raises(ValueError, match="rrf_k"):
        db.search(_q(), top_k=1, hybrid={"text": "x", "rrf_k": 0.5})


def test_oversample_zero_rejected(db):
    with pytest.raises(ValueError, match="oversample"):
        db.search(_q(), top_k=1, hybrid={"text": "x", "oversample": 0})


def test_oversample_negative_rejected(db):
    # PyO3's usize extraction will reject the negative; the exact error type
    # is OverflowError but ValueError is also acceptable.
    with pytest.raises((ValueError, OverflowError)):
        db.search(_q(), top_k=1, hybrid={"text": "x", "oversample": -3})


def test_empty_text_falls_back_silently(db):
    # Empty text is not an error — it collapses to dense-only.
    out = db.search(_q(), top_k=1, hybrid={"text": "", "weight": 0.5})
    assert isinstance(out, list)
    assert len(out) == 1


def test_unicode_text_accepted(db):
    # Non-ASCII / surrogate-prone characters must round-trip cleanly.
    out = db.search(_q(), top_k=1, hybrid={"text": "café résumé naïve"})
    assert isinstance(out, list)


# ── parse_hybrid_batch (query) ───────────────────────────────────────────────


def test_query_text_broadcasts(db):
    emb = np.stack([_q(), _q()], axis=0)
    out = db.query(emb, n_results=1, hybrid={"text": "hello"})
    assert isinstance(out, list)
    assert len(out) == 2


def test_query_texts_list_must_match_rows(db):
    emb = np.stack([_q(), _q()], axis=0)
    with pytest.raises(ValueError, match="texts"):
        db.query(emb, n_results=1, hybrid={"texts": ["one"]})


def test_query_text_and_texts_both_set_rejected(db):
    emb = np.stack([_q(), _q()], axis=0)
    with pytest.raises(ValueError, match="either"):
        db.query(emb, n_results=1, hybrid={"text": "a", "texts": ["a", "b"]})


def test_query_neither_text_nor_texts_rejected(db):
    emb = np.stack([_q(), _q()], axis=0)
    with pytest.raises(ValueError, match="missing"):
        db.query(emb, n_results=1, hybrid={"weight": 0.5})


def test_query_all_empty_texts_falls_back(db):
    emb = np.stack([_q(), _q()], axis=0)
    out = db.query(emb, n_results=1, hybrid={"texts": ["", ""]})
    assert isinstance(out, list)
    assert len(out) == 2
