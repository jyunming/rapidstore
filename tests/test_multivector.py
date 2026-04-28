"""Tests for the ColBERT-style multi-vector store at ``tqdb.multivector``."""

from __future__ import annotations

import gc
import shutil
import tempfile
from contextlib import contextmanager

import numpy as np
import pytest

from tqdb.multivector import MultiVectorStore


@contextmanager
def _tempdir():
    path = tempfile.mkdtemp()
    try:
        yield path
    finally:
        gc.collect()
        shutil.rmtree(path, ignore_errors=True)


def _unit_vec(d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(d).astype(np.float32)
    return v / max(np.linalg.norm(v), 1e-9)


def _doc_tokens(d: int, n: int, seed: int) -> np.ndarray:
    """Produce a small bag of token vectors for a fake doc."""
    return np.stack([_unit_vec(d, seed + i) for i in range(n)])


# ── core: insert + search round-trip ────────────────────────────────────


def test_insert_and_search_returns_inserted_doc():
    d = 16
    with _tempdir() as path:
        store = MultiVectorStore.open(path, dimension=d, bits=4, metric="cosine")
        try:
            doc_a_tokens = _doc_tokens(d, 4, 100)
            doc_b_tokens = _doc_tokens(d, 4, 200)
            store.insert("a", doc_a_tokens, document="doc a text")
            store.insert("b", doc_b_tokens, document="doc b text")
            assert len(store) == 2

            # Query with one of doc-a's tokens — MaxSim should put a at top.
            qv = doc_a_tokens[:2]  # 2 query tokens
            hits = store.search(qv, top_k=2)
            assert hits, "expected non-empty hits"
            assert hits[0]["doc_id"] == "a"
            assert hits[0]["document"] == "doc a text"
        finally:
            store._db.close()


def test_maxsim_prefers_doc_with_more_matching_tokens():
    """Doc whose tokens overlap with multiple query tokens should outrank
    a doc that only matches one query token."""
    d = 16
    with _tempdir() as path:
        store = MultiVectorStore.open(path, dimension=d, bits=4, metric="cosine")
        try:
            t0, t1, t2 = _unit_vec(d, 1), _unit_vec(d, 2), _unit_vec(d, 3)
            t_other = _unit_vec(d, 999)
            # `multi` has all three query tokens; `partial` has only t0 plus noise.
            store.insert("multi", np.stack([t0, t1, t2, t_other]), document="multi")
            store.insert("partial", np.stack([t0, t_other, t_other, t_other]), document="partial")

            hits = store.search(np.stack([t0, t1, t2]), top_k=2)
            assert hits[0]["doc_id"] == "multi", f"got {hits!r}"
            # multi's score should be ~3 (each query token finds a perfect hit);
            # partial's score should be ~1 + 2 * (cosine to noise) ≪ 3.
            assert hits[0]["score"] > hits[1]["score"]
        finally:
            store._db.close()


# ── replace + delete ────────────────────────────────────────────────────


def test_insert_replaces_existing_doc():
    d = 8
    with _tempdir() as path:
        store = MultiVectorStore.open(path, dimension=d, bits=2, metric="cosine")
        try:
            store.insert("x", _doc_tokens(d, 3, 1), document="v1")
            assert store.get("x")["document"] == "v1"
            store.insert("x", _doc_tokens(d, 5, 2), document="v2")
            info = store.get("x")
            assert info["document"] == "v2"
            assert info["n_tokens"] == 5
            assert len(store) == 1
        finally:
            store._db.close()


def test_delete_removes_doc():
    d = 8
    with _tempdir() as path:
        store = MultiVectorStore.open(path, dimension=d, bits=2, metric="cosine")
        try:
            store.insert("a", _doc_tokens(d, 3, 1))
            store.insert("b", _doc_tokens(d, 3, 2))
            assert store.delete("a") is True
            assert "a" not in store
            assert "b" in store
            # Deleting again is a no-op.
            assert store.delete("a") is False
            # Search must not surface "a".
            hits = store.search(_doc_tokens(d, 3, 1), top_k=10)
            assert all(h["doc_id"] != "a" for h in hits)
        finally:
            store._db.close()


# ── persistence round-trip ──────────────────────────────────────────────


def test_persistence_round_trip():
    d = 16
    with _tempdir() as path:
        # Phase 1: insert and close.
        store = MultiVectorStore.open(path, dimension=d, bits=4, metric="cosine")
        store.insert("a", _doc_tokens(d, 4, 7), document="alpha")
        store.insert("b", _doc_tokens(d, 4, 8), document="bravo")
        store._db.close()

        # Phase 2: reopen and verify retrievable.
        store2 = MultiVectorStore.open(path, dimension=d, bits=4, metric="cosine")
        try:
            assert len(store2) == 2
            assert sorted(store2.doc_ids()) == ["a", "b"]
            hits = store2.search(_doc_tokens(d, 4, 7)[:2], top_k=1)
            assert hits[0]["doc_id"] == "a"
            assert hits[0]["document"] == "alpha"
        finally:
            store2._db.close()


# ── edge cases ──────────────────────────────────────────────────────────


def test_empty_store_returns_empty():
    d = 8
    with _tempdir() as path:
        store = MultiVectorStore.open(path, dimension=d, bits=2, metric="cosine")
        try:
            hits = store.search(_doc_tokens(d, 2, 0), top_k=5)
            assert hits == []
        finally:
            store._db.close()


def test_zero_top_k_returns_empty():
    d = 8
    with _tempdir() as path:
        store = MultiVectorStore.open(path, dimension=d, bits=2, metric="cosine")
        try:
            store.insert("a", _doc_tokens(d, 3, 1))
            assert store.search(_doc_tokens(d, 2, 1), top_k=0) == []
        finally:
            store._db.close()


def test_invalid_input_shape_raises():
    d = 8
    with _tempdir() as path:
        store = MultiVectorStore.open(path, dimension=d, bits=2, metric="cosine")
        try:
            with pytest.raises(ValueError, match="2-D"):
                store.insert("a", np.zeros(d))  # 1-D, not 2-D
            with pytest.raises(ValueError, match="at least one"):
                store.insert("b", np.zeros((0, d)))
            store.insert("c", _doc_tokens(d, 1, 1))
            with pytest.raises(ValueError, match="2-D"):
                store.search(np.zeros(d), top_k=1)
        finally:
            store._db.close()


def test_metadata_preserved_and_filterable():
    d = 16
    with _tempdir() as path:
        store = MultiVectorStore.open(path, dimension=d, bits=4, metric="cosine")
        try:
            store.insert("a", _doc_tokens(d, 3, 1), document="A", metadata={"lang": "en"})
            store.insert("b", _doc_tokens(d, 3, 2), document="B", metadata={"lang": "fr"})

            # The store re-exposes user metadata cleanly (no internal keys leaking).
            info = store.get("a")
            assert info["metadata"] == {"lang": "en"}

            # candidate_filter restricts to just the English doc.
            hits = store.search(
                _doc_tokens(d, 3, 1),
                top_k=10,
                candidate_filter={"lang": "en"},
            )
            assert all(h["doc_id"] == "a" for h in hits)
        finally:
            store._db.close()


def test_insert_many_smoke():
    d = 8
    with _tempdir() as path:
        store = MultiVectorStore.open(path, dimension=d, bits=2, metric="cosine")
        try:
            store.insert_many([
                ("a", _doc_tokens(d, 2, 1), "A", None),
                ("b", _doc_tokens(d, 2, 2), "B", None),
                ("c", _doc_tokens(d, 2, 3), "C", None),
            ])
            assert len(store) == 3
        finally:
            store._db.close()
