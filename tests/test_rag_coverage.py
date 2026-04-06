"""
Coverage-gap tests for tqdb.rag (TurboQuantRetriever).

Targets lines identified as missing in the 74% coverage report:
  59-64  — insert_many fallback path and per-row insert loop
  77     — tuple-shaped result in similarity_search

Lines 6-19 (ImportError stub class) are intentionally excluded —
they require the native extension to be absent, which is impossible here.
"""

from __future__ import annotations

import numpy as np
import pytest

from tqdb.rag import TurboQuantRetriever


# ---------------------------------------------------------------------------
# Mock DB factories
# ---------------------------------------------------------------------------

class _DBInsertBatch:
    """Implements insert_batch only — existing fast path (already covered)."""

    def __init__(self):
        self.calls: list = []

    def insert_batch(self, ids, vecs, metas, docs, mode):
        self.calls.append(("insert_batch", list(ids), mode))

    def search(self, q, k):
        return []


class _DBInsertMany:
    """
    Implements insert_many but NOT insert_batch.
    Triggers lines 59-61 in add_texts().
    """

    def __init__(self):
        self.calls: list = []

    def insert_many(self, ids, vecs, metas, docs, mode):
        self.calls.append(("insert_many", list(ids), list(metas), list(docs), mode))

    def search(self, q, k):
        return []


class _DBInsertOne:
    """
    Implements only insert() (no insert_batch, no insert_many).
    Triggers the per-row loop at lines 63-64 in add_texts().
    """

    def __init__(self):
        self.calls: list = []

    def insert(self, doc_id, vec, meta, text):
        self.calls.append(("insert", doc_id, meta, text))

    def search(self, q, k):
        return []


class _DBTupleSearch:
    """
    search() returns (id, score) tuples — triggers line 77.
    insert_batch is present so add_texts() works normally.
    """

    def __init__(self, results: list = ()):
        self._results = list(results)
        self.calls: list = []

    def insert_batch(self, ids, vecs, metas, docs, mode):
        self.calls.append(("insert_batch",))

    def search(self, q, k):
        return list(self._results)


class _DBDictSearch:
    """
    search() returns dict results — the already-covered dict path (line 73-75).
    Kept here for contrast / regression.
    """

    def __init__(self, results: list = ()):
        self._results = list(results)

    def insert_batch(self, ids, vecs, metas, docs, mode):
        pass

    def search(self, q, k):
        return list(self._results)


# ---------------------------------------------------------------------------
# Retriever factory — bypasses __init__ to avoid touching the real DB
# ---------------------------------------------------------------------------

def _make_retriever(db, doc_store: dict | None = None) -> TurboQuantRetriever:
    r = TurboQuantRetriever.__new__(TurboQuantRetriever)
    r.db = db
    r.doc_store = {} if doc_store is None else doc_store
    return r


# ===========================================================================
# add_texts() — insert_many path  (lines 59-61)
# ===========================================================================

class TestAddTextsInsertManyPath:
    def test_insert_many_is_called_when_no_insert_batch(self):
        """
        Lines 59-61: when the DB has insert_many but not insert_batch,
        insert_many() is invoked and the method returns early.
        """
        mock = _DBInsertMany()
        r = _make_retriever(mock)
        r.add_texts(["hello", "world"], [[0.1] * 8, [0.2] * 8])

        assert len(mock.calls) == 1
        name, ids, metas, docs, mode = mock.calls[0]
        assert name == "insert_many"
        assert ids == ["doc_0", "doc_1"]
        assert mode == "insert"

    def test_insert_many_receives_correct_texts(self):
        """Lines 59-61: texts (docs) forwarded to insert_many are correct."""
        mock = _DBInsertMany()
        r = _make_retriever(mock)
        r.add_texts(["alpha", "beta", "gamma"], [[1.0] * 8] * 3)

        _, ids, metas, docs, mode = mock.calls[0]
        assert docs == ["alpha", "beta", "gamma"]

    def test_insert_many_receives_correct_metadatas(self):
        """Lines 59-61: custom metadatas are forwarded verbatim."""
        mock = _DBInsertMany()
        r = _make_retriever(mock)
        metas = [{"src": "a"}, {"src": "b"}]
        r.add_texts(["x", "y"], [[0.5] * 8, [0.6] * 8], metadatas=metas)

        _, ids, got_metas, docs, mode = mock.calls[0]
        assert got_metas == metas

    def test_insert_many_updates_doc_store(self):
        """Lines 59-61: doc_store is populated before the insert_many call."""
        mock = _DBInsertMany()
        r = _make_retriever(mock)
        r.add_texts(["hello"], [[1.0] * 8], metadatas=[{"tag": "test"}])

        assert "doc_0" in r.doc_store
        assert r.doc_store["doc_0"]["text"] == "hello"
        assert r.doc_store["doc_0"]["metadata"] == {"tag": "test"}

    def test_insert_many_ids_are_sequential_across_calls(self):
        """Lines 59-61: doc IDs continue from where the doc_store left off."""
        mock = _DBInsertMany()
        r = _make_retriever(mock)
        r.add_texts(["first"], [[1.0] * 8])      # doc_0
        r.add_texts(["second"], [[2.0] * 8])     # doc_1

        assert "doc_1" in r.doc_store
        _, ids2, *_ = mock.calls[1]
        assert ids2 == ["doc_1"]

    def test_insert_many_no_insert_batch_not_called(self):
        """Lines 59-61: insert_batch must NOT be called when only insert_many exists."""
        mock = _DBInsertMany()
        r = _make_retriever(mock)
        r.add_texts(["hi"], [[0.0] * 8])
        assert not any(c[0] == "insert_batch" for c in mock.calls)

    def test_insert_many_default_empty_metadatas(self):
        """Lines 59-61: metadatas defaults to list of empty dicts."""
        mock = _DBInsertMany()
        r = _make_retriever(mock)
        r.add_texts(["no meta"], [[1.0] * 8])  # no metadatas kwarg

        _, ids, metas, docs, mode = mock.calls[0]
        assert metas == [{}]


# ===========================================================================
# add_texts() — per-row insert loop  (lines 63-64)
# ===========================================================================

class TestAddTextsInsertLoopPath:
    def test_insert_called_once_per_text(self):
        """
        Lines 63-64: when the DB has neither insert_batch nor insert_many,
        db.insert() is called once per text.
        """
        mock = _DBInsertOne()
        r = _make_retriever(mock)
        r.add_texts(["a", "b", "c"], [[1.0] * 8, [2.0] * 8, [3.0] * 8])

        assert len(mock.calls) == 3

    def test_insert_doc_ids_are_correct(self):
        """Lines 63-64: each insert call gets the right sequential doc_id."""
        mock = _DBInsertOne()
        r = _make_retriever(mock)
        r.add_texts(["x", "y"], [[0.1] * 8, [0.2] * 8])

        assert mock.calls[0][1] == "doc_0"
        assert mock.calls[1][1] == "doc_1"

    def test_insert_texts_are_correct(self):
        """Lines 63-64: text argument forwarded to insert() correctly."""
        mock = _DBInsertOne()
        r = _make_retriever(mock)
        r.add_texts(["hello", "world"], [[1.0] * 8, [2.0] * 8])

        assert mock.calls[0][3] == "hello"
        assert mock.calls[1][3] == "world"

    def test_insert_metadatas_forwarded(self):
        """Lines 63-64: metadata dict for each row is correct."""
        mock = _DBInsertOne()
        r = _make_retriever(mock)
        metas = [{"src": "one"}, {"src": "two"}]
        r.add_texts(["p", "q"], [[1.0] * 8, [2.0] * 8], metadatas=metas)

        assert mock.calls[0][2] == {"src": "one"}
        assert mock.calls[1][2] == {"src": "two"}

    def test_insert_updates_doc_store(self):
        """Lines 63-64: doc_store is populated even through the loop path."""
        mock = _DBInsertOne()
        r = _make_retriever(mock)
        r.add_texts(["hello"], [[1.0] * 8])

        assert "doc_0" in r.doc_store
        assert r.doc_store["doc_0"]["text"] == "hello"

    def test_insert_loop_ids_continue_across_calls(self):
        """Lines 63-64: second batch starts at the right offset."""
        mock = _DBInsertOne()
        r = _make_retriever(mock)
        r.add_texts(["first"], [[1.0] * 8])   # → doc_0
        r.add_texts(["second"], [[2.0] * 8])  # → doc_1

        second_call_id = mock.calls[1][1]
        assert second_call_id == "doc_1"


# ===========================================================================
# similarity_search() — tuple result path  (line 77)
# ===========================================================================

class TestSimilaritySearchTuplePath:
    def _doc_store(self, n: int = 3) -> dict:
        return {
            f"doc_{i}": {"text": f"text {i}", "metadata": {"idx": i}}
            for i in range(n)
        }

    def test_tuple_result_returns_correct_texts(self):
        """Line 77: (id, score) tuple from db.search() is unpacked correctly."""
        mock = _DBTupleSearch(results=[("doc_0", 0.95), ("doc_1", 0.80)])
        r = _make_retriever(mock, doc_store=self._doc_store())
        results = r.similarity_search([1.0] * 8)

        assert len(results) == 2
        assert results[0]["text"] == "text 0"
        assert results[1]["text"] == "text 1"

    def test_tuple_result_scores_preserved(self):
        """Line 77: score from the tuple is forwarded to the output dict."""
        mock = _DBTupleSearch(results=[("doc_2", 0.42)])
        r = _make_retriever(mock, doc_store=self._doc_store())
        results = r.similarity_search([1.0] * 8)

        assert results[0]["score"] == pytest.approx(0.42)

    def test_tuple_result_metadata_preserved(self):
        """Line 77: metadata from doc_store is included in the output."""
        mock = _DBTupleSearch(results=[("doc_0", 0.9)])
        r = _make_retriever(mock, doc_store=self._doc_store())
        results = r.similarity_search([1.0] * 8)

        assert results[0]["metadata"] == {"idx": 0}

    def test_tuple_result_unknown_id_skipped(self):
        """Line 77: a tuple whose id is not in doc_store is silently skipped."""
        mock = _DBTupleSearch(results=[("doc_0", 0.9), ("__unknown__", 0.5)])
        r = _make_retriever(mock, doc_store=self._doc_store())
        results = r.similarity_search([1.0] * 8)

        # Only doc_0 is in doc_store
        assert len(results) == 1
        assert results[0]["text"] == "text 0"

    def test_tuple_result_empty_search(self):
        """Line 77 (not reached): empty tuple list returns empty output."""
        mock = _DBTupleSearch(results=[])
        r = _make_retriever(mock, doc_store=self._doc_store())
        results = r.similarity_search([1.0] * 8)
        assert results == []

    def test_tuple_result_multiple_results_ordering(self):
        """Line 77: result order follows db.search() return order."""
        mock = _DBTupleSearch(results=[
            ("doc_2", 0.99),
            ("doc_0", 0.88),
            ("doc_1", 0.77),
        ])
        r = _make_retriever(mock, doc_store=self._doc_store())
        results = r.similarity_search([1.0] * 8)

        assert [res["score"] for res in results] == pytest.approx([0.99, 0.88, 0.77])

    def test_tuple_result_vs_dict_result_same_output(self):
        """
        Sanity: tuple path (line 77) and dict path (lines 73-75) produce
        identical output for the same logical result.
        """
        doc_store = self._doc_store(1)

        mock_tuple = _DBTupleSearch(results=[("doc_0", 0.7)])
        mock_dict  = _DBDictSearch(results=[{"id": "doc_0", "score": 0.7}])

        r_tuple = _make_retriever(mock_tuple, doc_store=dict(doc_store))
        r_dict  = _make_retriever(mock_dict,  doc_store=dict(doc_store))

        out_tuple = r_tuple.similarity_search([1.0] * 8)
        out_dict  = r_dict.similarity_search([1.0] * 8)

        assert out_tuple[0]["text"]  == out_dict[0]["text"]
        assert out_tuple[0]["score"] == pytest.approx(out_dict[0]["score"])
        assert out_tuple[0]["metadata"] == out_dict[0]["metadata"]
