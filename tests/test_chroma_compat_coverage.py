"""
Coverage-targeted tests for python/tqdb/chroma_compat.py.

Targets the following previously-uncovered lines:

  Line 65       – _sanitize_metadata: non-scalar value coercion (``str(v)``)
  Lines 73, 75  – _apply_filter: ``$and`` / ``$or`` logical operators
  Line 81       – _apply_filter: ``$eq`` returns False (val != rhs)
  Line 83       – _apply_filter: ``$ne`` returns False (val == rhs)
  Line 85       – _apply_filter: ``$gt`` returns False
  Line 87       – _apply_filter: ``$gte`` returns False
  Line 89       – _apply_filter: ``$lt`` returns False
  Line 91       – _apply_filter: ``$lte`` returns False
  Line 93       – _apply_filter: ``$in`` returns False (val not in rhs)
  Line 95       – _apply_filter: ``$nin`` returns False (val in rhs)
  Lines 97-101  – _apply_filter: ``$exists`` operator (rhs=True and rhs=False)
  Lines 103-108 – _apply_filter: ``$contains`` operator + bare equality
  Lines 142-147 – CompatCollection.__init__: reading _dim from manifest.json
  Lines 157-159 – CompatCollection._open_db: reopen using manifest-loaded dim
  Line 160      – CompatCollection._open_db: RuntimeError when dim unknown
  Line 167      – CompatCollection._ensure_dim: raises on empty embeddings
  Line 172      – CompatCollection._ensure_dim: raises on dimension mismatch
  Line 183      – CompatCollection._embed: ndarray result converted to list
  Line 230      – CompatCollection.upsert: documents path via embedding_function
  Lines 282-283 – CompatCollection.delete: ids + where intersection logic
  Line 296      – CompatCollection.get: early-return empty when no manifest.json
  Lines 355-372 – CompatCollection.modify: all branches
  Line 377      – CompatCollection._list_all_records: returns [] when db is empty
  Line 400      – CompatClient._load_meta: returns None for nonexistent collection
  Line 451      – CompatClient.delete_collection: raises for nonexistent name

Lines 37-38 (ImportError fallback) are intentionally skipped — they cannot be
triggered without removing the compiled extension from the venv.
"""

from __future__ import annotations

import gc
import json
import os

import numpy as np
import pytest

from tqdb.chroma_compat import (
    CompatClient,
    CompatCollection,
    PersistentClient,
    _apply_filter,
    _sanitize_metadata,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

DIM = 8


def rand_vecs(n: int, d: int = DIM) -> list:
    rng = np.random.default_rng(7)
    return rng.random((n, d)).tolist()


def make_records(*metadata_dicts):
    """Build synthetic record dicts accepted by _apply_filter."""
    return [{"id": str(i), "metadata": m} for i, m in enumerate(metadata_dicts)]


# ===========================================================================
# _sanitize_metadata — line 65
# ===========================================================================

class TestSanitizeMetadata:
    """Unit tests for the module-level _sanitize_metadata helper."""

    def test_scalar_values_pass_through_unchanged(self):
        """str/int/float/bool values are returned as-is (no coercion needed)."""
        m = {"name": "test", "count": 5, "score": 3.14, "active": True}
        assert _sanitize_metadata(m) == m

    def test_list_value_coerced_to_str(self):
        """Line 65: a list value is not a scalar → coerced with str()."""
        result = _sanitize_metadata({"tags": ["a", "b"]})
        assert result["tags"] == "['a', 'b']"

    def test_dict_value_coerced_to_str(self):
        """Line 65: a nested dict is coerced to its str() representation."""
        result = _sanitize_metadata({"nested": {"key": "val"}})
        assert result["nested"] == "{'key': 'val'}"

    def test_none_value_coerced_to_str(self):
        """Line 65: None is not str/int/float/bool → coerced to 'None'."""
        result = _sanitize_metadata({"v": None})
        assert result["v"] == "None"

    def test_mixed_scalars_and_nonscalars(self):
        """Line 65: mix of scalar and non-scalar values in one dict."""
        result = _sanitize_metadata({"name": "ok", "tags": [1, 2, 3]})
        assert result["name"] == "ok"
        assert result["tags"] == "[1, 2, 3]"

    def test_empty_dict_returns_empty_dict(self):
        assert _sanitize_metadata({}) == {}


# ===========================================================================
# _apply_filter — lines 73, 75, 81, 83, 85, 87, 89, 91, 93, 95, 97-108
# ===========================================================================

class TestApplyFilterAnd:
    """$and operator (line 73)."""

    def test_and_all_conditions_match(self):
        """Line 73: $and returns True when every sub-expression matches."""
        records = make_records({"x": 10, "tag": "ml"})
        result = _apply_filter(records, {"$and": [{"x": {"$gt": 5}}, {"tag": "ml"}]})
        assert len(result) == 1

    def test_and_one_condition_fails(self):
        """Line 73: $and returns False when any sub-expression fails."""
        records = make_records({"x": 3, "tag": "ml"})
        result = _apply_filter(records, {"$and": [{"x": {"$gt": 5}}, {"tag": "ml"}]})
        assert result == []

    def test_and_filters_mixed_records(self):
        """Line 73: $and passes only records satisfying every clause."""
        records = make_records(
            {"score": 10, "group": "A"},  # passes
            {"score": 3, "group": "A"},   # fails score
            {"score": 10, "group": "B"},  # fails group
        )
        result = _apply_filter(records, {"$and": [{"score": {"$gte": 10}}, {"group": "A"}]})
        assert len(result) == 1
        assert result[0]["id"] == "0"


class TestApplyFilterOr:
    """$or operator (line 75)."""

    def test_or_both_branches_match(self):
        """Line 75: $or keeps record when all sub-expressions match."""
        records = make_records({"x": 5})
        result = _apply_filter(records, {"$or": [{"x": {"$eq": 5}}, {"x": {"$eq": 99}}]})
        assert len(result) == 1

    def test_or_one_branch_matches(self):
        """Line 75: $or keeps record when at least one sub-expression matches."""
        records = make_records({"x": 3}, {"x": 99})
        result = _apply_filter(records, {"$or": [{"x": {"$eq": 3}}, {"x": {"$eq": 5}}]})
        assert len(result) == 1
        assert result[0]["id"] == "0"

    def test_or_no_branch_matches(self):
        """Line 75: $or excludes record when no sub-expression matches."""
        records = make_records({"x": 7})
        result = _apply_filter(records, {"$or": [{"x": {"$eq": 3}}, {"x": {"$eq": 5}}]})
        assert result == []


class TestApplyFilterEq:
    """$eq operator (line 81: return False path)."""

    def test_eq_excludes_non_matching_value(self):
        """Line 81: val != rhs triggers ``return False``."""
        records = make_records({"x": 5}, {"x": 3})
        result = _apply_filter(records, {"x": {"$eq": 5}})
        assert len(result) == 1
        assert result[0]["metadata"]["x"] == 5

    def test_eq_all_fail(self):
        """Line 81: every record excluded when no val matches rhs."""
        records = make_records({"x": 1}, {"x": 2})
        result = _apply_filter(records, {"x": {"$eq": 99}})
        assert result == []


class TestApplyFilterNe:
    """$ne operator (line 83: return False path)."""

    def test_ne_excludes_equal_value(self):
        """Line 83: val == rhs triggers ``return False``."""
        records = make_records({"x": 5}, {"x": 3})
        result = _apply_filter(records, {"x": {"$ne": 5}})
        assert len(result) == 1
        assert result[0]["metadata"]["x"] == 3

    def test_ne_all_fail(self):
        """Line 83: all records excluded when every val equals rhs."""
        records = make_records({"x": 5})
        result = _apply_filter(records, {"x": {"$ne": 5}})
        assert result == []


class TestApplyFilterGt:
    """$gt operator (line 85: return False path)."""

    def test_gt_keeps_strictly_greater(self):
        """Line 85: keeps only records with val > rhs."""
        records = make_records({"score": 10}, {"score": 3}, {"score": 5})
        result = _apply_filter(records, {"score": {"$gt": 5}})
        assert len(result) == 1
        assert result[0]["metadata"]["score"] == 10

    def test_gt_excludes_equal_value(self):
        """Line 85: val == rhs triggers ``return False`` (not strictly greater)."""
        records = make_records({"score": 5})
        result = _apply_filter(records, {"score": {"$gt": 5}})
        assert result == []

    def test_gt_excludes_none_value(self):
        """Line 85: missing field (val is None) triggers ``return False``."""
        records = make_records({})
        result = _apply_filter(records, {"score": {"$gt": 0}})
        assert result == []


class TestApplyFilterGte:
    """$gte operator (line 87: return False path)."""

    def test_gte_keeps_equal_and_greater(self):
        """Line 87: keeps records with val >= rhs."""
        records = make_records({"score": 5}, {"score": 6}, {"score": 4})
        result = _apply_filter(records, {"score": {"$gte": 5}})
        assert len(result) == 2

    def test_gte_excludes_lesser(self):
        """Line 87: val < rhs triggers ``return False``."""
        records = make_records({"score": 4})
        result = _apply_filter(records, {"score": {"$gte": 5}})
        assert result == []

    def test_gte_excludes_none(self):
        """Line 87: missing field triggers ``return False``."""
        records = make_records({})
        result = _apply_filter(records, {"score": {"$gte": 0}})
        assert result == []


class TestApplyFilterLt:
    """$lt operator (line 89: return False path)."""

    def test_lt_keeps_strictly_lesser(self):
        """Line 89: keeps only records with val < rhs."""
        records = make_records({"score": 3}, {"score": 10})
        result = _apply_filter(records, {"score": {"$lt": 5}})
        assert len(result) == 1
        assert result[0]["metadata"]["score"] == 3

    def test_lt_excludes_equal(self):
        """Line 89: val == rhs triggers ``return False``."""
        records = make_records({"score": 5})
        result = _apply_filter(records, {"score": {"$lt": 5}})
        assert result == []

    def test_lt_excludes_none(self):
        """Line 89: missing field triggers ``return False``."""
        records = make_records({})
        result = _apply_filter(records, {"score": {"$lt": 100}})
        assert result == []


class TestApplyFilterLte:
    """$lte operator (line 91: return False path)."""

    def test_lte_keeps_equal_and_lesser(self):
        """Line 91: keeps records with val <= rhs."""
        records = make_records({"score": 5}, {"score": 4}, {"score": 6})
        result = _apply_filter(records, {"score": {"$lte": 5}})
        assert len(result) == 2

    def test_lte_excludes_greater(self):
        """Line 91: val > rhs triggers ``return False``."""
        records = make_records({"score": 6})
        result = _apply_filter(records, {"score": {"$lte": 5}})
        assert result == []

    def test_lte_excludes_none(self):
        """Line 91: missing field triggers ``return False``."""
        records = make_records({})
        result = _apply_filter(records, {"score": {"$lte": 100}})
        assert result == []


class TestApplyFilterIn:
    """$in operator (line 93: return False path)."""

    def test_in_keeps_values_in_list(self):
        """Line 93: keeps records whose val is in rhs list."""
        records = make_records({"tag": "a"}, {"tag": "b"}, {"tag": "c"})
        result = _apply_filter(records, {"tag": {"$in": ["a", "c"]}})
        assert len(result) == 2
        tags = {r["metadata"]["tag"] for r in result}
        assert tags == {"a", "c"}

    def test_in_excludes_value_not_in_list(self):
        """Line 93: val not in rhs triggers ``return False``."""
        records = make_records({"tag": "d"}, {"tag": "e"})
        result = _apply_filter(records, {"tag": {"$in": ["a", "b"]}})
        assert result == []


class TestApplyFilterNin:
    """$nin operator (line 95: return False path)."""

    def test_nin_keeps_values_not_excluded(self):
        """Line 95: keeps records whose val is NOT in the exclusion list."""
        records = make_records({"tag": "a"}, {"tag": "b"}, {"tag": "c"})
        result = _apply_filter(records, {"tag": {"$nin": ["b", "c"]}})
        assert len(result) == 1
        assert result[0]["metadata"]["tag"] == "a"

    def test_nin_excludes_all_matching(self):
        """Line 95: val in rhs triggers ``return False``."""
        records = make_records({"tag": "a"}, {"tag": "b"})
        result = _apply_filter(records, {"tag": {"$nin": ["a", "b"]}})
        assert result == []


class TestApplyFilterExists:
    """$exists operator (lines 97-101)."""

    def test_exists_true_keeps_present_field(self):
        """Lines 97-98: $exists:True keeps records where field IS present."""
        records = make_records({"field": "value"}, {})
        result = _apply_filter(records, {"field": {"$exists": True}})
        assert len(result) == 1
        assert "field" in result[0]["metadata"]

    def test_exists_true_excludes_absent_field(self):
        """Lines 98-99: $exists:True excludes records where field is absent."""
        records = make_records({})
        result = _apply_filter(records, {"field": {"$exists": True}})
        assert result == []

    def test_exists_false_keeps_absent_field(self):
        """Lines 97,100: $exists:False keeps records where field is absent."""
        records = make_records({"other": "x"}, {"field": "val"})
        result = _apply_filter(records, {"field": {"$exists": False}})
        assert len(result) == 1
        assert "field" not in result[0]["metadata"]

    def test_exists_false_excludes_present_field(self):
        """Lines 100-101: $exists:False excludes records where field is present."""
        records = make_records({"field": "value"})
        result = _apply_filter(records, {"field": {"$exists": False}})
        assert result == []


class TestApplyFilterContains:
    """$contains operator (lines 103-104)."""

    def test_contains_substring_match(self):
        """Lines 102-104: $contains keeps records where val contains rhs."""
        records = make_records({"text": "hello world"}, {"text": "goodbye"})
        result = _apply_filter(records, {"text": {"$contains": "world"}})
        assert len(result) == 1
        assert result[0]["metadata"]["text"] == "hello world"

    def test_contains_no_substring_match(self):
        """Lines 103-104: rhs not in val triggers ``return False``."""
        records = make_records({"text": "hello"})
        result = _apply_filter(records, {"text": {"$contains": "world"}})
        assert result == []

    def test_contains_non_string_value(self):
        """Lines 103-104: non-string val is not a str → ``return False``."""
        records = make_records({"num": 42})
        result = _apply_filter(records, {"num": {"$contains": "4"}})
        assert result == []

    def test_contains_missing_field(self):
        """Lines 103-104: missing field → val is None → ``return False``."""
        records = make_records({})
        result = _apply_filter(records, {"text": {"$contains": "x"}})
        assert result == []


class TestApplyFilterBareEquality:
    """Bare (non-dict) condition equality (lines 107-108)."""

    def test_bare_equality_match(self):
        """Lines 107-108 (match path): bare cond value passes equality check."""
        records = make_records({"tag": "ml"}, {"tag": "ai"})
        result = _apply_filter(records, {"tag": "ml"})
        assert len(result) == 1
        assert result[0]["metadata"]["tag"] == "ml"

    def test_bare_equality_no_match(self):
        """Line 108 (return False path): val != cond triggers ``return False``."""
        records = make_records({"tag": "ai"})
        result = _apply_filter(records, {"tag": "ml"})
        assert result == []

    def test_bare_equality_missing_field(self):
        """Line 108: missing field → val is None → doesn't equal string cond."""
        records = make_records({})
        result = _apply_filter(records, {"tag": "ml"})
        assert result == []


# ===========================================================================
# CompatCollection — manifest.json loading (lines 142-147, 157-160)
# ===========================================================================

class TestCompatCollectionManifest:
    def test_init_loads_dim_from_existing_manifest(self, tmp_path):
        """Lines 142-147: a fresh CompatCollection reads _dim from manifest.json."""
        col_path = str(tmp_path / "col")
        # First instance creates the DB and the manifest.json
        col1 = CompatCollection(col_path, "col", "ip")
        col1.add(["v1"], embeddings=[[float(i) for i in range(DIM)]])
        assert col1._dim == DIM
        # Release handles before opening the path again (Windows mmap constraint)
        del col1
        gc.collect()
        gc.collect()

        # Second instance: no add() yet, but manifest.json exists → _dim loaded
        col2 = CompatCollection(col_path, "col", "ip")
        assert col2._dim == DIM, "Expected _dim loaded from manifest.json"

    def test_open_db_uses_manifest_dim_without_explicit_dim(self, tmp_path):
        """Lines 157-159: _open_db(None) succeeds when _dim was loaded from manifest."""
        col_path = str(tmp_path / "col")

        # Phase 1: create the DB and write the manifest, then release all handles.
        col1 = CompatCollection(col_path, "col", "ip")
        col1.add(["v1"], embeddings=rand_vecs(1))
        # Explicitly close the DB and release mmap handles before opening again.
        del col1
        gc.collect()
        gc.collect()

        # Phase 2: fresh CompatCollection — _db is None but _dim loaded from manifest
        col2 = CompatCollection(col_path, "col", "ip")
        assert col2._db is None
        assert col2._dim == DIM

        # get() calls _open_db(None) → uses self._dim (lines 157-159)
        result = col2.get()
        assert "v1" in result["ids"]

    def test_open_db_no_manifest_and_no_dim_raises(self, tmp_path):
        """Line 160: _open_db(None) raises RuntimeError if _dim is still None."""
        col = CompatCollection(str(tmp_path / "col"), "col", "ip")
        assert col._dim is None
        with pytest.raises(RuntimeError, match="dimension unknown"):
            col._open_db(None)


# ===========================================================================
# CompatCollection._ensure_dim — lines 167, 172
# ===========================================================================

class TestEnsureDim:
    def test_empty_embeddings_raises_value_error(self, tmp_path):
        """Line 167: _ensure_dim raises ValueError when embeddings list is empty."""
        col = CompatCollection(str(tmp_path / "col"), "col", "ip")
        with pytest.raises(ValueError, match="empty"):
            col._ensure_dim([])

    def test_dimension_mismatch_raises_value_error(self, tmp_path):
        """Line 172: _ensure_dim raises ValueError when dim ≠ self._dim."""
        col = CompatCollection(str(tmp_path / "col"), "col", "ip")
        col._dim = DIM
        wrong_dim = DIM + 1
        with pytest.raises(ValueError, match="dimension"):
            col._ensure_dim([[1.0] * wrong_dim])

    def test_sets_dim_when_previously_none(self, tmp_path):
        """Lines 169-170: _ensure_dim sets _dim from the first embeddings batch."""
        col = CompatCollection(str(tmp_path / "col"), "col", "ip")
        assert col._dim is None
        returned = col._ensure_dim([[1.0] * DIM])
        assert col._dim == DIM
        assert returned == DIM

    def test_passes_when_dim_matches(self, tmp_path):
        """_ensure_dim does not raise when dim matches self._dim."""
        col = CompatCollection(str(tmp_path / "col"), "col", "ip")
        col._dim = DIM
        assert col._ensure_dim([[1.0] * DIM]) == DIM


# ===========================================================================
# CompatCollection._embed — line 183 (numpy-array path)
# ===========================================================================

class TestEmbed:
    def test_embed_passes_through_provided_embeddings(self):
        """_embed returns its second argument unchanged when embeddings is not None."""
        col = CompatCollection("/nonexistent", "col", "ip")
        embs = [[1.0, 2.0]]
        assert col._embed(["text"], embs) is embs

    def test_embed_ndarray_result_converted_to_list(self, tmp_path):
        """Line 183: _embed converts an ndarray from embedding_function to list."""
        def embed_fn(texts):
            # Returns ndarray, not a list
            return np.ones((len(texts), DIM), dtype=np.float32)

        col = CompatCollection(str(tmp_path / "col"), "col", "ip", embedding_function=embed_fn)
        result = col._embed(["hello", "world"], None)
        assert isinstance(result, list), "Expected list, not ndarray"
        assert isinstance(result[0], list), "Expected list of lists"
        assert len(result) == 2
        assert len(result[0]) == DIM

    def test_embed_list_result_wrapped_in_list(self, tmp_path):
        """_embed wraps a non-ndarray iterable in list()."""
        def embed_fn(texts):
            return [[0.5] * DIM for _ in texts]  # already a list

        col = CompatCollection(str(tmp_path / "col"), "col", "ip", embedding_function=embed_fn)
        result = col._embed(["hi"], None)
        assert isinstance(result, list)

    def test_embed_no_function_and_no_embeddings_raises(self, tmp_path):
        """_embed raises ValueError when no embedding_function and embeddings=None."""
        col = CompatCollection(str(tmp_path / "col"), "col", "ip")
        with pytest.raises(ValueError, match="No embedding_function"):
            col._embed(["text"], None)


# ===========================================================================
# CompatCollection.upsert — line 230 (documents path via embedding_function)
# ===========================================================================

class TestUpsertWithEmbeddingFunction:
    def test_upsert_documents_calls_embed_fn(self, tmp_path):
        """Line 230: upsert() with documents but no embeddings calls _embed."""
        def embed_fn(texts):
            return np.ones((len(texts), DIM), dtype=np.float32).tolist()

        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("col", embedding_function=embed_fn)
        col.upsert(ids=["doc1", "doc2"], documents=["first doc", "second doc"])
        assert col.count() == 2

    def test_upsert_embed_fn_returning_ndarray(self, tmp_path):
        """Lines 183+230: embedding_function that returns ndarray is handled."""
        def embed_fn(texts):
            # Returns ndarray — should be converted via line 183
            return np.ones((len(texts), DIM), dtype=np.float32)

        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("col", embedding_function=embed_fn)
        col.upsert(ids=["x"], documents=["some text"])
        assert col.count() == 1

    def test_upsert_no_docs_no_embeddings_no_fn_raises(self, tmp_path):
        """upsert() with no embeddings and no documents raises ValueError."""
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("col")
        with pytest.raises(ValueError):
            col.upsert(ids=["a"])


# ===========================================================================
# CompatCollection.delete — lines 282-283 (ids + where intersection)
# ===========================================================================

class TestDeleteWithIdsAndWhere:
    def test_delete_ids_and_where_removes_intersection(self, tmp_path):
        """Lines 282-283: delete(ids=..., where=...) removes only IDs in both sets."""
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("col")
        col.add(
            ids=["a", "b", "c"],
            embeddings=rand_vecs(3),
            metadatas=[{"tag": "del"}, {"tag": "del"}, {"tag": "del"}],
        )
        # ids=["a","c"] ∩ where-matches=["a","b","c"] → deletes "a" and "c"
        col.delete(ids=["a", "c"], where={"tag": {"$eq": "del"}})
        result = col.get()
        assert "b" in result["ids"]
        assert "a" not in result["ids"]
        assert "c" not in result["ids"]

    def test_delete_ids_and_where_empty_intersection(self, tmp_path):
        """Lines 282-283: when intersection is empty, nothing is deleted."""
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("col")
        col.add(
            ids=["a", "b"],
            embeddings=rand_vecs(2),
            metadatas=[{"tag": "keep"}, {"tag": "del"}],
        )
        # ids=["a"] matches, but "a" has tag="keep" → no intersection with where result
        col.delete(ids=["a"], where={"tag": {"$eq": "del"}})
        result = col.get()
        assert sorted(result["ids"]) == ["a", "b"]

    def test_delete_ids_and_where_partial_intersection(self, tmp_path):
        """Lines 282-283: partial intersection — only matching id deleted."""
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("col")
        col.add(
            ids=["a", "b", "c"],
            embeddings=rand_vecs(3),
            metadatas=[{"tag": "del"}, {"tag": "keep"}, {"tag": "del"}],
        )
        # ids=["a","b"] ∩ where-matches=["a","c"] → only "a" deleted
        col.delete(ids=["a", "b"], where={"tag": {"$eq": "del"}})
        result = col.get()
        assert "b" in result["ids"]
        assert "c" in result["ids"]
        assert "a" not in result["ids"]


# ===========================================================================
# CompatCollection.get — line 296 (no manifest → early empty return)
# ===========================================================================

class TestGetNoManifest:
    def test_get_returns_empty_dict_when_no_manifest(self, tmp_path):
        """Line 296: get() returns empty result dict when manifest.json absent."""
        col = CompatCollection(str(tmp_path / "col"), "col", "ip")
        result = col.get()
        assert result == {"ids": [], "metadatas": [], "documents": []}

    def test_get_with_ids_returns_empty_when_no_manifest(self, tmp_path):
        """Line 296: get(ids=...) also returns empty when no manifest.json exists."""
        col = CompatCollection(str(tmp_path / "col"), "col", "ip")
        result = col.get(ids=["a", "b"])
        assert result == {"ids": [], "metadatas": [], "documents": []}

    def test_get_with_where_returns_empty_when_no_manifest(self, tmp_path):
        """Line 296: get(where=...) also returns empty when no manifest.json."""
        col = CompatCollection(str(tmp_path / "col"), "col", "ip")
        result = col.get(where={"x": {"$eq": 1}})
        assert result == {"ids": [], "metadatas": [], "documents": []}


# ===========================================================================
# CompatCollection.modify — lines 355-372
# ===========================================================================

class TestModify:
    def test_modify_name_updates_attribute(self, tmp_path):
        """Lines 355,360-362: modify(name=...) updates col._name."""
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("col")
        col.modify(name="renamed")
        assert col.name == "renamed"

    def test_modify_name_persisted_to_file(self, tmp_path):
        """Lines 360-362,371-372: modified name is written to _chroma_meta.json."""
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("col")
        col.modify(name="persisted_name")
        meta_path = os.path.join(str(tmp_path), "col", "_chroma_meta.json")
        with open(meta_path) as f:
            info = json.load(f)
        assert info["name"] == "persisted_name"

    def test_modify_reads_existing_meta_on_second_call(self, tmp_path):
        """Lines 357-359: second modify() reads the _chroma_meta.json created by first."""
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("col")
        col.modify(name="first")
        col.modify(name="second")  # reads existing file written by first call
        assert col.name == "second"

    def test_modify_metadata_same_metric_succeeds(self, tmp_path):
        """Lines 363-365,371-372: modify(metadata=...) succeeds when metric is unchanged."""
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("col", metadata={"hnsw:space": "ip"})
        col.modify(metadata={"hnsw:space": "ip", "note": "updated"})
        # No exception expected
        assert col._metric == "ip"

    def test_modify_metric_change_raises(self, tmp_path):
        """Lines 366-370: modify() raises ValueError when hnsw:space changes metric."""
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("col", metadata={"hnsw:space": "ip"})
        with pytest.raises(ValueError, match="Cannot change metric"):
            col.modify(metadata={"hnsw:space": "cosine"})

    def test_modify_no_args_when_meta_file_missing(self, tmp_path):
        """Lines 355-357,371-372: modify() with no args creates file when it's absent."""
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("col")
        meta_path = os.path.join(str(tmp_path), "col", "_chroma_meta.json")
        os.remove(meta_path)  # simulate absent meta file
        col.modify()           # should succeed and recreate the file
        assert os.path.exists(meta_path)

    def test_modify_name_and_metadata_together(self, tmp_path):
        """Lines 355-372: both name and metadata can be updated in one call."""
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("col")
        col.modify(name="updated", metadata={"hnsw:space": "ip", "v": 2})
        assert col.name == "updated"
        assert col._metric == "ip"


# ===========================================================================
# CompatCollection._list_all_records — line 377 (returns [] for empty db)
# ===========================================================================

class TestListAllRecordsEmpty:
    def test_list_all_records_empty_after_all_deleted(self, tmp_path):
        """Line 377: _list_all_records returns [] when db.list_all() is empty."""
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("col")
        col.add(ids=["x"], embeddings=rand_vecs(1))
        col.delete(ids=["x"])
        # Manifest exists (DB was created) but 0 records remain
        result = col.get()   # → _list_all_records(db) → db.list_all() == [] → line 377
        assert result["ids"] == []

    def test_count_zero_after_all_deleted(self, tmp_path):
        """Companion to above: count() also returns 0 after full deletion."""
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("col")
        col.add(ids=["a", "b"], embeddings=rand_vecs(2))
        col.delete(ids=["a", "b"])
        assert col.count() == 0


# ===========================================================================
# CompatClient._load_meta — line 400 (returns None for nonexistent)
# ===========================================================================

class TestLoadMeta:
    def test_load_meta_returns_none_for_missing_collection(self, tmp_path):
        """Line 400: _load_meta returns None when the _chroma_meta.json file is absent."""
        client = CompatClient(str(tmp_path))
        result = client._load_meta("no_such_collection")
        assert result is None

    def test_load_meta_returns_dict_for_existing_collection(self, tmp_path):
        """Positive path: _load_meta returns parsed dict when the file exists."""
        client = CompatClient(str(tmp_path))
        client.create_collection("exists")
        result = client._load_meta("exists")
        assert result is not None
        assert "name" in result
        assert result["name"] == "exists"


# ===========================================================================
# CompatClient.delete_collection — line 451 (nonexistent raises ValueError)
# ===========================================================================

class TestDeleteCollectionNonexistent:
    def test_delete_nonexistent_raises_value_error(self, tmp_path):
        """Line 451: delete_collection raises ValueError for an unknown name."""
        client = PersistentClient(str(tmp_path))
        with pytest.raises(ValueError, match="not found"):
            client.delete_collection("doesnotexist")

    def test_delete_existing_collection_ok(self, tmp_path):
        """Sanity: delete_collection works for a real collection."""
        client = PersistentClient(str(tmp_path))
        client.create_collection("to_delete")
        client.delete_collection("to_delete")
        assert "to_delete" not in client.list_collections()


# ===========================================================================
# Integration — get(ids=..., where=...) drives _apply_filter (line 303)
# with $and / $or operators
# ===========================================================================

class TestGetIdsAndWhereTriggers_apply_filter:
    """
    When both ``ids`` and ``where`` are supplied to get(), the code path at
    line 303 calls ``_apply_filter`` directly in Python (not Rust list_ids).
    These tests verify $and and $or are evaluated correctly via that path.
    """

    def test_get_ids_where_and_operator(self, tmp_path):
        """Lines 303 + 73: $and filter applied via _apply_filter in get()."""
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("col")
        col.add(
            ids=["a", "b", "c"],
            embeddings=rand_vecs(3),
            metadatas=[
                {"score": 10, "tag": "ml"},   # passes both clauses
                {"score": 3,  "tag": "ml"},   # fails score clause
                {"score": 10, "tag": "ai"},   # fails tag clause
            ],
        )
        res = col.get(
            ids=["a", "b", "c"],
            where={"$and": [{"score": {"$gt": 5}}, {"tag": "ml"}]},
        )
        assert res["ids"] == ["a"]

    def test_get_ids_where_or_operator(self, tmp_path):
        """Lines 303 + 75: $or filter applied via _apply_filter in get()."""
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("col")
        col.add(
            ids=["a", "b", "c"],
            embeddings=rand_vecs(3),
            metadatas=[
                {"tag": "ml"},
                {"tag": "ai"},
                {"tag": "other"},
            ],
        )
        res = col.get(
            ids=["a", "b", "c"],
            where={"$or": [{"tag": "ml"}, {"tag": "ai"}]},
        )
        assert sorted(res["ids"]) == ["a", "b"]

    def test_get_ids_where_eq_fail_path(self, tmp_path):
        """Line 81 via collection API: $eq excludes non-matching records."""
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("col")
        col.add(
            ids=["a", "b"],
            embeddings=rand_vecs(2),
            metadatas=[{"v": 1}, {"v": 2}],
        )
        res = col.get(ids=["a", "b"], where={"v": {"$eq": 1}})
        assert res["ids"] == ["a"]


# ===========================================================================
# Integration — non-scalar metadata coercion via add() (line 65)
# ===========================================================================

class TestSanitizeMetadataViaAdd:
    def test_add_list_metadata_coerced_to_str(self, tmp_path):
        """Line 65: list-valued metadata field is stored as its str() form."""
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("col")
        col.add(
            ids=["v1"],
            embeddings=rand_vecs(1),
            metadatas=[{"tags": ["a", "b", "c"]}],
        )
        res = col.get(ids=["v1"])
        assert res["metadatas"][0]["tags"] == "['a', 'b', 'c']"

    def test_add_none_metadata_coerced_to_str(self, tmp_path):
        """Line 65: None-valued metadata field is stored as 'None'."""
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("col")
        col.add(
            ids=["v1"],
            embeddings=rand_vecs(1),
            metadatas=[{"optional": None}],
        )
        res = col.get(ids=["v1"])
        assert res["metadatas"][0]["optional"] == "None"
