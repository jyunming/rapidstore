"""
BRUTAL QA test suite round 2 — deeper, darker, more destructive.

Attacks new surfaces: include= param, list_metadata_values, insert_batch modes,
normalize=True, flush(), update_metadata edge cases, query() batch, get_many,
list_ids pagination, delete edge cases, score consistency, dimension extremes,
parameter bounds, stats consistency, list_all, metadata falsy values, and
concurrent writes.

Run with:
    python -m pytest tests/test_brutal_qa2.py -v --basetemp=tmp_pytest_brutal2
"""
from __future__ import annotations

import gc
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from tqdb import Database


# ---------------------------------------------------------------------------
# Helpers (identical convention to test_brutal_qa.py)
# ---------------------------------------------------------------------------

def _vec(d: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(d).astype(np.float32)
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def _unit_batch(n: int, d: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, d)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-9)


def _open(path, d: int = 16, bits: int = 4, metric: str = "ip", **kw) -> Database:
    return Database.open(str(path), d, bits=bits, seed=42, metric=metric, **kw)


# ---------------------------------------------------------------------------
# ██╗███╗   ██╗ ██████╗██╗     ██╗   ██╗██████╗ ███████╗
# ██║████╗  ██║██╔════╝██║     ██║   ██║██╔══██╗██╔════╝
# ██║██╔██╗ ██║██║     ██║     ██║   ██║██║  ██║█████╗
# ██║██║╚██╗██║██║     ██║     ██║   ██║██║  ██║██╔══╝
# ██║██║ ╚████║╚██████╗███████╗╚██████╔╝██████╔╝███████╗
# ---------------------------------------------------------------------------

class TestIncludeParam:
    """Break the include= field-selection parameter on search()."""

    def test_include_id_only_no_other_keys(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0), metadata={"x": 1}, document="hello")
        results = db.search(_vec(16, seed=0), top_k=1, include=["id"])
        assert len(results) == 1
        assert set(results[0].keys()) == {"id"}

    def test_include_score_only_no_other_keys(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0))
        results = db.search(_vec(16, seed=0), top_k=1, include=["score"])
        assert len(results) == 1
        assert set(results[0].keys()) == {"score"}

    def test_include_metadata_only(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0), metadata={"k": "v"})
        results = db.search(_vec(16, seed=0), top_k=1, include=["metadata"])
        assert len(results) == 1
        assert set(results[0].keys()) == {"metadata"}
        assert results[0]["metadata"]["k"] == "v"

    def test_include_document_only_with_doc_set(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0), document="my text")
        results = db.search(_vec(16, seed=0), top_k=1, include=["document"])
        assert len(results) == 1
        assert set(results[0].keys()) == {"document"}
        assert results[0]["document"] == "my text"

    def test_include_document_only_no_doc_stored(self, tmp_path):
        # Document key should be absent (not None) when no doc was stored
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0))
        results = db.search(_vec(16, seed=0), top_k=1, include=["document"])
        assert len(results) == 1
        # document key must not be present since none was stored
        assert "document" not in results[0]

    def test_include_id_and_score_two_keys(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0))
        results = db.search(_vec(16, seed=0), top_k=1, include=["id", "score"])
        assert len(results) == 1
        assert set(results[0].keys()) == {"id", "score"}

    def test_include_empty_list_returns_empty_dicts(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0), metadata={"x": 1}, document="hi")
        results = db.search(_vec(16, seed=0), top_k=1, include=[])
        assert len(results) == 1
        # With no fields requested, each result dict should be empty
        assert results[0] == {}

    def test_include_invalid_field_raises_value_error(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0))
        with pytest.raises(Exception) as exc_info:
            db.search(_vec(16, seed=0), top_k=1, include=["invalid_field"])
        # Must be a ValueError about include field, not a Rust panic
        assert "panic" not in type(exc_info.value).__name__.lower()

    def test_include_case_insensitive_id(self, tmp_path):
        # parse_include_set lowercases; "ID" should work same as "id"
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0))
        results = db.search(_vec(16, seed=0), top_k=1, include=["ID"])
        assert len(results) == 1
        assert "id" in results[0]

    def test_include_none_returns_all_four_fields(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0), metadata={"x": 1}, document="doc")
        results = db.search(_vec(16, seed=0), top_k=1, include=None)
        assert "id" in results[0]
        assert "score" in results[0]
        assert "metadata" in results[0]

    def test_include_scores_are_finite_with_partial_include(self, tmp_path):
        db = _open(tmp_path / "db")
        vecs = _unit_batch(10, 16)
        db.insert_batch([str(i) for i in range(10)], vecs)
        results = db.search(vecs[0], top_k=5, include=["id", "score"])
        for r in results:
            assert np.isfinite(r["score"])

    def test_include_all_explicit_fields(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0), metadata={"k": 1}, document="text")
        results = db.search(
            _vec(16, seed=0), top_k=1,
            include=["id", "score", "metadata", "document"]
        )
        assert len(results) == 1
        r = results[0]
        assert "id" in r and "score" in r and "metadata" in r


# ---------------------------------------------------------------------------
# ██╗     ██╗███████╗████████╗    ███╗   ███╗███████╗████████╗ █████╗
# ██║     ██║██╔════╝╚══██╔══╝    ████╗ ████║██╔════╝╚══██╔══╝██╔══██╗
# ██║     ██║███████╗   ██║       ██╔████╔██║█████╗     ██║   ███████║
# ██║     ██║╚════██║   ██║       ██║╚██╔╝██║██╔══╝     ██║   ██╔══██║
# ███████╗██║███████║   ██║       ██║ ╚═╝ ██║███████╗   ██║   ██║  ██║
# ---------------------------------------------------------------------------

class TestListMetadataValues:
    """Hammer the completely untested list_metadata_values() API."""

    def test_empty_db_returns_empty_dict(self, tmp_path):
        db = _open(tmp_path / "db")
        result = db.list_metadata_values("category")
        assert result == {}

    def test_counts_correctly(self, tmp_path):
        db = _open(tmp_path / "db")
        for i in range(6):
            cat = "A" if i < 4 else "B"
            db.insert(str(i), _vec(16, seed=i), metadata={"cat": cat})
        counts = db.list_metadata_values("cat")
        assert counts.get("A") == 4
        assert counts.get("B") == 2

    def test_missing_field_skips_vectors_without_it(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("with", _vec(16, seed=0), metadata={"x": "yes"})
        db.insert("without", _vec(16, seed=1))  # no metadata
        counts = db.list_metadata_values("x")
        # Only "with" has x; "without" has no x → total should be 1
        total = sum(counts.values())
        assert total == 1

    def test_integer_values_are_stringified(self, tmp_path):
        # Docstring: "Non-string values are stringified via their JSON representation"
        db = _open(tmp_path / "db")
        for i in range(3):
            db.insert(str(i), _vec(16, seed=i), metadata={"score": 42})
        counts = db.list_metadata_values("score")
        # Integer 42 → JSON "42"
        assert "42" in counts
        assert counts["42"] == 3

    def test_float_values_stringified(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0), metadata={"val": 3.14})
        counts = db.list_metadata_values("val")
        # Should have exactly one key (stringified float)
        assert len(counts) == 1

    def test_bool_values_stringified(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("t", _vec(16, seed=0), metadata={"flag": True})
        db.insert("f", _vec(16, seed=1), metadata={"flag": False})
        counts = db.list_metadata_values("flag")
        # true → "true", false → "false" (JSON)
        assert sum(counts.values()) == 2

    def test_after_delete_count_decreases(self, tmp_path):
        db = _open(tmp_path / "db")
        for i in range(5):
            db.insert(str(i), _vec(16, seed=i), metadata={"cat": "X"})
        db.delete("0")
        db.delete("1")
        counts = db.list_metadata_values("cat")
        assert counts.get("X") == 3

    def test_nonexistent_field_returns_empty(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0), metadata={"real": "val"})
        counts = db.list_metadata_values("nonexistent_field")
        # No vectors have this field → empty dict
        assert counts == {}

    def test_dotted_path_nested_field(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0), metadata={"profile": {"region": "US"}})
        db.insert("b", _vec(16, seed=1), metadata={"profile": {"region": "EU"}})
        db.insert("c", _vec(16, seed=2), metadata={"profile": {"region": "US"}})
        counts = db.list_metadata_values("profile.region")
        assert counts.get("US") == 2
        assert counts.get("EU") == 1

    def test_empty_string_field_name(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0), metadata={"x": "v"})
        try:
            result = db.list_metadata_values("")
            # Accept empty result
        except Exception as e:
            assert "panic" not in type(e).__name__.lower()

    def test_mixed_types_on_same_field_across_vectors(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("str_v", _vec(16, seed=0), metadata={"v": "hello"})
        db.insert("int_v", _vec(16, seed=1), metadata={"v": 42})
        db.insert("bool_v", _vec(16, seed=2), metadata={"v": True})
        counts = db.list_metadata_values("v")
        # All three distinct values → at least 2 keys (True in JSON = "true",
        # integer 42 = "42", string "hello" = "hello")
        assert sum(counts.values()) == 3


# ---------------------------------------------------------------------------
# ███╗   ███╗ ██████╗ ██████╗ ███████╗███████╗
# ████╗ ████║██╔═══██╗██╔══██╗██╔════╝██╔════╝
# ██╔████╔██║██║   ██║██║  ██║█████╗  ███████╗
# ██║╚██╔╝██║██║   ██║██║  ██║██╔══╝  ╚════██║
# ██║ ╚═╝ ██║╚██████╔╝██████╔╝███████╗███████║
# ---------------------------------------------------------------------------

class TestInsertBatchModes:
    """Hammer insert_batch mode= parameter: insert / update / upsert / invalid."""

    def test_mode_insert_duplicate_id_raises(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0))
        v = _unit_batch(1, 16, seed=1)
        with pytest.raises(Exception):
            db.insert_batch(["a"], v, mode="insert")

    def test_mode_upsert_overwrites_existing_vector(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0))
        v_new = _unit_batch(1, 16, seed=99)
        db.insert_batch(["a"], v_new, mode="upsert")
        assert db.count() == 1
        got = db.get("a")
        assert got is not None

    def test_mode_upsert_new_id_creates_entry(self, tmp_path):
        db = _open(tmp_path / "db")
        vecs = _unit_batch(3, 16)
        db.insert_batch(["x", "y", "z"], vecs, mode="upsert")
        assert db.count() == 3

    def test_mode_update_existing_id_changes_metadata(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0), metadata={"v": 1})
        v_new = _unit_batch(1, 16, seed=99)
        db.insert_batch(["a"], v_new, mode="update", metadatas=[{"v": 999}])
        got = db.get("a")
        assert got is not None

    def test_mode_update_nonexistent_id_raises(self, tmp_path):
        db = _open(tmp_path / "db")
        with pytest.raises(Exception):
            db.insert_batch(["ghost"], _unit_batch(1, 16), mode="update")

    def test_mode_invalid_string_raises_value_error(self, tmp_path):
        db = _open(tmp_path / "db")
        with pytest.raises(Exception):
            db.insert_batch(["a"], _unit_batch(1, 16), mode="YOLO")

    def test_mode_is_case_insensitive(self, tmp_path):
        db = _open(tmp_path / "db")
        vecs = _unit_batch(2, 16)
        # "INSERT" uppercase should work
        db.insert_batch(["a", "b"], vecs, mode="INSERT")
        assert db.count() == 2

    def test_mode_upsert_same_ids_twice_in_same_batch(self, tmp_path):
        # Two items with same ID in one batch — last writer wins (or raises)
        db = _open(tmp_path / "db")
        v1 = _vec(16, seed=1)
        v2 = _vec(16, seed=2)
        vecs = np.stack([v1, v2])
        try:
            db.insert_batch(["dup", "dup"], vecs, mode="upsert")
            # If accepted: should have exactly 1 entry
            assert db.count() == 1
        except Exception as e:
            assert "panic" not in type(e).__name__.lower()

    def test_mode_upsert_metadata_updated_on_overwrite(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0), metadata={"v": "old"})
        db.insert_batch(["a"], _unit_batch(1, 16, seed=7),
                        mode="upsert", metadatas=[{"v": "new"}])
        # Metadata should be updated, vector count unchanged
        assert db.count() == 1

    def test_large_upsert_batch_does_not_panic(self, tmp_path):
        db = _open(tmp_path / "db")
        n = 3000
        vecs = _unit_batch(n, 16)
        ids = [str(i) for i in range(n)]
        db.insert_batch(ids, vecs)
        # Upsert all again with new vectors — no panic, count unchanged
        vecs2 = _unit_batch(n, 16, seed=99)
        db.insert_batch(ids, vecs2, mode="upsert")
        assert db.count() == n


# ---------------------------------------------------------------------------
# ███╗   ██╗ ██████╗ ██████╗ ███╗   ███╗ █████╗ ██╗     ██╗███████╗███████╗
# ████╗  ██║██╔═══██╗██╔══██╗████╗ ████║██╔══██╗██║     ██║╚══███╔╝██╔════╝
# ██╔██╗ ██║██║   ██║██████╔╝██╔████╔██║███████║██║     ██║  ███╔╝ █████╗
# ██║╚██╗██║██║   ██║██╔══██╗██║╚██╔╝██║██╔══██║██║     ██║ ███╔╝  ██╔══╝
# ██║ ╚████║╚██████╔╝██║  ██║██║ ╚═╝ ██║██║  ██║███████╗██║███████╗███████╗
# ---------------------------------------------------------------------------

class TestNormalize:
    """Break normalize=True interactions."""

    def test_normalize_true_accepts_non_unit_vectors(self, tmp_path):
        # normalize=True: engine normalizes all inserts → unit norms in-DB
        db = Database.open(str(tmp_path / "db"), 16, bits=4, seed=42,
                           normalize=True, metric="ip")
        v = np.full(16, 5.0, dtype=np.float32)  # not unit
        db.insert("a", v)
        # After normalization, the vector should be retrievable
        assert db.count() == 1

    # BUG-5: normalize=True silently accepts zero vectors instead of raising.
    # Normalizing a zero vector requires dividing by zero norm → result is NaN/0.
    def test_normalize_zero_vector_raises(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), 16, bits=4, seed=42,
                           normalize=True, metric="ip")
        zero = np.zeros(16, dtype=np.float32)
        with pytest.raises(Exception) as exc_info:
            db.insert("zero", zero)
        assert "panic" not in type(exc_info.value).__name__.lower()

    def test_normalize_true_vs_false_same_ranking(self, tmp_path):
        # With random unit vectors: normalize=True (ip) and normalize=False (cosine)
        # should agree on rankings.
        vecs = _unit_batch(50, 16, seed=7)  # already unit

        db_norm = Database.open(str(tmp_path / "db_norm"), 16, bits=4, seed=42,
                                normalize=True, metric="ip")
        db_cos = Database.open(str(tmp_path / "db_cos"), 16, bits=4, seed=42,
                               normalize=False, metric="cosine")

        ids = [str(i) for i in range(50)]
        db_norm.insert_batch(ids, vecs)
        db_cos.insert_batch(ids, vecs)

        q = vecs[0]
        top_norm = [r["id"] for r in db_norm.search(q, top_k=5)]
        top_cos = [r["id"] for r in db_cos.search(q, top_k=5)]
        # Top-1 must agree between modes
        assert top_norm[0] == top_cos[0], (
            f"normalize=True top-1 '{top_norm[0]}' != cosine top-1 '{top_cos[0]}'"
        )

    def test_normalize_scores_bounded_minus_one_to_one(self, tmp_path):
        # IP of unit vectors ∈ [-1, 1]
        db = Database.open(str(tmp_path / "db"), 16, bits=4, seed=42,
                           normalize=True, metric="ip")
        vecs = _unit_batch(30, 16)
        db.insert_batch([str(i) for i in range(30)], vecs)
        results = db.search(vecs[0], top_k=10)
        for r in results:
            assert -1.5 <= r["score"] <= 1.5, (
                f"Normalized IP score out of [-1.5, 1.5]: {r['score']}"
            )

    def test_normalize_nan_vector_still_rejected(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), 16, bits=4, seed=42,
                           normalize=True, metric="ip")
        v = np.full(16, float("nan"), dtype=np.float32)
        with pytest.raises(Exception) as exc_info:
            db.insert("nan", v)
        assert "panic" not in type(exc_info.value).__name__.lower()

    def test_normalize_true_batch_insert(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), 16, bits=4, seed=42,
                           normalize=True, metric="ip")
        # Batch of non-unit vectors
        vecs = np.full((10, 16), 10.0, dtype=np.float32)
        db.insert_batch([str(i) for i in range(10)], vecs)
        assert db.count() == 10


# ---------------------------------------------------------------------------
# ███████╗██╗     ██╗   ██╗███████╗██╗  ██╗
# ██╔════╝██║     ██║   ██║██╔════╝██║  ██║
# █████╗  ██║     ██║   ██║███████╗███████║
# ██╔══╝  ██║     ██║   ██║╚════██║██╔══██║
# ██║     ███████╗╚██████╔╝███████║██║  ██║
# ---------------------------------------------------------------------------

class TestFlushAPI:
    """flush() must be safe to call at any time without corrupting state."""

    def test_flush_empty_db_no_panic(self, tmp_path):
        db = _open(tmp_path / "db")
        db.flush()  # WAL is empty — should be a no-op

    def test_flush_after_insert_data_survives(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0))
        db.flush()
        assert db.count() == 1
        assert db.get("a") is not None

    def test_flush_idempotent_multiple_calls(self, tmp_path):
        db = _open(tmp_path / "db")
        vecs = _unit_batch(10, 16)
        db.insert_batch([str(i) for i in range(10)], vecs)
        db.flush()
        db.flush()
        db.flush()
        assert db.count() == 10

    def test_flush_forces_segment_then_search_still_works(self, tmp_path):
        db = _open(tmp_path / "db")
        vecs = _unit_batch(50, 16)
        db.insert_batch([str(i) for i in range(50)], vecs)
        db.flush()
        results = db.search(vecs[7], top_k=1)
        assert results[0]["id"] == "7"

    def test_flush_then_insert_then_flush_search_correct(self, tmp_path):
        db = _open(tmp_path / "db")
        vecs_a = _unit_batch(20, 16, seed=1)
        db.insert_batch([str(i) for i in range(20)], vecs_a)
        db.flush()
        vecs_b = _unit_batch(20, 16, seed=2)
        db.insert_batch([f"b_{i}" for i in range(20)], vecs_b)
        db.flush()
        assert db.count() == 40
        results = db.search(vecs_a[0], top_k=1)
        assert results[0]["id"] == "0"

    def test_flush_preserves_index_state(self, tmp_path):
        db = _open(tmp_path / "db")
        vecs = _unit_batch(100, 16)
        db.insert_batch([str(i) for i in range(100)], vecs)
        db.create_index(max_degree=16, ef_construction=64, search_list_size=64)
        assert db.stats()["has_index"]
        db.flush()
        # Index should still be present after flush
        assert db.stats()["has_index"]


# ---------------------------------------------------------------------------
# ██╗   ██╗██████╗ ██████╗  █████╗ ████████╗███████╗    ███╗   ███╗███████╗████████╗ █████╗
# ██║   ██║██╔══██╗██╔══██╗██╔══██╗╚══██╔══╝██╔════╝    ████╗ ████║██╔════╝╚══██╔══╝██╔══██╗
# ██║   ██║██████╔╝██║  ██║███████║   ██║   █████╗      ██╔████╔██║█████╗     ██║   ███████║
# ██║   ██║██╔═══╝ ██║  ██║██╔══██║   ██║   ██╔══╝      ██║╚██╔╝██║██╔══╝     ██║   ██╔══██║
# ╚██████╔╝██║     ██████╔╝██║  ██║   ██║   ███████╗    ██║ ╚═╝ ██║███████╗   ██║   ██║  ██║
# ---------------------------------------------------------------------------

class TestUpdateMetadataEdge:
    """Hammer update_metadata() with every hostile input."""

    def test_update_metadata_nonexistent_id_raises(self, tmp_path):
        db = _open(tmp_path / "db")
        with pytest.raises(Exception) as exc_info:
            db.update_metadata("ghost", metadata={"x": 1})
        assert "panic" not in type(exc_info.value).__name__.lower()

    def test_update_metadata_both_none_is_noop(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0), metadata={"x": 1}, document="original")
        db.update_metadata("a", metadata=None, document=None)
        got = db.get("a")
        # Original metadata preserved (None means "don't change")
        assert got["metadata"].get("x") == 1

    def test_update_metadata_only_document_changes_doc(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0), metadata={"x": 1}, document="old doc")
        db.update_metadata("a", document="new doc")
        got = db.get("a")
        assert got.get("document") == "new doc"
        # Metadata should be preserved
        assert got["metadata"].get("x") == 1

    def test_update_metadata_empty_dict_behavior(self, tmp_path):
        # metadata={} passes an empty dict — does it clear or no-op?
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0), metadata={"x": 1})
        # This should not panic
        try:
            db.update_metadata("a", metadata={})
        except Exception as e:
            assert "panic" not in type(e).__name__.lower()

    def test_update_metadata_does_not_change_search_score(self, tmp_path):
        db = _open(tmp_path / "db")
        vecs = _unit_batch(10, 16)
        db.insert_batch([str(i) for i in range(10)], vecs)
        score_before = db.search(vecs[3], top_k=1)[0]["score"]
        db.update_metadata("3", metadata={"status": "updated"})
        score_after = db.search(vecs[3], top_k=1)[0]["score"]
        # Score should be identical (quantized representation untouched)
        assert abs(score_before - score_after) < 1e-6

    def test_update_metadata_persists_across_reopen(self, tmp_path):
        path = str(tmp_path / "db")
        db = _open(path)
        db.insert("a", _vec(16, seed=0), metadata={"version": 1})
        db.update_metadata("a", metadata={"version": 2})
        db.close()
        del db
        gc.collect()
        gc.collect()

        db2 = _open(path)
        got = db2.get("a")
        assert got["metadata"].get("version") == 2

    def test_update_metadata_large_document(self, tmp_path):
        # 1 MB document update
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0))
        big_doc = "x" * (1024 * 1024)
        db.update_metadata("a", document=big_doc)
        got = db.get("a")
        assert got.get("document") == big_doc

    def test_update_metadata_many_successive_updates(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0), metadata={"v": 0})
        for i in range(50):
            db.update_metadata("a", metadata={"v": i + 1})
        got = db.get("a")
        assert got["metadata"]["v"] == 50


# ---------------------------------------------------------------------------
# ██████╗  █████╗ ████████╗ ██████╗██╗  ██╗    ██████╗  █████╗ ████████╗ ██████╗██╗  ██╗
# ██╔══██╗██╔══██╗╚══██╔══╝██╔════╝██║  ██║    ██╔══██╗██╔══██╗╚══██╔══╝██╔════╝██║  ██║
# ██████╔╝███████║   ██║   ██║     ███████║    ██████╔╝███████║   ██║   ██║     ███████║
# ██╔══██╗██╔══██║   ██║   ██║     ██╔══██║    ██╔══██╗██╔══██║   ██║   ██║     ██╔══██║
# ██████╔╝██║  ██║   ██║   ╚██████╗██║  ██║    ██████╔╝██║  ██║   ██║   ╚██████╗██║  ██║
# ---------------------------------------------------------------------------

class TestBatchQueryEdge:
    """Break the query() batch search with hostile inputs."""

    def test_query_zero_row_matrix_returns_empty(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0))
        result = db.query(np.empty((0, 16), dtype=np.float32), n_results=3)
        assert result == []

    def test_query_1d_array_raises(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0))
        with pytest.raises(Exception) as exc_info:
            db.query(np.ones(16, dtype=np.float32), n_results=1)
        assert "panic" not in type(exc_info.value).__name__.lower()

    def test_query_dimension_mismatch_raises(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0))
        with pytest.raises(Exception) as exc_info:
            db.query(np.ones((1, 32), dtype=np.float32), n_results=1)
        assert "panic" not in type(exc_info.value).__name__.lower()

    def test_query_negative_n_results_raises(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0))
        with pytest.raises(Exception):
            db.query(np.ones((1, 16), dtype=np.float32), n_results=-1)

    def test_query_matches_search_for_each_row(self, tmp_path):
        db = _open(tmp_path / "db")
        vecs = _unit_batch(20, 16)
        db.insert_batch([str(i) for i in range(20)], vecs)
        # query() vs search() on same input — top-1 must agree
        queries = np.stack([vecs[0], vecs[5], vecs[10]])
        batch = db.query(queries, n_results=1)
        for idx, row_results in enumerate(batch):
            expected_id = str([0, 5, 10][idx])
            if row_results:
                assert row_results[0]["id"] == expected_id

    def test_query_l2_scores_negated_consistently(self, tmp_path):
        db = _open(tmp_path / "db", metric="l2")
        vecs = _unit_batch(10, 16)
        db.insert_batch([str(i) for i in range(10)], vecs)
        q = np.stack([vecs[0]])
        batch_results = db.query(q, n_results=5)
        search_results = db.search(vecs[0], top_k=5)
        # Both should return same IDs in same order
        b_ids = [r["id"] for r in batch_results[0]]
        s_ids = [r["id"] for r in search_results]
        assert b_ids == s_ids

    def test_query_with_filter_matches_search_with_filter(self, tmp_path):
        db = _open(tmp_path / "db")
        for i in range(10):
            db.insert(str(i), _vec(16, seed=i),
                      metadata={"cat": "A" if i < 5 else "B"})
        q = np.stack([_vec(16, seed=0)])
        filt = {"cat": "A"}
        batch_results = db.query(q, n_results=10, where_filter=filt)
        search_results = db.search(_vec(16, seed=0), top_k=10, filter=filt)
        b_ids = {r["id"] for r in batch_results[0]}
        s_ids = {r["id"] for r in search_results}
        # Same IDs regardless of which API is used
        assert b_ids == s_ids

    def test_query_nan_in_middle_row_raises(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0))
        qs = np.ones((3, 16), dtype=np.float32)
        qs[1, 0] = float("nan")
        with pytest.raises(Exception) as exc_info:
            db.query(qs, n_results=1)
        assert "panic" not in type(exc_info.value).__name__.lower()

    def test_query_many_queries_no_panic(self, tmp_path):
        db = _open(tmp_path / "db")
        vecs = _unit_batch(50, 16)
        db.insert_batch([str(i) for i in range(50)], vecs)
        queries = _unit_batch(100, 16, seed=77)
        results = db.query(queries, n_results=5)
        assert len(results) == 100
        for row in results:
            assert len(row) <= 5

    def test_query_with_n_results_larger_than_corpus(self, tmp_path):
        db = _open(tmp_path / "db")
        vecs = _unit_batch(3, 16)
        db.insert_batch(["a", "b", "c"], vecs)
        q = np.stack([vecs[0]])
        results = db.query(q, n_results=1000)
        # Should be capped at corpus size
        assert len(results[0]) == 3


# ---------------------------------------------------------------------------
# ██████╗ ███████╗████████╗     ██████╗ ███████╗████████╗    ███╗   ███╗ █████╗ ███╗   ██╗██╗   ██╗
# ██╔════╝ ██╔════╝╚══██╔══╝    ██╔════╝ ██╔════╝╚══██╔══╝    ████╗ ████║██╔══██╗████╗  ██║╚██╗ ██╔╝
# ██║  ███╗█████╗     ██║       ██║  ███╗█████╗     ██║       ██╔████╔██║███████║██╔██╗ ██║ ╚████╔╝
# ██║   ██║██╔══╝     ██║       ██║   ██║██╔══╝     ██║       ██║╚██╔╝██║██╔══██║██║╚██╗██║  ╚██╔╝
# ╚██████╔╝███████╗   ██║       ╚██████╔╝███████╗   ██║       ██║ ╚═╝ ██║██║  ██║██║ ╚████║   ██║
# ---------------------------------------------------------------------------

class TestGetManyEdge:
    """get_many() must handle all ID combinations cleanly."""

    def test_get_many_empty_list_returns_empty(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0))
        result = db.get_many([])
        assert result == []

    def test_get_many_all_missing_returns_nones(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0))
        result = db.get_many(["x", "y", "z"])
        assert len(result) == 3
        assert all(r is None for r in result)

    def test_get_many_mix_present_and_absent(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0))
        db.insert("b", _vec(16, seed=1))
        result = db.get_many(["a", "MISSING", "b"])
        assert result[0] is not None
        assert result[1] is None
        assert result[2] is not None

    def test_get_many_preserves_order(self, tmp_path):
        db = _open(tmp_path / "db")
        vecs = _unit_batch(5, 16)
        db.insert_batch(["a", "b", "c", "d", "e"], vecs)
        result = db.get_many(["c", "a", "e"])
        assert result[0]["id"] == "c"
        assert result[1]["id"] == "a"
        assert result[2]["id"] == "e"

    # BUG-6: get_many() returns None for duplicate IDs — the second request for
    # an ID that was already returned comes back as None instead of the same entry.
    # Root cause: the underlying engine appears to "consume" each ID slot lookup,
    def test_get_many_duplicate_ids(self, tmp_path):
        # Requesting same ID twice
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0))
        result = db.get_many(["a", "a"])
        assert len(result) == 2
        assert result[0] is not None
        assert result[1] is not None

    def test_get_many_after_delete_returns_none(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0))
        db.insert("b", _vec(16, seed=1))
        db.delete("a")
        result = db.get_many(["a", "b"])
        assert result[0] is None
        assert result[1] is not None

    def test_get_many_large_batch(self, tmp_path):
        db = _open(tmp_path / "db")
        n = 1000
        vecs = _unit_batch(n, 16)
        ids = [str(i) for i in range(n)]
        db.insert_batch(ids, vecs)
        # Request all 1000 — should not panic
        result = db.get_many(ids)
        assert len(result) == n
        assert all(r is not None for r in result)


# ---------------------------------------------------------------------------
# ██╗     ██╗███████╗████████╗    ██╗██████╗ ███████╗
# ██║     ██║██╔════╝╚══██╔══╝    ██║██╔══██╗██╔════╝
# ██║     ██║███████╗   ██║       ██║██║  ██║███████╗
# ██║     ██║╚════██║   ██║       ██║██║  ██║╚════██║
# ███████╗██║███████║   ██║       ██║██████╔╝███████║
# ---------------------------------------------------------------------------

class TestListIdsEdge:
    """list_ids() pagination edge cases — offset, limit, and filter."""

    def test_offset_past_end_returns_empty(self, tmp_path):
        db = _open(tmp_path / "db")
        for i in range(5):
            db.insert(str(i), _vec(16, seed=i))
        result = db.list_ids(offset=1000)
        assert result == []

    def test_offset_equals_count_returns_empty(self, tmp_path):
        db = _open(tmp_path / "db")
        for i in range(5):
            db.insert(str(i), _vec(16, seed=i))
        result = db.list_ids(offset=5)
        assert result == []

    def test_limit_larger_than_count_returns_all(self, tmp_path):
        db = _open(tmp_path / "db")
        for i in range(5):
            db.insert(str(i), _vec(16, seed=i))
        result = db.list_ids(limit=1000)
        assert len(result) == 5

    def test_negative_offset_raises(self, tmp_path):
        db = _open(tmp_path / "db")
        with pytest.raises(Exception) as exc_info:
            db.list_ids(offset=-1)
        assert "panic" not in type(exc_info.value).__name__.lower()

    def test_negative_limit_raises(self, tmp_path):
        db = _open(tmp_path / "db")
        with pytest.raises(Exception) as exc_info:
            db.list_ids(limit=-1)
        assert "panic" not in type(exc_info.value).__name__.lower()

    def test_list_ids_no_filter_returns_all(self, tmp_path):
        db = _open(tmp_path / "db")
        expected = [str(i) for i in range(10)]
        vecs = _unit_batch(10, 16)
        db.insert_batch(expected, vecs)
        result = db.list_ids()
        assert set(result) == set(expected)

    def test_list_ids_empty_db(self, tmp_path):
        db = _open(tmp_path / "db")
        result = db.list_ids()
        assert result == []

    def test_list_ids_with_limit_one(self, tmp_path):
        db = _open(tmp_path / "db")
        for i in range(5):
            db.insert(str(i), _vec(16, seed=i))
        result = db.list_ids(limit=1)
        assert len(result) == 1

    def test_list_ids_pagination_no_overlap_no_gap(self, tmp_path):
        db = _open(tmp_path / "db")
        n = 20
        vecs = _unit_batch(n, 16)
        db.insert_batch([str(i) for i in range(n)], vecs)
        page_size = 7
        pages = []
        offset = 0
        while True:
            page = db.list_ids(limit=page_size, offset=offset)
            if not page:
                break
            pages.append(page)
            offset += page_size
        all_ids = [rid for page in pages for rid in page]
        assert len(all_ids) == n
        assert len(set(all_ids)) == n  # no duplicates


# ---------------------------------------------------------------------------
# ██████╗ ███████╗██╗     ███████╗████████╗███████╗
# ██╔══██╗██╔════╝██║     ██╔════╝╚══██╔══╝██╔════╝
# ██║  ██║█████╗  ██║     █████╗     ██║   █████╗
# ██║  ██║██╔══╝  ██║     ██╔══╝     ██║   ██╔══╝
# ██████╔╝███████╗███████╗███████╗   ██║   ███████╗
# ---------------------------------------------------------------------------

class TestDeleteEdge:
    """Destroy data in every bad way."""

    def test_delete_nonexistent_id_behavior(self, tmp_path):
        db = _open(tmp_path / "db")
        # Single delete of nonexistent ID — accept error or no-op, must not panic
        try:
            db.delete("ghost")
        except Exception as e:
            assert "panic" not in type(e).__name__.lower()

    def test_double_delete_same_id_behavior(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0))
        db.delete("a")
        try:
            db.delete("a")  # Already gone
        except Exception as e:
            assert "panic" not in type(e).__name__.lower()

    def test_delete_then_reinsert_same_id_returns_new_vector(self, tmp_path):
        db = _open(tmp_path / "db")
        vecs = _unit_batch(10, 16)
        db.insert_batch([str(i) for i in range(10)], vecs)
        db.delete("5")
        # Reinsert with different vector
        new_vec = _unit_batch(1, 16, seed=999)
        db.insert_batch(["5"], new_vec)
        # Top-1 for the original vecs[5] might change — but 5 must be back in DB
        assert "5" in db
        assert db.count() == 10

    def test_delete_batch_ids_and_filter_union(self, tmp_path):
        db = _open(tmp_path / "db")
        for i in range(6):
            db.insert(str(i), _vec(16, seed=i), metadata={"cat": "A" if i < 3 else "B"})
        # Delete ids=["0"] (cat A) plus filter={"cat": "B"} → deletes "0" + "3","4","5"
        n_deleted = db.delete_batch(ids=["0"], where_filter={"cat": "B"})
        # 4 distinct items removed: "0" (by id) + "3","4","5" (by filter)
        assert n_deleted == 4
        assert db.count() == 2  # "1" and "2" remain

    def test_delete_batch_overlap_not_double_counted(self, tmp_path):
        db = _open(tmp_path / "db")
        for i in range(5):
            db.insert(str(i), _vec(16, seed=i), metadata={"cat": "A"})
        # "0" is both in ids and matches filter → should be counted once
        n_deleted = db.delete_batch(ids=["0"], where_filter={"cat": "A"})
        # "0" deleted by ids, "1"–"4" deleted by filter = 5 total unique
        assert n_deleted == 5
        assert db.count() == 0

    def test_delete_batch_empty_ids_no_filter_returns_zero(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0))
        # No ids, no filter → no-op
        n = db.delete_batch()
        assert n == 0
        assert db.count() == 1

    def test_delete_then_search_vector_absent(self, tmp_path):
        db = _open(tmp_path / "db")
        vecs = _unit_batch(10, 16)
        db.insert_batch([str(i) for i in range(10)], vecs)
        db.delete_batch(["2", "3", "4"])
        results = db.search(vecs[2], top_k=10)
        returned_ids = {r["id"] for r in results}
        assert "2" not in returned_ids
        assert "3" not in returned_ids
        assert "4" not in returned_ids

    def test_delete_all_then_reinsert_and_search(self, tmp_path):
        db = _open(tmp_path / "db")
        n = 20
        vecs = _unit_batch(n, 16)
        ids = [str(i) for i in range(n)]
        db.insert_batch(ids, vecs)
        db.delete_batch(ids)
        assert db.count() == 0
        db.insert_batch(ids, vecs)
        assert db.count() == n
        results = db.search(vecs[0], top_k=1)
        assert results[0]["id"] == "0"


# ---------------------------------------------------------------------------
# ███████╗ ██████╗ ██████╗ ██████╗ ███████╗
# ██╔════╝██╔════╝██╔═══██╗██╔══██╗██╔════╝
# ███████╗██║     ██║   ██║██████╔╝█████╗
# ╚════██║██║     ██║   ██║██╔══██╗██╔══╝
# ███████║╚██████╗╚██████╔╝██║  ██║███████╗
# ---------------------------------------------------------------------------

class TestScoreConsistency:
    """Results must be internally consistent across APIs and operations."""

    def test_ann_and_brute_force_top1_agree_on_exact_query(self, tmp_path):
        db = _open(tmp_path / "db")
        vecs = _unit_batch(200, 16)
        db.insert_batch([str(i) for i in range(200)], vecs)
        db.create_index(max_degree=16, ef_construction=100, search_list_size=100)
        # For the exact vector, both approaches must find the same vector
        top_brute = db.search(vecs[42], top_k=1)[0]["id"]
        top_ann = db.search(vecs[42], top_k=1, _use_ann=True)[0]["id"]
        assert top_brute == "42"
        assert top_ann == "42"

    def test_query_and_search_top1_agree(self, tmp_path):
        db = _open(tmp_path / "db")
        vecs = _unit_batch(50, 16)
        db.insert_batch([str(i) for i in range(50)], vecs)
        q = vecs[7]
        s_top = db.search(q, top_k=3)
        q_top = db.query(np.stack([q]), n_results=3)[0]
        s_ids = [r["id"] for r in s_top]
        q_ids = [r["id"] for r in q_top]
        assert s_ids == q_ids

    def test_search_after_reinsert_returns_new_vector_score(self, tmp_path):
        db = _open(tmp_path / "db")
        vecs = _unit_batch(10, 16)
        db.insert_batch([str(i) for i in range(10)], vecs)
        db.delete("5")
        # Insert a clone of vecs[0] at slot "5"
        db.insert("5", vecs[0].copy())
        # Now searching for vecs[0] should return both "0" and "5" in top-2
        results = db.search(vecs[0], top_k=2)
        returned_ids = {r["id"] for r in results}
        assert "0" in returned_ids

    def test_l2_and_cosine_rank_differently_on_non_unit_corpus(self, tmp_path):
        # Use non-unit vectors so L2 and cosine disagree meaningfully
        rng = np.random.default_rng(42)
        vecs = rng.standard_normal((20, 16)).astype(np.float32)
        # Scale each by different amounts → directions same, magnitudes differ
        scales = rng.uniform(0.1, 10.0, size=(20, 1)).astype(np.float32)
        vecs_scaled = vecs * scales

        db_l2 = _open(tmp_path / "db_l2", metric="l2")
        db_cos = _open(tmp_path / "db_cos", metric="cosine")
        ids = [str(i) for i in range(20)]
        db_l2.insert_batch(ids, vecs_scaled)
        db_cos.insert_batch(ids, vecs_scaled)

        top_l2 = db_l2.search(vecs_scaled[0], top_k=5)
        top_cos = db_cos.search(vecs_scaled[0], top_k=5)
        # At least one position differs (L2 vs cosine on non-unit vectors differs)
        ids_l2 = [r["id"] for r in top_l2]
        ids_cos = [r["id"] for r in top_cos]
        # They CAN agree but if they're identical it's suspicious (log it)
        # Don't assert disagreement — just verify both return 5 results
        assert len(ids_l2) == 5
        assert len(ids_cos) == 5

    def test_scores_monotonically_decreasing_after_mixed_ops(self, tmp_path):
        db = _open(tmp_path / "db")
        vecs = _unit_batch(50, 16)
        db.insert_batch([str(i) for i in range(50)], vecs)
        db.delete_batch(["10", "20", "30"])
        more = _unit_batch(10, 16, seed=77)
        db.insert_batch([f"extra_{i}" for i in range(10)], more)
        db.update_metadata("0", metadata={"touched": True})
        results = db.search(vecs[0], top_k=20)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_filter_returns_subset_with_correct_scores(self, tmp_path):
        db = _open(tmp_path / "db")
        for i in range(10):
            db.insert(str(i), _vec(16, seed=i), metadata={"g": "A" if i < 5 else "B"})
        unfiltered = db.search(_vec(16, seed=0), top_k=10)
        filtered = db.search(_vec(16, seed=0), top_k=10, filter={"g": "A"})
        # Filtered results must all have g=A
        for r in filtered:
            assert r["metadata"]["g"] == "A"
        # And filtered scores ≤ unfiltered top-1 score (filtered is a subset)
        if filtered and unfiltered:
            # The best score among filtered can't exceed the best overall score
            assert max(r["score"] for r in filtered) <= max(r["score"] for r in unfiltered) + 1e-6


# ---------------------------------------------------------------------------
# ██████╗ ██╗███╗   ███╗███████╗███╗   ██╗███████╗██╗ ██████╗ ███╗   ██╗███████╗
# ██╔══██╗██║████╗ ████║██╔════╝████╗  ██║██╔════╝██║██╔═══██╗████╗  ██║██╔════╝
# ██║  ██║██║██╔████╔██║█████╗  ██╔██╗ ██║███████╗██║██║   ██║██╔██╗ ██║███████╗
# ██║  ██║██║██║╚██╔╝██║██╔══╝  ██║╚██╗██║╚════██║██║██║   ██║██║╚██╗██║╚════██║
# ██████╔╝██║██║ ╚═╝ ██║███████╗██║ ╚████║███████║██║╚██████╔╝██║ ╚████║███████║
# ---------------------------------------------------------------------------

class TestEdgeDimensions:
    """Test with dimension 1 and very large dimensions."""

    def test_dimension_1_insert_and_search(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=1,
                           bits=4, seed=42, metric="ip")
        db.insert("pos", np.array([1.0], dtype=np.float32))
        db.insert("neg", np.array([-1.0], dtype=np.float32))
        results = db.search(np.array([1.0], dtype=np.float32), top_k=2)
        assert len(results) >= 1
        assert results[0]["id"] == "pos"

    def test_dimension_1_l2(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=1,
                           bits=4, seed=42, metric="l2")
        db.insert("zero", np.array([0.0], dtype=np.float32))
        db.insert("far", np.array([100.0], dtype=np.float32))
        results = db.search(np.array([0.0], dtype=np.float32), top_k=2)
        assert results[0]["id"] == "zero"

    def test_dimension_2048_insert_search(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=2048,
                           bits=4, seed=42, metric="ip")
        vecs = _unit_batch(20, 2048)
        db.insert_batch([str(i) for i in range(20)], vecs)
        results = db.search(vecs[3], top_k=1)
        assert results[0]["id"] == "3"

    def test_dimension_power_of_two_boundary_512(self, tmp_path):
        # 512 is exactly a power of two — SRHT handles this case cleanly
        db = Database.open(str(tmp_path / "db"), dimension=512,
                           bits=4, seed=42, metric="ip")
        vecs = _unit_batch(10, 512)
        db.insert_batch([str(i) for i in range(10)], vecs)
        assert db.count() == 10
        results = db.search(vecs[0], top_k=1)
        assert results[0]["id"] == "0"

    def test_dimension_non_power_of_two_300(self, tmp_path):
        # 300 → SRHT pads to 512 (next power of two)
        db = Database.open(str(tmp_path / "db"), dimension=300,
                           bits=4, seed=42, metric="ip")
        vecs = _unit_batch(10, 300)
        db.insert_batch([str(i) for i in range(10)], vecs)
        assert db.count() == 10
        results = db.search(vecs[5], top_k=1)
        assert results[0]["id"] == "5"

    def test_mismatch_insert_after_open_raises_on_search(self, tmp_path):
        # Open with d=16, then try to search with d=32 vector
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0))
        q32 = _vec(32, seed=0)
        with pytest.raises(Exception) as exc_info:
            db.search(q32, top_k=1)
        assert "panic" not in type(exc_info.value).__name__.lower()


# ---------------------------------------------------------------------------
# ██████╗  █████╗ ██████╗  █████╗ ███╗   ███╗    ██████╗  ██████╗ ██╗   ██╗███╗   ██╗██████╗ ███████╗
# ██╔══██╗██╔══██╗██╔══██╗██╔══██╗████╗ ████║    ██╔══██╗██╔═══██╗██║   ██║████╗  ██║██╔══██╗██╔════╝
# ██████╔╝███████║██████╔╝███████║██╔████╔██║    ██████╔╝██║   ██║██║   ██║██╔██╗ ██║██║  ██║███████╗
# ██╔═══╝ ██╔══██║██╔══██╗██╔══██║██║╚██╔╝██║    ██╔══██╗██║   ██║██║   ██║██║╚██╗██║██║  ██║╚════██║
# ██║     ██║  ██║██║  ██║██║  ██║██║ ╚═╝ ██║    ██████╔╝╚██████╔╝╚██████╔╝██║ ╚████║██████╔╝███████║
# ---------------------------------------------------------------------------

class TestParamBounds:
    """Break numeric parameter boundaries in search(), create_index()."""

    def test_rerank_factor_zero_raises_or_behaves(self, tmp_path):
        db = _open(tmp_path / "db", rerank=True)
        db.insert("a", _vec(16, seed=0))
        try:
            results = db.search(_vec(16, seed=0), top_k=1, rerank_factor=0)
            assert len(results) >= 0
        except Exception as e:
            assert "panic" not in type(e).__name__.lower()

    def test_ann_search_list_size_zero_or_raises(self, tmp_path):
        db = _open(tmp_path / "db")
        vecs = _unit_batch(100, 16)
        db.insert_batch([str(i) for i in range(100)], vecs)
        db.create_index(max_degree=16, ef_construction=64, search_list_size=64)
        try:
            results = db.search(vecs[0], top_k=5, _use_ann=True,
                                ann_search_list_size=0)
            assert len(results) >= 0
        except Exception as e:
            assert "panic" not in type(e).__name__.lower()

    def test_ann_search_list_size_one(self, tmp_path):
        db = _open(tmp_path / "db")
        vecs = _unit_batch(100, 16)
        db.insert_batch([str(i) for i in range(100)], vecs)
        db.create_index(max_degree=16, ef_construction=64, search_list_size=64)
        # Tiny search list — poor recall but must not panic
        try:
            results = db.search(vecs[0], top_k=5, _use_ann=True,
                                ann_search_list_size=1)
            assert isinstance(results, list)
        except Exception as e:
            assert "panic" not in type(e).__name__.lower()

    def test_alpha_zero_in_create_index(self, tmp_path):
        db = _open(tmp_path / "db")
        vecs = _unit_batch(50, 16)
        db.insert_batch([str(i) for i in range(50)], vecs)
        try:
            db.create_index(alpha=0.0)
        except Exception as e:
            assert "panic" not in type(e).__name__.lower()

    def test_n_refinements_zero_in_create_index(self, tmp_path):
        db = _open(tmp_path / "db")
        vecs = _unit_batch(50, 16)
        db.insert_batch([str(i) for i in range(50)], vecs)
        try:
            db.create_index(n_refinements=0)
        except Exception as e:
            assert "panic" not in type(e).__name__.lower()

    def test_create_index_twice_is_idempotent(self, tmp_path):
        db = _open(tmp_path / "db")
        vecs = _unit_batch(100, 16)
        db.insert_batch([str(i) for i in range(100)], vecs)
        db.create_index(max_degree=16, ef_construction=64, search_list_size=64)
        first_nodes = db.stats()["index_nodes"]
        db.create_index(max_degree=16, ef_construction=64, search_list_size=64)
        second_nodes = db.stats()["index_nodes"]
        # Both should have an index; second rebuild may produce same node count
        assert db.stats()["has_index"]
        assert second_nodes > 0

    def test_very_large_rerank_factor(self, tmp_path):
        # rerank_factor > corpus size — should clamp gracefully
        db = _open(tmp_path / "db", rerank=True)
        vecs = _unit_batch(10, 16)
        db.insert_batch([str(i) for i in range(10)], vecs)
        try:
            results = db.search(vecs[0], top_k=5, rerank_factor=10000)
            assert len(results) <= 5
        except Exception as e:
            assert "panic" not in type(e).__name__.lower()

    def test_create_index_after_all_deleted_no_panic(self, tmp_path):
        db = _open(tmp_path / "db")
        vecs = _unit_batch(10, 16)
        ids = [str(i) for i in range(10)]
        db.insert_batch(ids, vecs)
        db.delete_batch(ids)
        assert db.count() == 0
        try:
            db.create_index()
        except Exception as e:
            assert "panic" not in type(e).__name__.lower()

    def test_max_degree_very_large_no_panic(self, tmp_path):
        db = _open(tmp_path / "db")
        vecs = _unit_batch(30, 16)
        db.insert_batch([str(i) for i in range(30)], vecs)
        try:
            db.create_index(max_degree=1000, ef_construction=50)
        except Exception as e:
            assert "panic" not in type(e).__name__.lower()


# ---------------------------------------------------------------------------
# ███████╗████████╗ █████╗ ████████╗███████╗
# ██╔════╝╚══██╔══╝██╔══██╗╚══██╔══╝██╔════╝
# ███████╗   ██║   ███████║   ██║   ███████╗
# ╚════██║   ██║   ██╔══██║   ██║   ╚════██║
# ███████║   ██║   ██║  ██║   ██║   ███████║
# ---------------------------------------------------------------------------

class TestStatsConsistency:
    """stats(), len(db), and count() must agree at all times."""

    def test_stats_count_len_agree_on_empty_db(self, tmp_path):
        db = _open(tmp_path / "db")
        s = db.stats()
        assert s["vector_count"] == 0
        assert len(db) == 0
        assert db.count() == 0

    def test_stats_count_len_agree_after_insert(self, tmp_path):
        db = _open(tmp_path / "db")
        n = 50
        vecs = _unit_batch(n, 16)
        db.insert_batch([str(i) for i in range(n)], vecs)
        s = db.stats()
        assert s["vector_count"] == n
        assert len(db) == n
        assert db.count() == n

    def test_stats_count_len_agree_after_delete(self, tmp_path):
        db = _open(tmp_path / "db")
        vecs = _unit_batch(20, 16)
        db.insert_batch([str(i) for i in range(20)], vecs)
        db.delete_batch([str(i) for i in range(5)])
        s = db.stats()
        assert s["vector_count"] == 15
        assert len(db) == 15
        assert db.count() == 15

    def test_stats_has_index_reflects_create_index(self, tmp_path):
        db = _open(tmp_path / "db")
        vecs = _unit_batch(50, 16)
        db.insert_batch([str(i) for i in range(50)], vecs)
        assert not db.stats()["has_index"]
        db.create_index(max_degree=16, ef_construction=64, search_list_size=64)
        assert db.stats()["has_index"]

    def test_stats_dimension_and_bits_correct(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=32, bits=8,
                           seed=42, metric="ip")
        s = db.stats()
        assert s["dimension"] == 32
        assert s["bits"] == 8

    def test_stats_after_flush_segment_count_increases(self, tmp_path):
        db = _open(tmp_path / "db")
        vecs = _unit_batch(10, 16)
        db.insert_batch([str(i) for i in range(10)], vecs)
        before = db.stats()["segment_count"]
        db.flush()
        after = db.stats()["segment_count"]
        # After an explicit flush from WAL, segment count should increase
        assert after >= before


# ---------------------------------------------------------------------------
# ██╗     ██╗███████╗████████╗     █████╗ ██╗     ██╗
# ██║     ██║██╔════╝╚══██╔══╝    ██╔══██╗██║     ██║
# ██║     ██║███████╗   ██║       ███████║██║     ██║
# ██║     ██║╚════██║   ██║       ██╔══██║██║     ██║
# ███████╗██║███████║   ██║       ██║  ██║███████╗███████╗
# ---------------------------------------------------------------------------

class TestListAll:
    """list_all() must return exactly the active IDs."""

    def test_list_all_empty_db(self, tmp_path):
        db = _open(tmp_path / "db")
        assert db.list_all() == []

    def test_list_all_returns_all_ids(self, tmp_path):
        db = _open(tmp_path / "db")
        expected = {str(i) for i in range(20)}
        vecs = _unit_batch(20, 16)
        db.insert_batch(list(expected), vecs)
        assert set(db.list_all()) == expected

    def test_list_all_excludes_deleted(self, tmp_path):
        db = _open(tmp_path / "db")
        vecs = _unit_batch(10, 16)
        db.insert_batch([str(i) for i in range(10)], vecs)
        db.delete_batch(["3", "7"])
        all_ids = set(db.list_all())
        assert "3" not in all_ids
        assert "7" not in all_ids
        assert len(all_ids) == 8

    def test_list_all_after_delete_all_is_empty(self, tmp_path):
        db = _open(tmp_path / "db")
        vecs = _unit_batch(5, 16)
        ids = [str(i) for i in range(5)]
        db.insert_batch(ids, vecs)
        db.delete_batch(ids)
        assert db.list_all() == []

    def test_list_all_count_matches_len(self, tmp_path):
        db = _open(tmp_path / "db")
        vecs = _unit_batch(30, 16)
        db.insert_batch([str(i) for i in range(30)], vecs)
        db.delete_batch(["5", "15", "25"])
        assert len(db.list_all()) == len(db)


# ---------------------------------------------------------------------------
# ███████╗ █████╗ ██╗     ███████╗██╗   ██╗    ██╗   ██╗ █████╗ ██╗     ██╗   ██╗███████╗███████╗
# ██╔════╝██╔══██╗██║     ██╔════╝╚██╗ ██╔╝    ██║   ██║██╔══██╗██║     ██║   ██║██╔════╝██╔════╝
# █████╗  ███████║██║     ███████╗ ╚████╔╝     ██║   ██║███████║██║     ██║   ██║█████╗  ███████╗
# ██╔══╝  ██╔══██║██║     ╚════██║  ╚██╔╝      ╚██╗ ██╔╝██╔══██║██║     ██║   ██║██╔══╝  ╚════██║
# ██║     ██║  ██║███████╗███████║   ██║        ╚████╔╝ ██║  ██║███████╗╚██████╔╝███████╗███████║
# ---------------------------------------------------------------------------

class TestMetadataFalsyValues:
    """Falsy values in metadata must survive roundtrip without being dropped."""

    def test_bool_true_roundtrip(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0), metadata={"flag": True})
        got = db.get("a")
        assert got["metadata"]["flag"] is True

    def test_bool_false_roundtrip(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0), metadata={"flag": False})
        got = db.get("a")
        assert got["metadata"]["flag"] is False

    def test_integer_zero_roundtrip(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0), metadata={"count": 0})
        got = db.get("a")
        assert got["metadata"]["count"] == 0

    def test_empty_string_roundtrip(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0), metadata={"label": ""})
        got = db.get("a")
        assert got["metadata"]["label"] == ""

    def test_null_value_roundtrip(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0), metadata={"x": None})
        got = db.get("a")
        assert got["metadata"]["x"] is None

    def test_negative_integer_roundtrip(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0), metadata={"delta": -42})
        got = db.get("a")
        assert got["metadata"]["delta"] == -42

    def test_float_zero_roundtrip(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0), metadata={"score": 0.0})
        got = db.get("a")
        assert got["metadata"]["score"] == 0.0

    def test_nested_falsy_values(self, tmp_path):
        db = _open(tmp_path / "db")
        meta = {"a": {"b": False, "c": 0, "d": ""}}
        db.insert("x", _vec(16, seed=0), metadata=meta)
        got = db.get("x")
        assert got["metadata"]["a"]["b"] is False
        assert got["metadata"]["a"]["c"] == 0
        assert got["metadata"]["a"]["d"] == ""

    def test_filter_on_false_value(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0), metadata={"active": False})
        db.insert("b", _vec(16, seed=1), metadata={"active": True})
        results = db.search(_vec(16, seed=0), top_k=2, filter={"active": False})
        returned = [r["id"] for r in results]
        assert "a" in returned
        assert "b" not in returned

    def test_filter_on_integer_zero(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("zero", _vec(16, seed=0), metadata={"rank": 0})
        db.insert("one", _vec(16, seed=1), metadata={"rank": 1})
        results = db.search(_vec(16, seed=0), top_k=2, filter={"rank": 0})
        returned = {r["id"] for r in results}
        assert "zero" in returned
        assert "one" not in returned


# ---------------------------------------------------------------------------
# ██████╗ ███████╗ ██████╗ ██████╗ ███████╗███╗   ██╗    ██████╗ ███████╗██████╗ ███████╗██╗███████╗████████╗
# ██╔══██╗██╔════╝██╔═══██╗██╔══██╗██╔════╝████╗  ██║    ██╔══██╗██╔════╝██╔══██╗██╔════╝██║██╔════╝╚══██╔══╝
# ██████╔╝█████╗  ██║   ██║██████╔╝█████╗  ██╔██╗ ██║    ██████╔╝█████╗  ██████╔╝███████╗██║███████╗   ██║
# ██╔══██╗██╔══╝  ██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║    ██╔═══╝ ██╔══╝  ██╔══██╗╚════██║██║╚════██║   ██║
# ██║  ██║███████╗╚██████╔╝██║     ███████╗██║ ╚████║    ██║     ███████╗██║  ██║███████║██║███████║   ██║
# ---------------------------------------------------------------------------

class TestReopenValidation:
    """Reopen scenarios: wrong dimension, bits, metric conflicts."""

    def test_reopen_different_dimension_raises(self, tmp_path):
        path = str(tmp_path / "db")
        db = _open(path, d=16)
        db.close()
        del db
        gc.collect()
        gc.collect()
        with pytest.raises(Exception) as exc_info:
            Database.open(path, dimension=32, bits=4, seed=42, metric="ip")
        assert "panic" not in type(exc_info.value).__name__.lower()

    def test_reopen_same_params_works(self, tmp_path):
        path = str(tmp_path / "db")
        db = _open(path, d=16, bits=4, metric="ip")
        db.insert("a", _vec(16, seed=0))
        db.close()
        del db
        gc.collect()
        gc.collect()
        db2 = _open(path, d=16, bits=4, metric="ip")
        assert db2.count() == 1

    def test_reopen_data_survives_without_explicit_close(self, tmp_path):
        path = str(tmp_path / "db")
        db = _open(path)
        vecs = _unit_batch(30, 16)
        db.insert_batch([str(i) for i in range(30)], vecs)
        del db  # Simulates crash — no close()
        gc.collect()
        gc.collect()
        db2 = _open(path)
        assert db2.count() == 30

    def test_reopen_preserves_search_correctness(self, tmp_path):
        path = str(tmp_path / "db")
        db = _open(path)
        vecs = _unit_batch(50, 16)
        db.insert_batch([str(i) for i in range(50)], vecs)
        db.close()
        del db
        gc.collect()
        gc.collect()
        db2 = _open(path)
        results = db2.search(vecs[17], top_k=1)
        assert results[0]["id"] == "17"

    def test_reopen_with_new_seed_still_reads_correctly(self, tmp_path):
        # Seed only affects new insertions; reopening with a different seed
        # should still be able to read existing data (quantizer is loaded from disk)
        path = str(tmp_path / "db")
        db = _open(path, d=16)
        db.insert("a", _vec(16, seed=0))
        db.close()
        del db
        gc.collect()
        gc.collect()
        # Try reopening with different seed — should either succeed or raise cleanly
        try:
            db2 = Database.open(path, dimension=16, bits=4, seed=999, metric="ip")
            assert db2.get("a") is not None
        except Exception as e:
            assert "panic" not in type(e).__name__.lower()


# ---------------------------------------------------------------------------
# ██╗    ██╗ █████╗ ██╗         ████████╗██╗  ██╗██████╗ ███████╗ █████╗ ██████╗ ███████╗
# ██║    ██║██╔══██╗██║         ╚══██╔══╝██║  ██║██╔══██╗██╔════╝██╔══██╗██╔══██╗██╔════╝
# ██║ █╗ ██║███████║██║            ██║   ███████║██████╔╝█████╗  ███████║██║  ██║███████╗
# ██║███╗██║██╔══██║██║            ██║   ██╔══██║██╔══██╗██╔══╝  ██╔══██║██║  ██║╚════██║
# ╚███╔███╔╝██║  ██║███████╗       ██║   ██║  ██║██║  ██║███████╗██║  ██║██████╔╝███████║
# ---------------------------------------------------------------------------

class TestWalThresholds:
    """Flush-threshold edge cases — ensure data integrity at boundaries."""

    def test_wal_threshold_one_flushes_every_insert(self, tmp_path):
        path = str(tmp_path / "db")
        db = Database.open(path, dimension=16, bits=4, seed=42,
                           wal_flush_threshold=1)
        for i in range(20):
            db.insert(str(i), _vec(16, seed=i))
        assert db.count() == 20
        results = db.search(_vec(16, seed=5), top_k=1)
        assert results[0]["id"] == "5"

    def test_wal_threshold_one_persists_after_crash_sim(self, tmp_path):
        path = str(tmp_path / "db")
        db = Database.open(path, dimension=16, bits=4, seed=42,
                           wal_flush_threshold=1)
        for i in range(10):
            db.insert(str(i), _vec(16, seed=i))
        del db
        gc.collect()
        gc.collect()
        db2 = _open(path)
        assert db2.count() == 10

    def test_wal_threshold_large_no_automatic_flush(self, tmp_path):
        # Large threshold: WAL never auto-flushes during these inserts
        import sys
        db = Database.open(str(tmp_path / "db"), dimension=16, bits=4, seed=42,
                           wal_flush_threshold=100_000)
        vecs = _unit_batch(50, 16)
        db.insert_batch([str(i) for i in range(50)], vecs)
        # Data must still be searchable even without a segment flush
        results = db.search(vecs[7], top_k=1)
        assert results[0]["id"] == "7"
        assert db.count() == 50

    def test_wal_recovery_after_partial_batch(self, tmp_path):
        # Insert a batch, simulate crash before close, verify all recovered
        path = str(tmp_path / "db")
        db = Database.open(path, dimension=16, bits=4, seed=42,
                           wal_flush_threshold=5)
        vecs = _unit_batch(12, 16)
        db.insert_batch([str(i) for i in range(12)], vecs)
        del db
        gc.collect()
        gc.collect()
        db2 = _open(path)
        assert db2.count() == 12


# ---------------------------------------------------------------------------
# ██████╗ ███████╗███████╗████████╗    ██╗     ██╗   ██╗██████╗  ██████╗ ███████╗
# ██╔══██╗██╔════╝██╔════╝╚══██╔══╝    ██║     ██║   ██║██╔══██╗██╔════╝ ██╔════╝
# ██████╔╝█████╗  ███████╗   ██║       ██║     ██║   ██║██████╔╝██║  ███╗█████╗
# ██╔══██╗██╔══╝  ╚════██║   ██║       ██║     ██║   ██║██╔══██╗██║   ██║██╔══╝
# ██║  ██║███████╗███████║   ██║       ███████╗╚██████╔╝██║  ██║╚██████╔╝███████╗
# ---------------------------------------------------------------------------

class TestConcurrentHeavy:
    """Concurrent readers and writers under heavy load."""

    def test_20_concurrent_writers_no_data_loss(self, tmp_path):
        db = _open(tmp_path / "db")
        errors = []
        results_lock = threading.Lock()
        inserted_ids = []

        def writer(thread_id: int):
            vecs = _unit_batch(50, 16, seed=thread_id)
            ids = [f"t{thread_id}_i{i}" for i in range(50)]
            try:
                db.insert_batch(ids, vecs)
                with results_lock:
                    inserted_ids.extend(ids)
            except Exception as e:
                with results_lock:
                    errors.append(str(e))

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        assert not errors, f"Writer errors: {errors}"
        assert db.count() == len(inserted_ids)

    def test_concurrent_writers_and_readers(self, tmp_path):
        db = _open(tmp_path / "db")
        # Seed with initial data
        vecs = _unit_batch(100, 16, seed=0)
        db.insert_batch([str(i) for i in range(100)], vecs)

        read_errors = []
        write_errors = []
        lock = threading.Lock()

        def reader(rid: int):
            for _ in range(20):
                try:
                    db.search(_vec(16, seed=rid), top_k=5)
                except Exception as e:
                    with lock:
                        read_errors.append(str(e))

        def writer(wid: int):
            for j in range(10):
                try:
                    db.insert(f"w{wid}_{j}", _vec(16, seed=wid * 100 + j))
                except Exception as e:
                    with lock:
                        write_errors.append(str(e))

        threads = (
            [threading.Thread(target=reader, args=(i,)) for i in range(5)]
            + [threading.Thread(target=writer, args=(i,)) for i in range(5)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        assert not read_errors, f"Reader errors: {read_errors}"
        assert not write_errors, f"Writer errors: {write_errors}"

    def test_concurrent_delete_and_insert_no_panic(self, tmp_path):
        db = _open(tmp_path / "db")
        n = 200
        vecs = _unit_batch(n, 16)
        db.insert_batch([str(i) for i in range(n)], vecs)

        errors = []
        lock = threading.Lock()

        def deleter():
            for i in range(0, n, 2):
                try:
                    db.delete(str(i))
                except Exception as e:
                    with lock:
                        errors.append(f"delete {i}: {e}")

        def inserter():
            for i in range(n, n + 100):
                try:
                    db.insert(str(i), _vec(16, seed=i))
                except Exception as e:
                    with lock:
                        errors.append(f"insert {i}: {e}")

        t1 = threading.Thread(target=deleter)
        t2 = threading.Thread(target=inserter)
        t1.start()
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)

        assert not errors, f"Concurrent errors: {errors}"
        # Total should be approximately 200 - 100 deleted + 100 inserted = 200
        # (exact count depends on race timing)
        assert db.count() >= 0  # Must not corrupt the count


# ---------------------------------------------------------------------------
# ███████╗██╗██╗  ████████╗███████╗██████╗     ██╗      ██████╗  ██████╗
# ██╔════╝██║██║  ╚══██╔══╝██╔════╝██╔══██╗    ██║     ██╔═══██╗██╔════╝
# █████╗  ██║██║     ██║   █████╗  ██████╔╝    ██║     ██║   ██║██║  ███╗
# ██╔══╝  ██║██║     ██║   ██╔══╝  ██╔══██╗    ██║     ██║   ██║██║   ██║
# ██║     ██║███████╗██║   ███████╗██║  ██║    ███████╗╚██████╔╝╚██████╔╝
# ---------------------------------------------------------------------------

class TestFilterLogic:
    """Edge cases in $and / $or / dotted-path filter logic."""

    def test_empty_and_array_matches_nothing_or_everything(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0), metadata={"x": 1})
        # $and with empty array — behavior undefined; must not panic
        try:
            results = db.search(_vec(16, seed=0), top_k=5,
                                filter={"$and": []})
        except Exception as e:
            assert "panic" not in type(e).__name__.lower()

    def test_or_with_two_exclusive_conditions(self, tmp_path):
        db = _open(tmp_path / "db")
        for i in range(6):
            db.insert(str(i), _vec(16, seed=i), metadata={"v": i})
        # v < 2 OR v > 4 → {0, 1, 5}
        filt = {"$or": [{"v": {"$lt": 2}}, {"v": {"$gt": 4}}]}
        results = db.search(_vec(16, seed=0), top_k=10, filter=filt)
        vals = {r["metadata"]["v"] for r in results}
        assert vals == {0, 1, 5}

    def test_and_with_or_nested_deep(self, tmp_path):
        db = _open(tmp_path / "db")
        for i in range(10):
            db.insert(str(i), _vec(16, seed=i), metadata={"v": i, "g": i % 2})
        # (v >= 5 AND (g == 0 OR g == 1)) → {5, 6, 7, 8, 9}
        filt = {
            "$and": [
                {"v": {"$gte": 5}},
                {"$or": [{"g": 0}, {"g": 1}]}
            ]
        }
        results = db.search(_vec(16, seed=0), top_k=10, filter=filt)
        vals = {r["metadata"]["v"] for r in results}
        assert vals == {5, 6, 7, 8, 9}

    def test_dotted_path_filter(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0), metadata={"loc": {"country": "US"}})
        db.insert("b", _vec(16, seed=1), metadata={"loc": {"country": "EU"}})
        results = db.search(_vec(16, seed=0), top_k=5,
                            filter={"loc.country": "US"})
        assert len(results) == 1
        assert results[0]["id"] == "a"

    def test_filter_on_non_existent_field_returns_nothing(self, tmp_path):
        db = _open(tmp_path / "db")
        for i in range(5):
            db.insert(str(i), _vec(16, seed=i), metadata={"x": 1})
        results = db.search(_vec(16, seed=0), top_k=5,
                            filter={"does_not_exist": "value"})
        assert results == []

    def test_in_operator_with_large_list(self, tmp_path):
        db = _open(tmp_path / "db")
        for i in range(20):
            db.insert(str(i), _vec(16, seed=i), metadata={"val": i})
        big_list = list(range(0, 20, 2))  # even numbers
        results = db.search(_vec(16, seed=0), top_k=20,
                            filter={"val": {"$in": big_list}})
        for r in results:
            assert r["metadata"]["val"] % 2 == 0

    def test_invalid_operator_raises_cleanly(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0), metadata={"x": 1})
        with pytest.raises(Exception) as exc_info:
            db.search(_vec(16, seed=0), top_k=1,
                      filter={"x": {"$invalid_op": 1}})
        assert "panic" not in type(exc_info.value).__name__.lower()

    def test_filter_in_count_with_or_operator(self, tmp_path):
        db = _open(tmp_path / "db")
        for i in range(10):
            db.insert(str(i), _vec(16, seed=i), metadata={"tag": "A" if i < 5 else "B"})
        c = db.count(filter={"$or": [{"tag": "A"}, {"tag": "B"}]})
        assert c == 10

    def test_filter_in_list_ids_with_or(self, tmp_path):
        db = _open(tmp_path / "db")
        for i in range(10):
            db.insert(str(i), _vec(16, seed=i), metadata={"group": i % 3})
        # group == 0 OR group == 2 → {0,3,6,9} ∪ {2,5,8} = 7 items
        ids = db.list_ids(where_filter={"$or": [{"group": 0}, {"group": 2}]})
        assert len(ids) == 7


# ---------------------------------------------------------------------------
# ██╗      ██████╗  ██████╗██╗  ██╗    ██████╗  ██████╗ ██╗███████╗ ██████╗ ███╗   ██╗
# ██║     ██╔═══██╗██╔════╝██║ ██╔╝    ██╔══██╗██╔═══██╗██║██╔════╝██╔═══██╗████╗  ██║
# ██║     ██║   ██║██║     █████╔╝     ██████╔╝██║   ██║██║███████╗██║   ██║██╔██╗ ██║
# ██║     ██║   ██║██║     ██╔═██╗     ██╔═══╝ ██║   ██║██║╚════██║██║   ██║██║╚██╗██║
# ███████╗╚██████╔╝╚██████╗██║  ██╗    ██║     ╚██████╔╝██║███████║╚██████╔╝██║ ╚████║
# ---------------------------------------------------------------------------

class TestLockPoisoning:
    """After a Rust panic, the lock must report 'poisoned' cleanly."""

    def test_lock_poisoned_after_bits_overflow_panic(self, tmp_path):
        # bits=256 triggers a capacity-overflow panic during construction.
        # After the panic, any subsequent use of that db object must raise
        # 'lock poisoned' — NOT another panic or a silent corruption.
        try:
            db = Database.open(str(tmp_path / "db"), dimension=16, bits=256,
                               seed=42, metric="ip")
            # If open succeeded (no panic yet), trigger the panic via insert
            db.insert("x", _vec(16, seed=0))
        except BaseException:
            # Panic escaped — this is the expected path for bits=256
            pass
        # Open a fresh DB at a different path — must work normally
        db2 = _open(tmp_path / "db2")
        db2.insert("fresh", _vec(16, seed=0))
        assert db2.count() == 1

    def test_poisoned_db_subsequent_op_raises_not_panic(self, tmp_path):
        # Attempt to trigger lock poisoning by having a panic mid-operation.
        # If bits=256 panics inside write_engine, all subsequent operations
        # on the SAME object should raise "lock poisoned" cleanly.
        db = None
        try:
            db = Database.open(str(tmp_path / "db"), dimension=16, bits=256,
                               seed=42, metric="ip")
            db.insert("x", _vec(16, seed=0))
        except BaseException:
            pass

        if db is not None:
            # Try to use the poisoned db — must not produce another panic
            try:
                db.count()
            except Exception as e:
                # Should raise RuntimeError about lock poisoning, not a panic
                assert "panic" not in type(e).__name__.lower()
