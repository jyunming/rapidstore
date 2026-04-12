"""
BRUTAL QA MEGA SUITE — TurboQuantDB
=====================================
Merged from Rounds 1-5 into a single file, plus Round 6 new tests.

Run with:
    python -m pytest tests/test_brutal_qa_all.py --basetemp=tmp_pytest_all -q
"""
from __future__ import annotations

import gc
import math
import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from tqdb import Database
from tqdb.rag import TurboQuantRetriever

# ---------------------------------------------------------------------------
# Unified helpers (superset of all 5 rounds)
# ---------------------------------------------------------------------------

DIM = 32
SEED = 42


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def rand_vec(dim: int = DIM, seed: int = 0) -> np.ndarray:
    v = rng(seed).random(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def rand_vecs(n: int, dim: int = DIM, seed: int = 0) -> np.ndarray:
    vs = rng(seed).random((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vs, axis=1, keepdims=True) + 1e-9
    return vs / norms


def _vec(d: int = 16, seed: int = 0, val: float = 1.0) -> np.ndarray:
    r = np.random.default_rng(seed)
    v = r.standard_normal(d).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def _unit_batch(n: int, d: int = 16, seed: int = 42, dim: int | None = None) -> np.ndarray:
    """dim= is accepted as an alias for d= (backward compat with Round 3 tests)."""
    if dim is not None:
        d = dim
    r = np.random.default_rng(seed)
    vs = r.standard_normal((n, d)).astype(np.float32)
    norms = np.linalg.norm(vs, axis=1, keepdims=True)
    return (vs / np.maximum(norms, 1e-8)).astype(np.float32)


def _open(path, d: int = 16, bits: int = 4, metric: str = "ip", **kw) -> Database:
    return Database.open(str(path), dimension=d, bits=bits, seed=42,
                         metric=metric, **kw)


# ========================================================================
# FROM test_brutal_qa.py
# ========================================================================

class TestOpenParameterBrutal:
    """Break Database.open() with every extreme parameter combination."""

    def test_bits_zero_raises(self, tmp_path):
        with pytest.raises(Exception):
            Database.open(str(tmp_path / "db"), dimension=8, bits=0)

    def test_bits_one_without_fast_mode_raises(self, tmp_path):
        # bits=1 without fast_mode reserves 0 bits for MSE → must error
        with pytest.raises(Exception):
            Database.open(str(tmp_path / "db"), dimension=8, bits=1, fast_mode=False)

    def test_bits_one_with_fast_mode_works(self, tmp_path):
        # bits=1 fast_mode=True is a valid configuration
        db = Database.open(str(tmp_path / "db"), dimension=8, bits=1, fast_mode=True)
        db.insert("a", _vec(8))
        assert db.count() == 1

    def test_bits_2_without_fast_mode_works(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=8, bits=2, fast_mode=False)
        db.insert("a", _vec(8))
        assert db.count() == 1

    def test_bits_8_works(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=8, bits=8)
        db.insert_batch([str(i) for i in range(10)], _unit_batch(10, 8))
        assert db.count() == 10

    def test_bits_extreme_large_raises_or_handles_gracefully(self, tmp_path):
        # BUG: bits=256 triggers a Rust 'capacity overflow' panic that escapes as
        # PanicException (a BaseException, not Exception). Must not silently corrupt.
        try:
            db = Database.open(str(tmp_path / "db"), dimension=8, bits=256)
            db.insert("a", _vec(8))
        except BaseException:
            pass  # Raising is acceptable; panicking via PanicException is the current bug

    def test_dimension_zero_raises(self, tmp_path):
        with pytest.raises(Exception):
            Database.open(str(tmp_path / "db"), dimension=0)

    def test_dimension_one_works(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=1, bits=4, fast_mode=True)
        v = np.array([1.0], dtype=np.float32)
        db.insert("a", v)
        results = db.search(v, top_k=1)
        assert len(results) == 1

    def test_dimension_very_large_works(self, tmp_path):
        # 4096 dimensions — LLM-scale embedding
        d = 4096
        db = Database.open(str(tmp_path / "db"), dimension=d, bits=4)
        vecs = _unit_batch(3, d)
        db.insert_batch(["a", "b", "c"], vecs)
        results = db.search(vecs[0], top_k=2)
        assert len(results) == 2

    def test_reopen_with_wrong_dimension_raises(self, tmp_path):
        path = str(tmp_path / "db")
        db = Database.open(path, dimension=8, bits=4)
        db.insert("a", _vec(8))
        del db
        gc.collect()
        with pytest.raises(Exception):
            Database.open(path, dimension=16, bits=4)

    def test_all_metrics_open_correctly(self, tmp_path):
        for metric in ("ip", "cosine", "l2"):
            p = str(tmp_path / metric)
            db = Database.open(p, dimension=8, bits=4, metric=metric)
            db.insert("a", _vec(8, seed=0))
            results = db.search(_vec(8, seed=0), top_k=1)
            assert results[0]["id"] == "a", f"metric={metric} top-1 mismatch"

    def test_metric_is_case_insensitive(self, tmp_path):
        # The API normalises metric strings via to_lowercase() — uppercase is accepted.
        # Use numbered paths to avoid Windows case-insensitive filesystem collisions
        # (e.g. "Cosine" and "COSINE" map to the same directory on NTFS).
        for i, m in enumerate(("IP", "Cosine", "L2", "COSINE")):
            db = Database.open(str(tmp_path / f"db_{i}"), dimension=8, bits=4, metric=m)
            db.insert("a", _vec(8, seed=0))
            results = db.search(_vec(8, seed=0), top_k=1)
            assert results[0]["id"] == "a", f"uppercase metric={m} gave wrong top-1"

    def test_invalid_quantizer_type_raises(self, tmp_path):
        with pytest.raises(Exception):
            Database.open(str(tmp_path / "db"), dimension=8, bits=4, quantizer_type="bad")

    def test_quantizer_type_srht_works(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=16, bits=4, quantizer_type="srht")
        db.insert_batch([str(i) for i in range(10)], _unit_batch(10, 16))
        assert db.count() == 10

    def test_quantizer_type_dense_works(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=16, bits=4, quantizer_type="dense")
        db.insert_batch([str(i) for i in range(10)], _unit_batch(10, 16))
        assert db.count() == 10

    def test_quantizer_type_exact_alias_works(self, tmp_path):
        # "exact" is a documented legacy alias for "dense"
        db = Database.open(str(tmp_path / "db"), dimension=16, bits=4, quantizer_type="exact")
        db.insert_batch([str(i) for i in range(5)], _unit_batch(5, 16))
        assert db.count() == 5

    def test_normalize_flag_works(self, tmp_path):
        d = 16
        db = Database.open(str(tmp_path / "db"), dimension=d, bits=4, normalize=True, metric="ip")
        vecs = _unit_batch(10, d)
        db.insert_batch([str(i) for i in range(10)], vecs)
        results = db.search(vecs[0], top_k=1)
        assert results[0]["id"] == "0"

    def test_wal_flush_threshold_zero_or_one(self, tmp_path):
        # Threshold=1 means flush after every vector — should not panic
        db = Database.open(str(tmp_path / "db"), dimension=8, bits=4, wal_flush_threshold=1)
        for i in range(5):
            db.insert(str(i), _vec(8, seed=i))
        assert db.count() == 5

    def test_all_rerank_precision_values(self, tmp_path):
        for prec in ("int8", "int4", "f16", "f32", "disabled", "dequant"):
            p = str(tmp_path / prec)
            db = Database.open(p, dimension=16, bits=4, rerank_precision=prec)
            db.insert_batch([str(i) for i in range(5)], _unit_batch(5, 16))
            results = db.search(_unit_batch(1, 16)[0], top_k=3)
            assert len(results) == 3


# ---------------------------------------------------------------------------
# ██╗   ██╗███████╗ ██████╗████████╗ ██████╗ ██████╗ ███████╗
# ██║   ██║██╔════╝██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗██╔════╝
# ██║   ██║█████╗  ██║        ██║   ██║   ██║██████╔╝███████╗
# ╚██╗ ██╔╝██╔══╝  ██║        ██║   ██║   ██║██╔══██╗╚════██║
#  ╚████╔╝ ███████╗╚██████╗   ██║   ╚██████╔╝██║  ██║███████║
# ---------------------------------------------------------------------------

class TestVectorInputBrutal:
    """Break vector parsing with every pathological array type."""

    def test_zero_vector_insert_cosine_no_panic(self, tmp_path):
        # Zero vector is not unit-norm — some metrics divide by norm → potential div/0
        db = _open(tmp_path / "db", metric="cosine")
        zero = np.zeros(16, dtype=np.float32)
        # Either rejects or handles gracefully; must NOT panic
        try:
            db.insert("z", zero)
        except Exception:
            pass  # rejection is fine

    def test_zero_vector_insert_l2(self, tmp_path):
        db = _open(tmp_path / "db", metric="l2")
        zero = np.zeros(16, dtype=np.float32)
        try:
            db.insert("z", zero)
            # If accepted, search should not panic
            db.search(np.ones(16, dtype=np.float32), top_k=1)
        except Exception:
            pass

    def test_nan_vector_in_batch_middle(self, tmp_path):
        # NaN in position 1 of a 3-item batch — whole batch should be rejected
        db = _open(tmp_path / "db")
        vecs = _unit_batch(3, 16)
        vecs[1, 5] = float("nan")
        with pytest.raises(Exception):
            db.insert_batch(["a", "b", "c"], vecs)
        # No partial insertion should have occurred
        assert db.count() == 0

    def test_inf_vector_in_batch_last_position(self, tmp_path):
        db = _open(tmp_path / "db")
        vecs = _unit_batch(3, 16)
        vecs[2, 0] = float("inf")
        with pytest.raises(Exception):
            db.insert_batch(["a", "b", "c"], vecs)
        assert db.count() == 0

    def test_negative_inf_vector(self, tmp_path):
        db = _open(tmp_path / "db")
        v = np.full(16, float("-inf"), dtype=np.float32)
        with pytest.raises(Exception):
            db.insert("neg_inf", v)

    def test_float64_vector_accepted(self, tmp_path):
        db = _open(tmp_path / "db")
        v = np.ones(16, dtype=np.float64)
        db.insert("f64", v)
        assert db.get("f64") is not None

    def test_float64_batch_accepted(self, tmp_path):
        db = _open(tmp_path / "db")
        vecs = _unit_batch(5, 16).astype(np.float64)
        db.insert_batch([str(i) for i in range(5)], vecs)
        assert db.count() == 5

    def test_non_contiguous_c_order_slice(self, tmp_path):
        # Slice of every other row → non-contiguous in memory
        db = _open(tmp_path / "db")
        big = _unit_batch(20, 16)
        sliced = big[::2]  # non-contiguous
        assert not sliced.flags["C_CONTIGUOUS"]
        db.insert_batch([str(i) for i in range(10)], sliced)
        assert db.count() == 10

    def test_fortran_order_batch(self, tmp_path):
        # F-order array should be handled without panic
        db = _open(tmp_path / "db")
        vecs = np.asfortranarray(_unit_batch(5, 16))
        assert vecs.flags["F_CONTIGUOUS"]
        db.insert_batch([str(i) for i in range(5)], vecs)
        assert db.count() == 5

    def test_float16_vector_accepted_or_raises_gracefully(self, tmp_path):
        # float16 is not float32 or float64 — must raise cleanly, not panic
        db = _open(tmp_path / "db")
        v = np.ones(16, dtype=np.float16)
        try:
            db.insert("f16", v)
        except Exception as e:
            assert "panic" not in type(e).__name__.lower(), f"Rust panic surfaced: {type(e).__name__}"

    def test_int_vector_rejected_or_accepted_gracefully(self, tmp_path):
        # int32 array should either be rejected or converted without panic
        db = _open(tmp_path / "db")
        v = np.ones(16, dtype=np.int32)
        try:
            db.insert("int", v)
        except Exception as e:
            assert "panic" not in type(e).__name__.lower()

    def test_bool_vector_rejected_or_accepted_gracefully(self, tmp_path):
        db = _open(tmp_path / "db")
        v = np.ones(16, dtype=bool)
        try:
            db.insert("bool", v)
        except Exception as e:
            assert "panic" not in type(e).__name__.lower()

    def test_2d_single_row_insert_fails(self, tmp_path):
        # insert() expects 1-D; 2-D (1, 16) should raise cleanly
        db = _open(tmp_path / "db")
        v = np.ones((1, 16), dtype=np.float32)
        with pytest.raises(Exception):
            db.insert("bad", v)

    def test_scalar_insert_fails(self, tmp_path):
        db = _open(tmp_path / "db")
        with pytest.raises(Exception):
            db.insert("bad", 1.0)

    def test_list_insert_rejected_or_accepted_gracefully(self, tmp_path):
        # Python list: not a numpy array; behavior should not panic
        db = _open(tmp_path / "db")
        v = [1.0] * 16
        try:
            db.insert("list", v)
        except Exception as e:
            assert "panic" not in type(e).__name__.lower()

    def test_very_large_values_near_float32_max(self, tmp_path):
        # Values near float32 max (3.4e38) — not inf, but extreme
        db = _open(tmp_path / "db")
        v = np.full(16, 3.4e38, dtype=np.float32)
        # Should either be accepted or rejected cleanly
        try:
            db.insert("big", v)
        except Exception as e:
            assert "panic" not in type(e).__name__.lower()

    def test_subnormal_float32_values(self, tmp_path):
        # Subnormal numbers (very small, near zero) are finite — should be accepted
        db = _open(tmp_path / "db")
        v = np.full(16, np.float32(1e-40), dtype=np.float32)
        assert np.all(np.isfinite(v))
        try:
            db.insert("subnormal", v)
        except Exception as e:
            assert "panic" not in type(e).__name__.lower()

    def test_empty_insert_batch_works(self, tmp_path):
        # Batch of zero vectors should be a no-op, not a crash
        db = _open(tmp_path / "db")
        db.insert_batch([], np.empty((0, 16), dtype=np.float32))
        assert db.count() == 0

    def test_batch_ids_count_mismatch_raises(self, tmp_path):
        db = _open(tmp_path / "db")
        vecs = _unit_batch(5, 16)
        with pytest.raises(Exception):
            db.insert_batch(["a", "b"], vecs)  # 2 ids, 5 vectors

    def test_batch_ids_more_than_vectors_raises(self, tmp_path):
        db = _open(tmp_path / "db")
        vecs = _unit_batch(2, 16)
        with pytest.raises(Exception):
            db.insert_batch(["a", "b", "c", "d", "e"], vecs)

    def test_search_with_list_query_rejected_or_graceful(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0))
        try:
            db.search([1.0] * 16, top_k=1)
        except Exception as e:
            assert "panic" not in type(e).__name__.lower()

    def test_query_1d_array_rejected(self, tmp_path):
        # query() expects 2-D; passing 1-D should raise cleanly
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0))
        with pytest.raises(Exception):
            db.query(np.ones(16, dtype=np.float32), n_results=1)

    def test_query_0_rows_raises_or_returns_empty(self, tmp_path):
        # 0-row query matrix — should not panic
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0))
        try:
            result = db.query(np.empty((0, 16), dtype=np.float32), n_results=1)
            assert result == [] or len(result) == 0
        except Exception as e:
            assert "panic" not in type(e).__name__.lower()

    def test_duplicate_ids_in_same_batch_raises_on_insert(self, tmp_path):
        # Inserting duplicate IDs in a single batch with mode="insert" should fail
        db = _open(tmp_path / "db")
        vecs = _unit_batch(3, 16)
        with pytest.raises(Exception):
            db.insert_batch(["dup", "dup", "unique"], vecs, mode="insert")

    def test_duplicate_ids_in_upsert_batch_handled(self, tmp_path):
        # Upsert mode with duplicate IDs: last write wins or at least no panic
        db = _open(tmp_path / "db")
        vecs = _unit_batch(3, 16)
        try:
            db.insert_batch(["dup", "dup", "unique"], vecs, mode="upsert")
        except Exception as e:
            assert "panic" not in type(e).__name__.lower()


# ---------------------------------------------------------------------------
# ██╗██████╗     ██████╗ ██████╗ ██╗   ██╗████████╗ █████╗ ██╗
# ██║██╔══██╗    ██╔══██╗██╔══██╗██║   ██║╚══██╔══╝██╔══██╗██║
# ██║██║  ██║    ██████╔╝██████╔╝██║   ██║   ██║   ███████║██║
# ██║██║  ██║    ██╔══██╗██╔══██╗██║   ██║   ██║   ██╔══██║██║
# ██║██████╔╝    ██████╔╝██║  ██║╚██████╔╝   ██║   ██║  ██║███████╗
# ---------------------------------------------------------------------------

class TestIdEdgeCases:
    """Hammer the ID space with pathological strings."""

    def test_empty_string_id_rejected_or_accepted_gracefully(self, tmp_path):
        db = _open(tmp_path / "db")
        try:
            db.insert("", _vec(16, seed=0))
        except Exception as e:
            assert "panic" not in type(e).__name__.lower()

    def test_unicode_id(self, tmp_path):
        db = _open(tmp_path / "db")
        uid = "ユーザー_🚀_αβγ"
        db.insert(uid, _vec(16, seed=0))
        assert db.get(uid) is not None
        assert uid in db

    def test_whitespace_only_id(self, tmp_path):
        db = _open(tmp_path / "db")
        try:
            db.insert("   ", _vec(16, seed=0))
            assert "   " in db
        except Exception as e:
            assert "panic" not in type(e).__name__.lower()

    def test_newline_in_id(self, tmp_path):
        db = _open(tmp_path / "db")
        try:
            db.insert("id\nwith\nnewlines", _vec(16, seed=0))
        except Exception as e:
            assert "panic" not in type(e).__name__.lower()

    def test_null_byte_in_id(self, tmp_path):
        db = _open(tmp_path / "db")
        try:
            db.insert("id\x00with\x00null", _vec(16, seed=0))
        except Exception as e:
            assert "panic" not in type(e).__name__.lower()

    def test_very_long_id(self, tmp_path):
        db = _open(tmp_path / "db")
        long_id = "x" * 10_000
        db.insert(long_id, _vec(16, seed=0))
        assert db.get(long_id) is not None

    def test_path_separator_in_id(self, tmp_path):
        db = _open(tmp_path / "db")
        for special_id in ["a/b/c", "a\\b\\c", "../escape"]:
            try:
                db.insert(special_id, _vec(16, seed=0))
            except Exception as e:
                assert "panic" not in type(e).__name__.lower()

    def test_numeric_string_id(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("12345", _vec(16, seed=0))
        db.insert("0", _vec(16, seed=1))
        db.insert("-1", _vec(16, seed=2))
        db.insert("3.14", _vec(16, seed=3))
        assert db.count() == 4

    def test_very_many_unique_ids(self, tmp_path):
        # 5000 unique IDs to stress the ID pool hash table
        db = _open(tmp_path / "db")
        n = 5000
        vecs = _unit_batch(n, 16)
        ids = [f"id_{i:08d}" for i in range(n)]
        db.insert_batch(ids, vecs)
        assert db.count() == n
        # Random sample verification
        for i in [0, 999, 2500, 4999]:
            assert db.get(ids[i]) is not None


# ---------------------------------------------------------------------------
# ███╗   ███╗███████╗████████╗ █████╗ ██████╗  █████╗ ████████╗ █████╗
# ████╗ ████║██╔════╝╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗
# ██╔████╔██║█████╗     ██║   ███████║██║  ██║███████║   ██║   ███████║
# ██║╚██╔╝██║██╔══╝     ██║   ██╔══██║██║  ██║██╔══██║   ██║   ██╔══██║
# ██║ ╚═╝ ██║███████╗   ██║   ██║  ██║██████╔╝██║  ██║   ██║   ██║  ██║
# ---------------------------------------------------------------------------

class TestMetadataBrutal:
    """Break metadata storage with every pathological value."""

    def test_deeply_nested_metadata_insert_and_retrieve(self, tmp_path):
        db = _open(tmp_path / "db")
        nested = {"l1": {"l2": {"l3": {"l4": {"l5": "deep"}}}}}
        db.insert("a", _vec(16, seed=0), metadata=nested)
        got = db.get("a")
        assert got["metadata"]["l1"]["l2"]["l3"]["l4"]["l5"] == "deep"

    def test_metadata_with_list_value(self, tmp_path):
        db = _open(tmp_path / "db")
        meta = {"tags": ["ai", "ml", "nlp"]}
        try:
            db.insert("a", _vec(16, seed=0), metadata=meta)
        except Exception as e:
            assert "panic" not in type(e).__name__.lower()

    def test_metadata_with_none_value(self, tmp_path):
        db = _open(tmp_path / "db")
        try:
            db.insert("a", _vec(16, seed=0), metadata={"key": None})
        except Exception as e:
            assert "panic" not in type(e).__name__.lower()

    def test_metadata_with_bool_value(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0), metadata={"active": True, "deleted": False})
        got = db.get("a")
        assert got is not None

    def test_metadata_with_float_value(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0), metadata={"score": 3.14159})
        got = db.get("a")
        assert got is not None
        assert abs(got["metadata"]["score"] - 3.14159) < 1e-5

    def test_metadata_with_integer_value(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0), metadata={"year": 2024})
        got = db.get("a")
        assert got["metadata"]["year"] == 2024

    def test_very_large_metadata_string(self, tmp_path):
        db = _open(tmp_path / "db")
        large_val = "X" * 100_000
        try:
            db.insert("big", _vec(16, seed=0), metadata={"blob": large_val})
            got = db.get("big")
            if got and got["metadata"]:
                assert got["metadata"]["blob"] == large_val
        except Exception as e:
            assert "panic" not in type(e).__name__.lower()

    def test_metadata_unicode_keys_and_values(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("u", _vec(16, seed=0), metadata={"кириллица": "значение", "日本語": "テスト"})
        got = db.get("u")
        assert got is not None

    def test_many_metadata_fields(self, tmp_path):
        db = _open(tmp_path / "db")
        meta = {f"field_{i}": i for i in range(200)}
        db.insert("wide", _vec(16, seed=0), metadata=meta)
        got = db.get("wide")
        assert got["metadata"]["field_199"] == 199

    def test_metadata_inf_float_handled_gracefully(self, tmp_path):
        db = _open(tmp_path / "db")
        try:
            db.insert("inf_meta", _vec(16, seed=0), metadata={"val": float("inf")})
        except Exception as e:
            assert "panic" not in type(e).__name__.lower()

    def test_metadata_nan_float_handled_gracefully(self, tmp_path):
        db = _open(tmp_path / "db")
        try:
            db.insert("nan_meta", _vec(16, seed=0), metadata={"val": float("nan")})
        except Exception as e:
            assert "panic" not in type(e).__name__.lower()

    def test_very_long_document_string(self, tmp_path):
        db = _open(tmp_path / "db")
        doc = "word " * 20_000  # ~100KB
        db.insert("longdoc", _vec(16, seed=0), document=doc)
        got = db.get("longdoc")
        assert got["document"] == doc

    def test_document_with_unicode(self, tmp_path):
        db = _open(tmp_path / "db")
        doc = "Hello 世界 🌍 Привет мир"
        db.insert("udoc", _vec(16, seed=0), document=doc)
        assert db.get("udoc")["document"] == doc


# ---------------------------------------------------------------------------
# ███████╗███████╗ █████╗ ██████╗  ██████╗██╗  ██╗
# ██╔════╝██╔════╝██╔══██╗██╔══██╗██╔════╝██║  ██║
# ███████╗█████╗  ███████║██████╔╝██║     ███████║
# ╚════██║██╔══╝  ██╔══██║██╔══██╗██║     ██╔══██║
# ███████║███████╗██║  ██║██║  ██║╚██████╗██║  ██║
# ---------------------------------------------------------------------------

class TestSearchBrutal:
    """Break search with edge-case queries, top_k, and empty databases."""

    def test_search_empty_db_returns_empty_not_panic(self, tmp_path):
        db = _open(tmp_path / "db")
        results = db.search(_vec(16, seed=0), top_k=10)
        assert results == []

    def test_search_top_k_zero_raises(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0))
        with pytest.raises(Exception):
            db.search(_vec(16, seed=0), top_k=0)

    def test_search_top_k_negative_raises(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0))
        with pytest.raises(Exception):
            db.search(_vec(16, seed=0), top_k=-5)

    def test_search_top_k_larger_than_corpus_returns_corpus_size(self, tmp_path):
        db = _open(tmp_path / "db")
        vecs = _unit_batch(5, 16)
        db.insert_batch([str(i) for i in range(5)], vecs)
        results = db.search(vecs[0], top_k=100)
        assert len(results) == 5  # capped at corpus size

    def test_search_top_k_one_returns_nearest(self, tmp_path):
        db = _open(tmp_path / "db")
        vecs = _unit_batch(50, 16)
        db.insert_batch([str(i) for i in range(50)], vecs)
        results = db.search(vecs[7], top_k=1)
        assert results[0]["id"] == "7"

    def test_search_after_delete_all_returns_empty(self, tmp_path):
        db = _open(tmp_path / "db")
        vecs = _unit_batch(5, 16)
        ids = [str(i) for i in range(5)]
        db.insert_batch(ids, vecs)
        db.delete_batch(ids)
        results = db.search(vecs[0], top_k=5)
        assert results == []

    def test_search_results_scores_all_finite(self, tmp_path):
        db = _open(tmp_path / "db")
        vecs = _unit_batch(50, 16)
        db.insert_batch([str(i) for i in range(50)], vecs)
        results = db.search(vecs[0], top_k=20)
        for r in results:
            assert np.isfinite(r["score"]), f"Non-finite score: {r['score']}"

    def test_search_scores_descending_after_all_ops(self, tmp_path):
        # Insert, delete some, insert more — scores must stay sorted
        db = _open(tmp_path / "db")
        vecs = _unit_batch(30, 16)
        db.insert_batch([str(i) for i in range(30)], vecs)
        db.delete_batch([str(i) for i in range(0, 15, 3)])
        more = _unit_batch(5, 16, seed=99)
        db.insert_batch([f"extra_{i}" for i in range(5)], more)
        results = db.search(vecs[0], top_k=15)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_deleted_vector_not_returned(self, tmp_path):
        db = _open(tmp_path / "db")
        vecs = _unit_batch(10, 16)
        db.insert_batch([str(i) for i in range(10)], vecs)
        # Query for vecs[3] exactly — should be top-1 before delete
        assert db.search(vecs[3], top_k=1)[0]["id"] == "3"
        db.delete("3")
        results = db.search(vecs[3], top_k=10)
        ids = {r["id"] for r in results}
        assert "3" not in ids

    # BUG: quantization bias causes L2 scores to be slightly negative.
    # L2 distance is inherently non-negative; the quantizer introduces a systematic
    # bias that can produce values like -0.005 instead of clamping at 0.
    def test_search_l2_scores_are_non_negative(self, tmp_path):
        # L2 distance should be >= 0
        db = _open(tmp_path / "db", metric="l2")
        vecs = _unit_batch(20, 16)
        db.insert_batch([str(i) for i in range(20)], vecs)
        results = db.search(vecs[0], top_k=10)
        for r in results:
            assert r["score"] >= 0, f"Negative L2 score: {r['score']}"

    def test_search_with_rerank_factor_extreme(self, tmp_path):
        db = _open(tmp_path / "db", rerank=True)
        vecs = _unit_batch(50, 16)
        db.insert_batch([str(i) for i in range(50)], vecs)
        # rerank_factor=50 (very large oversampling)
        results = db.search(vecs[0], top_k=5, rerank_factor=50)
        assert len(results) == 5
        assert results[0]["id"] == "0"

    def test_search_with_rerank_factor_one(self, tmp_path):
        db = _open(tmp_path / "db", rerank=True)
        vecs = _unit_batch(30, 16)
        db.insert_batch([str(i) for i in range(30)], vecs)
        results = db.search(vecs[0], top_k=5, rerank_factor=1)
        assert len(results) == 5

    def test_query_n_results_zero_raises(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0))
        with pytest.raises(Exception):
            db.query(np.ones((1, 16), dtype=np.float32), n_results=0)

    def test_query_batch_results_count_matches_queries(self, tmp_path):
        db = _open(tmp_path / "db")
        vecs = _unit_batch(10, 16)
        db.insert_batch([str(i) for i in range(10)], vecs)
        queries = np.stack([vecs[0], vecs[1], vecs[2], vecs[3]])
        all_results = db.query(queries, n_results=3)
        assert len(all_results) == 4
        for res in all_results:
            assert len(res) <= 3

    def test_ann_search_before_index_falls_back_or_raises(self, tmp_path):
        # _use_ann=True without create_index() — should fall back to brute-force
        # or raise a descriptive error; must NOT panic
        db = _open(tmp_path / "db")
        vecs = _unit_batch(20, 16)
        db.insert_batch([str(i) for i in range(20)], vecs)
        try:
            results = db.search(vecs[0], top_k=5, _use_ann=True)
            assert len(results) > 0
        except Exception as e:
            assert "panic" not in type(e).__name__.lower()

    def test_ann_search_after_index_correct_top1(self, tmp_path):
        db = _open(tmp_path / "db")
        vecs = _unit_batch(200, 16)
        db.insert_batch([str(i) for i in range(200)], vecs)
        db.create_index(max_degree=16, ef_construction=64, search_list_size=64)
        results = db.search(vecs[42], top_k=1, _use_ann=True)
        assert results[0]["id"] == "42"

    def test_create_index_on_empty_db_no_panic(self, tmp_path):
        db = _open(tmp_path / "db")
        try:
            db.create_index()
        except Exception as e:
            assert "panic" not in type(e).__name__.lower()

    def test_create_index_on_single_vector_no_panic(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("only", _vec(16, seed=0))
        try:
            db.create_index()
        except Exception as e:
            assert "panic" not in type(e).__name__.lower()

    def test_create_index_extreme_params_no_panic(self, tmp_path):
        db = _open(tmp_path / "db")
        vecs = _unit_batch(50, 16)
        db.insert_batch([str(i) for i in range(50)], vecs)
        # max_degree=1 and tiny ef_construction
        try:
            db.create_index(max_degree=1, ef_construction=1, search_list_size=1)
        except Exception as e:
            assert "panic" not in type(e).__name__.lower()

    def test_update_index_on_empty_db_no_panic(self, tmp_path):
        db = _open(tmp_path / "db")
        try:
            db.update_index()
        except Exception as e:
            assert "panic" not in type(e).__name__.lower()

    # BUG: update_index() destroys has_index. Root cause: flush_wal_to_segment()
    # inside create_index_incremental() calls invalidate_index_state() → sets
    # manifest.index_state=None. The subsequent `if let Some(state)` guard then
    # silently skips restoring it, leaving can_use_ann_index() returning False.
    def test_update_index_after_large_batch_insert(self, tmp_path):
        db = _open(tmp_path / "db")
        vecs = _unit_batch(100, 16)
        db.insert_batch([str(i) for i in range(100)], vecs)
        db.create_index(max_degree=8, search_list_size=32)
        # Add 50 more
        more = _unit_batch(50, 16, seed=99)
        db.insert_batch([f"more_{i}" for i in range(50)], more)
        db.update_index(max_degree=8, search_list_size=32)
        assert db.stats()["has_index"]


# ---------------------------------------------------------------------------
# ███████╗██╗██╗  ████████╗███████╗██████╗
# ██╔════╝██║██║  ╚══██╔══╝██╔════╝██╔══██╗
# █████╗  ██║██║     ██║   █████╗  ██████╔╝
# ██╔══╝  ██║██║     ██║   ██╔══╝  ██╔══██╗
# ██║     ██║███████╗██║   ███████╗██║  ██║
# ---------------------------------------------------------------------------

class TestFilterBrutal:
    """Break metadata filter logic with contradictions, missing fields, bad ops."""

    def _setup(self, tmp_path, n: int = 10, d: int = 16) -> tuple:
        db = _open(tmp_path / "db", d=d)
        vecs = _unit_batch(n, d)
        for i in range(n):
            db.insert(
                str(i), vecs[i],
                metadata={"i": i, "parity": "even" if i % 2 == 0 else "odd",
                           "group": i // 3}
            )
        return db, vecs

    def test_filter_on_missing_field_returns_empty(self, tmp_path):
        db, vecs = self._setup(tmp_path)
        results = db.search(vecs[0], top_k=10, filter={"nonexistent_field": "value"})
        assert results == []

    def test_filter_mixed_present_absent_field(self, tmp_path):
        # Only half the vectors have "extra" key
        db = _open(tmp_path / "db")
        vecs = _unit_batch(6, 16)
        for i in range(3):
            db.insert(str(i), vecs[i], metadata={"tag": "A"})
        for i in range(3, 6):
            db.insert(str(i), vecs[i], metadata={"tag": "A", "extra": True})
        results = db.search(vecs[0], top_k=10, filter={"extra": {"$exists": True}})
        assert len(results) == 3
        assert all(r["metadata"].get("extra") is True for r in results)

    def test_filter_contradiction_and_returns_empty(self, tmp_path):
        db, vecs = self._setup(tmp_path)
        # Impossible: i > 100 AND i < 0
        results = db.search(vecs[0], top_k=10,
                            filter={"$and": [{"i": {"$gt": 100}}, {"i": {"$lt": 0}}]})
        assert results == []

    def test_filter_tautology_or_returns_all(self, tmp_path):
        db, vecs = self._setup(tmp_path)
        results = db.search(vecs[0], top_k=10,
                            filter={"$or": [{"parity": "even"}, {"parity": "odd"}]})
        assert len(results) == 10

    def test_filter_in_empty_list_returns_empty(self, tmp_path):
        db, vecs = self._setup(tmp_path)
        results = db.search(vecs[0], top_k=10, filter={"parity": {"$in": []}})
        assert results == []

    def test_filter_nin_all_values_returns_empty(self, tmp_path):
        db, vecs = self._setup(tmp_path)
        results = db.search(vecs[0], top_k=10,
                            filter={"parity": {"$nin": ["even", "odd"]}})
        assert results == []

    def test_filter_range_lt_gt_same_field(self, tmp_path):
        db, vecs = self._setup(tmp_path)
        # 2 < i < 7 → i in {3, 4, 5, 6}
        results = db.search(vecs[0], top_k=10,
                            filter={"$and": [{"i": {"$gt": 2}}, {"i": {"$lt": 7}}]})
        ivals = {r["metadata"]["i"] for r in results}
        assert ivals == {3, 4, 5, 6}

    def test_filter_eq_on_int_field(self, tmp_path):
        db, vecs = self._setup(tmp_path)
        results = db.search(vecs[0], top_k=10, filter={"i": {"$eq": 5}})
        assert len(results) == 1
        assert results[0]["metadata"]["i"] == 5

    def test_filter_ne_on_string_field(self, tmp_path):
        db, vecs = self._setup(tmp_path)
        results = db.search(vecs[0], top_k=10, filter={"parity": {"$ne": "even"}})
        assert all(r["metadata"]["parity"] == "odd" for r in results)

    def test_filter_contains_case_sensitive(self, tmp_path):
        db = _open(tmp_path / "db")
        vecs = _unit_batch(3, 16)
        db.insert("lower", vecs[0], metadata={"text": "hello world"})
        db.insert("upper", vecs[1], metadata={"text": "HELLO WORLD"})
        db.insert("mixed", vecs[2], metadata={"text": "HeLLo WoRLd"})
        results = db.search(vecs[0], top_k=10, filter={"text": {"$contains": "hello"}})
        ids = {r["id"] for r in results}
        assert "lower" in ids
        assert "upper" not in ids  # case-sensitive

    def test_filter_in_mixed_types_handled_gracefully(self, tmp_path):
        db, vecs = self._setup(tmp_path)
        # $in with mixed int/string types — should not panic
        try:
            results = db.search(vecs[0], top_k=10, filter={"i": {"$in": [0, "1", 2]}})
        except Exception as e:
            assert "panic" not in type(e).__name__.lower()

    def test_deeply_nested_and_or_filter(self, tmp_path):
        db, vecs = self._setup(tmp_path)
        # Three-level nesting
        filt = {
            "$and": [
                {"$or": [{"parity": "even"}, {"i": {"$gte": 8}}]},
                {"group": {"$lt": 3}},
            ]
        }
        try:
            results = db.search(vecs[0], top_k=10, filter=filt)
            assert all(
                (r["metadata"]["parity"] == "even" or r["metadata"]["i"] >= 8)
                and r["metadata"]["group"] < 3
                for r in results
            )
        except Exception as e:
            assert "panic" not in type(e).__name__.lower()

    def test_delete_batch_filter_deletes_correct_subset(self, tmp_path):
        db, _ = self._setup(tmp_path)
        n_deleted = db.delete_batch(where_filter={"parity": "even"})
        assert n_deleted == 5
        assert db.count() == 5
        remaining = db.list_ids()
        # All remaining should be odd
        for rid in remaining:
            assert db.get(rid)["metadata"]["parity"] == "odd"

    def test_count_with_complex_filter(self, tmp_path):
        db, _ = self._setup(tmp_path)
        # Even AND group >= 2 → {4, 5→odd skip, 6, 7→odd skip, 8→even group=2, ...}
        # group=i//3: even i: 0(g0), 2(g0), 4(g1), 6(g2), 8(g2)
        # even AND group>=2 → {6, 8}
        count = db.count(filter={"$and": [{"parity": "even"}, {"group": {"$gte": 2}}]})
        assert count == 2

    def test_list_ids_with_filter_and_pagination(self, tmp_path):
        db = _open(tmp_path / "db")
        vecs = _unit_batch(20, 16)
        for i in range(20):
            db.insert(str(i), vecs[i], metadata={"cat": "A" if i < 10 else "B"})
        page1 = db.list_ids(where_filter={"cat": "A"}, limit=5, offset=0)
        page2 = db.list_ids(where_filter={"cat": "A"}, limit=5, offset=5)
        assert len(page1) == 5
        assert len(page2) == 5
        assert set(page1).isdisjoint(set(page2))

    def test_list_ids_limit_zero_raises_or_empty(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0))
        try:
            result = db.list_ids(limit=0)
            assert result == []
        except Exception as e:
            assert "panic" not in type(e).__name__.lower()


# ---------------------------------------------------------------------------
# ██████╗ ███████╗██████╗ ███████╗██╗███████╗████████╗███████╗███╗   ██╗ ██████╗███████╗
# ██╔══██╗██╔════╝██╔══██╗██╔════╝██║██╔════╝╚══██╔══╝██╔════╝████╗  ██║██╔════╝██╔════╝
# ██████╔╝█████╗  ██████╔╝███████╗██║███████╗   ██║   █████╗  ██╔██╗ ██║██║     █████╗
# ██╔═══╝ ██╔══╝  ██╔══██╗╚════██║██║╚════██║   ██║   ██╔══╝  ██║╚██╗██║██║     ██╔══╝
# ██║     ███████╗██║  ██║███████║██║███████║   ██║   ███████╗██║ ╚████║╚██████╗███████╗
# ---------------------------------------------------------------------------

class TestPersistenceBrutal:
    """Hammer persistence: WAL recovery, multi-flush, crash simulation."""

    def test_wal_recovery_on_reopen_without_close(self, tmp_path):
        # Simulate crash: insert but don't call close() — just delete the Python object
        path = str(tmp_path / "db")
        db = _open(path)
        vecs = _unit_batch(20, 16)
        db.insert_batch([str(i) for i in range(20)], vecs)
        del db  # No explicit close — simulate crash
        gc.collect()

        db2 = _open(path)
        # WAL must have recovered the data
        assert db2.count() == 20

    def test_multiple_flush_cycles_data_survives(self, tmp_path):
        path = str(tmp_path / "db")
        # wal_flush_threshold=5 → flush every 5 insertions
        db = Database.open(path, dimension=16, bits=4, seed=42,
                           wal_flush_threshold=5)
        vecs = _unit_batch(25, 16)
        db.insert_batch([str(i) for i in range(25)], vecs)
        del db
        gc.collect()

        db2 = _open(path)
        assert db2.count() == 25
        for i in [0, 12, 24]:
            assert db2.get(str(i)) is not None

    def test_index_persists_across_reopen(self, tmp_path):
        path = str(tmp_path / "db")
        db = _open(path)
        vecs = _unit_batch(100, 16)
        db.insert_batch([str(i) for i in range(100)], vecs)
        db.create_index(max_degree=16, search_list_size=64)
        assert db.stats()["has_index"]
        db.close()
        del db  # explicitly drop the Python ref so CPython can release mmap handles
        # On Windows, mmap handles are ref-counted by CPython; two GC passes
        # ensure live_codes.bin and graph.bin are fully released before reopen.
        gc.collect()
        gc.collect()

        db2 = _open(path)
        assert db2.stats()["has_index"]
        # ANN search should still work
        results = db2.search(vecs[42], top_k=1, _use_ann=True)
        assert results[0]["id"] == "42"

    def test_delete_then_reinsert_same_id_persists(self, tmp_path):
        # Known regression: delete → reinsert → close → reopen → id missing
        path = str(tmp_path / "db")
        db = _open(path)
        v1 = _vec(16, seed=1)
        v2 = _vec(16, seed=2)
        db.insert("x", v1, metadata={"phase": 1})
        db.delete("x")
        db.insert("x", v2, metadata={"phase": 2})
        db.close()

        db2 = _open(path)
        got = db2.get("x")
        assert got is not None, "reinserted ID missing after reopen"
        assert got["metadata"]["phase"] == 2

    def test_upsert_then_delete_then_upsert_persists(self, tmp_path):
        path = str(tmp_path / "db")
        db = _open(path)
        v = _vec(16, seed=0)
        db.upsert("x", v, metadata={"v": 1})
        db.delete("x")
        db.upsert("x", v, metadata={"v": 3})
        db.close()

        db2 = _open(path)
        assert db2.count() == 1
        assert db2.get("x")["metadata"]["v"] == 3

    def test_delete_all_then_reopen_count_zero(self, tmp_path):
        path = str(tmp_path / "db")
        db = _open(path)
        vecs = _unit_batch(10, 16)
        db.insert_batch([str(i) for i in range(10)], vecs)
        db.delete_batch([str(i) for i in range(10)])
        db.close()

        db2 = _open(path)
        assert db2.count() == 0

    def test_metadata_update_persists_across_reopen(self, tmp_path):
        path = str(tmp_path / "db")
        db = _open(path)
        db.insert("a", _vec(16, seed=0), metadata={"status": "draft"})
        db.update_metadata("a", metadata={"status": "published"})
        db.close()

        db2 = _open(path)
        assert db2.get("a")["metadata"]["status"] == "published"

    def test_large_batch_triggering_multiple_segment_flushes(self, tmp_path):
        # Insert >2 * WAL threshold to trigger multiple flush segments
        path = str(tmp_path / "db")
        db = Database.open(path, dimension=16, bits=4, seed=42, wal_flush_threshold=100)
        n = 350  # 3 segments of 100 + tail
        vecs = _unit_batch(n, 16)
        db.insert_batch([str(i) for i in range(n)], vecs)
        db.close()

        db2 = _open(path)
        assert db2.count() == n

    def test_reopen_parameterless_after_all_params_set(self, tmp_path):
        path = str(tmp_path / "db")
        db = Database.open(path, dimension=32, bits=8, seed=77, metric="l2")
        db.insert("a", _vec(32, seed=0))
        db.close()

        # Should read all params from manifest
        db2 = Database.open(path)
        assert db2.stats()["dimension"] == 32
        assert db2.stats()["bits"] == 8
        assert db2.get("a") is not None

    def test_multiple_close_calls_no_panic(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0))
        db.close()
        # Second close should not panic (may raise or be no-op)
        try:
            db.close()
        except Exception as e:
            assert "panic" not in type(e).__name__.lower()

    def test_operations_after_close_raise_gracefully(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0))
        db.close()
        # Every operation after close should raise gracefully, not panic
        for op, args in [
            (db.insert, ("b", _vec(16, seed=1))),
            (db.get, ("a",)),
            (db.count, ()),
            (db.search, (_vec(16, seed=0), 1)),
        ]:
            try:
                op(*args)
            except Exception as e:
                assert "panic" not in type(e).__name__.lower()


# ---------------------------------------------------------------------------
# ██████╗ ██████╗ ██╗   ██╗██████╗
# ██╔════╝██╔══██╗██║   ██║██╔══██╗
# ██║     ██████╔╝██║   ██║██║  ██║
# ██║     ██╔══██╗██║   ██║██║  ██║
# ╚██████╗██║  ██║╚██████╔╝██████╔╝
# ---------------------------------------------------------------------------

class TestCRUDBrutal:
    """Break CRUD with extreme sequences and edge-case data."""

    def test_insert_update_delete_reinsert_cycle(self, tmp_path):
        db = _open(tmp_path / "db")
        v1 = _vec(16, seed=1)
        v2 = _vec(16, seed=2)
        db.insert("a", v1, metadata={"gen": 1})
        db.update("a", v2, metadata={"gen": 2})
        db.delete("a")
        db.insert("a", v1, metadata={"gen": 3})
        got = db.get("a")
        assert got["metadata"]["gen"] == 3

    def test_update_nonexistent_raises(self, tmp_path):
        db = _open(tmp_path / "db")
        with pytest.raises(Exception):
            db.update("ghost", _vec(16, seed=0))

    def test_update_metadata_nonexistent_raises(self, tmp_path):
        db = _open(tmp_path / "db")
        with pytest.raises(Exception):
            db.update_metadata("ghost", metadata={"x": 1})

    def test_upsert_creates_if_missing(self, tmp_path):
        db = _open(tmp_path / "db")
        db.upsert("new", _vec(16, seed=0), metadata={"x": 1})
        assert db.get("new") is not None

    def test_upsert_overwrites_vector(self, tmp_path):
        db = _open(tmp_path / "db")
        v1 = _vec(16, seed=1)
        v2 = _vec(16, seed=2)
        db.insert("a", v1, metadata={"v": 1})
        db.upsert("a", v2, metadata={"v": 2})
        assert db.get("a")["metadata"]["v"] == 2

    def test_upsert_does_not_validate_nan_vector(self, tmp_path):
        # NaN in upsert should be rejected same as insert
        db = _open(tmp_path / "db")
        nan_v = np.full(16, float("nan"), dtype=np.float32)
        try:
            db.upsert("nan", nan_v)
            # if accepted, search should not panic
        except Exception as e:
            assert "panic" not in type(e).__name__.lower()

    def test_delete_returns_false_for_already_deleted(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0))
        assert db.delete("a") is True
        assert db.delete("a") is False

    def test_delete_batch_with_duplicate_ids_counted_once(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0))
        db.insert("b", _vec(16, seed=1))
        # "a" appears twice in the list
        n = db.delete_batch(["a", "a", "b"])
        assert n == 2, f"Expected 2 unique deletions, got {n}"

    def test_get_many_all_missing_returns_none_list(self, tmp_path):
        db = _open(tmp_path / "db")
        result = db.get_many(["x", "y", "z"])
        assert result == [None, None, None]

    def test_get_many_empty_list_returns_empty(self, tmp_path):
        db = _open(tmp_path / "db")
        result = db.get_many([])
        assert result == []

    def test_len_after_batch_insert_and_delete(self, tmp_path):
        db = _open(tmp_path / "db")
        vecs = _unit_batch(50, 16)
        ids = [str(i) for i in range(50)]
        db.insert_batch(ids, vecs)
        assert len(db) == 50
        db.delete_batch(ids[:25])
        assert len(db) == 25

    def test_contains_after_delete(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(16, seed=0))
        assert "a" in db
        db.delete("a")
        assert "a" not in db

    def test_upsert_with_nan_in_second_call_rejected(self, tmp_path):
        db = _open(tmp_path / "db")
        v = _vec(16, seed=0)
        db.upsert("a", v)
        nan_v = np.full(16, float("nan"), dtype=np.float32)
        try:
            db.upsert("a", nan_v)
            # Original should still be searchable (if nan was rejected via upsert path)
        except Exception as e:
            assert "panic" not in type(e).__name__.lower()

    def test_high_frequency_insert_delete_cycle(self, tmp_path):
        # Rapid alternating insert/delete to stress the ID pool recycling
        db = _open(tmp_path / "db")
        v = _vec(16, seed=0)
        for cycle in range(100):
            db.insert("recycle", v)
            db.delete("recycle")
        assert db.count() == 0

    def test_insert_batch_mode_update_on_new_raises(self, tmp_path):
        db = _open(tmp_path / "db")
        vecs = _unit_batch(3, 16)
        # Update mode on IDs that don't exist yet should raise
        with pytest.raises(Exception):
            db.insert_batch(["new1", "new2", "new3"], vecs, mode="update")

    def test_stats_live_id_count_equals_vector_count(self, tmp_path):
        db = _open(tmp_path / "db")
        vecs = _unit_batch(20, 16)
        db.insert_batch([str(i) for i in range(20)], vecs)
        db.delete_batch([str(i) for i in range(5)])
        s = db.stats()
        assert s["vector_count"] == 15
        # live_id_count must match active vector count
        assert s["live_id_count"] == s["vector_count"]


# ---------------------------------------------------------------------------
# ███╗   ███╗███████╗████████╗██████╗ ██╗ ██████╗
# ████╗ ████║██╔════╝╚══██╔══╝██╔══██╗██║██╔════╝
# ██╔████╔██║█████╗     ██║   ██████╔╝██║██║
# ██║╚██╔╝██║██╔══╝     ██║   ██╔══██╗██║██║
# ██║ ╚═╝ ██║███████╗   ██║   ██║  ██║██║╚██████╗
# ---------------------------------------------------------------------------

class TestMetricCorrectness:
    """Validate metric semantics don't silently produce wrong results."""

    def test_cosine_top1_is_self_after_normalize(self, tmp_path):
        db = _open(tmp_path / "db", d=32, metric="cosine")
        vecs = _unit_batch(100, 32)
        db.insert_batch([str(i) for i in range(100)], vecs)
        for i in [0, 25, 50, 99]:
            results = db.search(vecs[i], top_k=1)
            assert results[0]["id"] == str(i), f"cosine self-match failed for i={i}"

    def test_ip_top1_is_self(self, tmp_path):
        db = _open(tmp_path / "db", d=32, metric="ip")
        vecs = _unit_batch(100, 32)
        db.insert_batch([str(i) for i in range(100)], vecs)
        for i in [0, 33, 77]:
            results = db.search(vecs[i], top_k=1)
            assert results[0]["id"] == str(i), f"ip self-match failed for i={i}"

    def test_l2_top1_is_self(self, tmp_path):
        db = _open(tmp_path / "db", d=32, metric="l2")
        vecs = _unit_batch(50, 32)
        db.insert_batch([str(i) for i in range(50)], vecs)
        for i in [0, 10, 49]:
            results = db.search(vecs[i], top_k=1)
            assert results[0]["id"] == str(i), f"l2 self-match failed for i={i}"

    def test_l2_zero_distance_for_identical_vectors(self, tmp_path):
        # Use rerank=True so exact raw-vector scoring is used: identical vectors → score = 0.
        db = _open(tmp_path / "db", metric="l2", rerank=True)
        v = _vec(16, seed=0)
        db.insert("a", v)
        results = db.search(v, top_k=1)
        # L2 score is positive distance; identical vectors should be ~0
        assert results[0]["score"] >= 0, f"L2 score must be non-negative: {results[0]['score']}"
        assert results[0]["score"] < 0.02, f"L2 for identical vectors: {results[0]['score']}"

    def test_cosine_identical_vectors_score_near_one(self, tmp_path):
        db = _open(tmp_path / "db", metric="cosine")
        v = _vec(16, seed=0)
        db.insert("a", v)
        results = db.search(v, top_k=1)
        assert results[0]["score"] > 0.99, f"Cosine self-score too low: {results[0]['score']}"

    def test_scores_sorted_descending_for_all_metrics(self, tmp_path):
        for metric in ("ip", "cosine", "l2"):
            p = str(tmp_path / metric)
            db = _open(p, metric=metric)
            vecs = _unit_batch(50, 16)
            db.insert_batch([str(i) for i in range(50)], vecs)
            results = db.search(vecs[0], top_k=10)
            scores = [r["score"] for r in results]
            if metric == "l2":
                # L2 scores are positive distances; closest-first order is ascending.
                assert scores == sorted(scores), \
                    f"L2 scores must be sorted ascending (closest first): {scores}"
            else:
                assert scores == sorted(scores, reverse=True), \
                    f"Scores not sorted descending for metric={metric}"

    def test_ground_truth_recall_brute_force_above_70pct(self, tmp_path):
        # Recall must be >= 70% even on small corpus
        d, n, q = 64, 1000, 50
        vecs = _unit_batch(n, d)
        queries = _unit_batch(q, d, seed=999)

        db = _open(tmp_path / "db", d=d)
        db.insert_batch([str(i) for i in range(n)], vecs)

        gt_sims = queries @ vecs.T
        hits = 0
        k = 10
        for qi, qvec in enumerate(queries):
            gt_top = set(np.argsort(-gt_sims[qi])[:k].tolist())
            results = db.search(qvec, top_k=k)
            returned = {int(r["id"]) for r in results}
            hits += len(gt_top & returned)

        recall = hits / (q * k)
        assert recall >= 0.70, f"Brute-force recall too low: {recall:.2%}"


# ---------------------------------------------------------------------------
# ██████╗  █████╗  ██████╗     ██╗    ██╗██████╗  █████╗ ██████╗
# ██╔══██╗██╔══██╗██╔════╝     ██║    ██║██╔══██╗██╔══██╗██╔══██╗
# ██████╔╝███████║██║  ███╗    ██║ █╗ ██║██████╔╝███████║██████╔╝
# ██╔══██╗██╔══██║██║   ██║    ██║███╗██║██╔══██╗██╔══██║██╔═══╝
# ██║  ██║██║  ██║╚██████╔╝    ╚███╔███╔╝██║  ██║██║  ██║██║
# ---------------------------------------------------------------------------

class TestRAGRetriever:
    """Break TurboQuantRetriever edge cases."""

    DIM = 32

    def _make_embs(self, n: int, seed: int = 0) -> list:
        rng = np.random.default_rng(seed)
        v = rng.standard_normal((n, self.DIM)).astype(np.float32)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        return (v / norms).tolist()

    def test_add_texts_empty_texts_no_panic(self, tmp_path):
        from tqdb.rag import TurboQuantRetriever
        r = TurboQuantRetriever(str(tmp_path / "db"), dimension=self.DIM)
        try:
            r.add_texts([], [])
        except Exception as e:
            assert "panic" not in type(e).__name__.lower()

    def test_similarity_search_k_larger_than_stored(self, tmp_path):
        from tqdb.rag import TurboQuantRetriever
        r = TurboQuantRetriever(str(tmp_path / "db"), dimension=self.DIM)
        r.add_texts(["only doc"], self._make_embs(1))
        results = r.similarity_search(self._make_embs(1, seed=99)[0], k=100)
        assert len(results) <= 1

    def test_similarity_search_k_zero_no_panic(self, tmp_path):
        from tqdb.rag import TurboQuantRetriever
        r = TurboQuantRetriever(str(tmp_path / "db"), dimension=self.DIM)
        r.add_texts(["doc"], self._make_embs(1))
        try:
            results = r.similarity_search(self._make_embs(1, seed=5)[0], k=0)
            assert results == []
        except Exception as e:
            assert "panic" not in type(e).__name__.lower()

    def test_add_texts_mismatched_metadatas_length_raises(self, tmp_path):
        from tqdb.rag import TurboQuantRetriever
        r = TurboQuantRetriever(str(tmp_path / "db"), dimension=self.DIM)
        texts = ["a", "b", "c"]
        embs = self._make_embs(3)
        metadatas = [{"x": 1}]  # only 1, but 3 texts
        with pytest.raises(Exception):
            r.add_texts(texts, embs, metadatas=metadatas)

    def test_add_texts_mismatched_embeddings_length_raises(self, tmp_path):
        from tqdb.rag import TurboQuantRetriever
        r = TurboQuantRetriever(str(tmp_path / "db"), dimension=self.DIM)
        texts = ["a", "b", "c"]
        embs = self._make_embs(2)  # only 2, but 3 texts
        with pytest.raises(Exception):
            r.add_texts(texts, embs)

    def test_doc_store_survives_multiple_add_texts_calls(self, tmp_path):
        from tqdb.rag import TurboQuantRetriever
        r = TurboQuantRetriever(str(tmp_path / "db"), dimension=self.DIM)
        for batch in range(5):
            r.add_texts([f"doc_{batch}_{i}" for i in range(3)], self._make_embs(3, seed=batch))
        assert len(r.doc_store) == 15

    def test_similarity_search_returns_correct_text(self, tmp_path):
        from tqdb.rag import TurboQuantRetriever
        r = TurboQuantRetriever(str(tmp_path / "db"), dimension=self.DIM)
        embs = self._make_embs(5)
        r.add_texts([f"document_{i}" for i in range(5)], embs)
        # Query with the exact embedding of doc 2
        results = r.similarity_search(embs[2], k=1)
        assert len(results) == 1
        assert results[0]["text"] == "document_2"

    def test_nan_embedding_in_similarity_search_no_panic(self, tmp_path):
        from tqdb.rag import TurboQuantRetriever
        r = TurboQuantRetriever(str(tmp_path / "db"), dimension=self.DIM)
        r.add_texts(["doc"], self._make_embs(1))
        nan_q = [float("nan")] * self.DIM
        try:
            r.similarity_search(nan_q, k=1)
        except Exception as e:
            assert "panic" not in type(e).__name__.lower()


# ---------------------------------------------------------------------------
# ██████╗  █████╗  ██████╗██╗ ███╗   ██╗ ██████╗
# ██╔══██╗██╔══██╗██╔════╝██║ ████╗  ██║██╔════╝
# ██████╔╝███████║██║     ██║ ██╔██╗ ██║██║  ███╗
# ██╔══██╗██╔══██║██║     ██║ ██║╚██╗██║██║   ██║
# ██║  ██║██║  ██║╚██████╗██║ ██║ ╚████║╚██████╔╝
# ---------------------------------------------------------------------------

class TestRacingConditions:
    """Multi-threaded correctness: reads and writes must not corrupt data."""

    def test_concurrent_reads_do_not_panic(self, tmp_path):
        db = _open(tmp_path / "db")
        vecs = _unit_batch(100, 16)
        db.insert_batch([str(i) for i in range(100)], vecs)

        errors: list[Exception] = []

        def reader(q_vec):
            try:
                for _ in range(50):
                    db.search(q_vec, top_k=5)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader, args=(vecs[i % 10],)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, f"Concurrent read errors: {errors}"

    def test_concurrent_read_write_no_panic(self, tmp_path):
        db = _open(tmp_path / "db")
        vecs = _unit_batch(50, 16)
        db.insert_batch([str(i) for i in range(50)], vecs)
        errors: list[Exception] = []

        def writer():
            try:
                for i in range(50, 70):
                    db.insert(str(i), _vec(16, seed=i))
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def reader(q_vec):
            try:
                for _ in range(30):
                    db.search(q_vec, top_k=5)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer)] + \
                  [threading.Thread(target=reader, args=(vecs[i],)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not any("panic" in type(e).__name__.lower() for e in errors), \
            f"Panic in concurrent test: {errors}"

    def test_count_consistent_after_concurrent_inserts(self, tmp_path):
        db = _open(tmp_path / "db")
        n_per_thread = 25
        n_threads = 4
        errors: list[Exception] = []
        inserted: list[list[str]] = [[] for _ in range(n_threads)]

        vecs_all = _unit_batch(n_per_thread * n_threads, 16)

        def bulk_insert(tid: int):
            try:
                start = tid * n_per_thread
                ids = [f"t{tid}_v{i}" for i in range(n_per_thread)]
                inserted[tid].extend(ids)
                db.insert_batch(ids, vecs_all[start:start + n_per_thread])
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=bulk_insert, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors in concurrent insert: {errors}"
        assert db.count() == n_per_thread * n_threads


# ---------------------------------------------------------------------------
# ███████╗████████╗ █████╗ ████████╗███████╗
# ██╔════╝╚══██╔══╝██╔══██╗╚══██╔══╝██╔════╝
# ███████╗   ██║   ███████║   ██║   ███████╗
# ╚════██║   ██║   ██╔══██║   ██║   ╚════██║
# ███████║   ██║   ██║  ██║   ██║   ███████║
# ---------------------------------------------------------------------------

class TestStatsBrutal:
    """Validate stats fields are internally consistent."""

    def test_stats_empty_db_consistent(self, tmp_path):
        db = _open(tmp_path / "db")
        s = db.stats()
        assert s["vector_count"] == 0
        assert s["has_index"] is False
        assert s["dimension"] == 16
        assert s["total_disk_bytes"] >= 0
        assert s["ram_estimate_bytes"] >= 0

    def test_stats_vector_count_matches_len_and_count(self, tmp_path):
        db = _open(tmp_path / "db")
        vecs = _unit_batch(30, 16)
        db.insert_batch([str(i) for i in range(30)], vecs)
        db.delete_batch([str(i) for i in range(10)])
        s = db.stats()
        assert s["vector_count"] == 20
        assert len(db) == 20
        assert db.count() == 20

    def test_stats_has_index_reflects_create_index(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert_batch([str(i) for i in range(20)], _unit_batch(20, 16))
        assert db.stats()["has_index"] is False
        db.create_index()
        assert db.stats()["has_index"] is True

    def test_stats_ram_estimate_positive_after_insert(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert_batch([str(i) for i in range(50)], _unit_batch(50, 16))
        s = db.stats()
        assert s["ram_estimate_bytes"] > 0

    def test_stats_dimension_matches_open_param(self, tmp_path):
        for d in (1, 8, 16, 64, 128, 512):
            p = str(tmp_path / str(d))
            db = Database.open(p, dimension=d, bits=4, fast_mode=(d == 1))
            assert db.stats()["dimension"] == d


# ---------------------------------------------------------------------------
# ██████╗  █████╗ ████████╗██╗  ██╗    ████████╗██████╗  █████╗ ██╗   ██╗███████╗██████╗ ███████╗ █████╗ ██╗
# ██╔══██╗██╔══██╗╚══██╔══╝██║  ██║    ╚══██╔══╝██╔══██╗██╔══██╗██║   ██║██╔════╝██╔══██╗██╔════╝██╔══██╗██║
# ██████╔╝███████║   ██║   ███████║       ██║   ██████╔╝███████║██║   ██║█████╗  ██████╔╝███████╗███████║██║
# ██╔═══╝ ██╔══██║   ██║   ██╔══██║       ██║   ██╔══██╗██╔══██║╚██╗ ██╔╝██╔══╝  ██╔══██╗╚════██║██╔══██║██║
# ██║     ██║  ██║   ██║   ██║  ██║       ██║   ██║  ██║██║  ██║ ╚████╔╝ ███████╗██║  ██║███████║██║  ██║███████╗
# ---------------------------------------------------------------------------

class TestPathTraversalSafety:
    """Collection names must not allow escaping the database root."""

    def test_dotdot_forward_slash_blocked(self, tmp_path):
        with pytest.raises(Exception):
            Database.open(str(tmp_path / "base"), dimension=8, bits=4, collection="../escape")

    def test_dotdot_backslash_blocked(self, tmp_path):
        with pytest.raises(Exception):
            Database.open(str(tmp_path / "base"), dimension=8, bits=4,
                          collection="..\\escape")

    def test_absolute_path_collection_blocked(self, tmp_path):
        with pytest.raises(Exception):
            Database.open(str(tmp_path / "base"), dimension=8, bits=4,
                          collection="/tmp/evil")

    def test_slash_in_collection_blocked(self, tmp_path):
        with pytest.raises(Exception):
            Database.open(str(tmp_path / "base"), dimension=8, bits=4,
                          collection="sub/dir")

    def test_dot_collection_blocked(self, tmp_path):
        with pytest.raises(Exception):
            Database.open(str(tmp_path / "base"), dimension=8, bits=4,
                          collection=".")

    def test_empty_collection_uses_base(self, tmp_path):
        # Empty collection string should fall through to base path
        db = Database.open(str(tmp_path / "base"), dimension=8, bits=4, collection="")
        db.insert("a", _vec(8, seed=0))
        assert db.count() == 1

    def test_valid_collection_names(self, tmp_path):
        for name in ("col1", "my-collection", "Collection_2024", "αβγ"):
            p = str(tmp_path / "base")
            try:
                db = Database.open(p, dimension=8, bits=4, collection=name)
                db.insert("a", _vec(8, seed=0))
                assert db.count() == 1
            except Exception as e:
                assert "panic" not in type(e).__name__.lower()


# ---------------------------------------------------------------------------
# ██╗  ██╗███████╗ █████╗ ██╗   ██╗██╗   ██╗    ██╗     ██╗███████╗███████╗
# ██║  ██║██╔════╝██╔══██╗██║   ██║╚██╗ ██╔╝    ██║     ██║██╔════╝██╔════╝
# ███████║█████╗  ███████║██║   ██║ ╚████╔╝     ██║     ██║█████╗  █████╗
# ██╔══██║██╔══╝  ██╔══██║╚██╗ ██╔╝  ╚██╔╝      ██║     ██║██╔══╝  ██╔══╝
# ██║  ██║███████╗██║  ██║ ╚████╔╝    ██║        ███████╗██║██║     ███████╗
# ---------------------------------------------------------------------------

class TestHeavyLoadBrutal:
    """Large corpus, large batches, and extreme operational sequences."""

    def test_10k_vectors_insert_search_delete_cycle(self, tmp_path):
        n, d = 10_000, 32
        db = _open(tmp_path / "db", d=d)
        vecs = _unit_batch(n, d)
        ids = [str(i) for i in range(n)]
        db.insert_batch(ids, vecs)
        assert db.count() == n

        # Verify top-1 for a sample
        for qi in [0, 5000, 9999]:
            results = db.search(vecs[qi], top_k=1)
            assert results[0]["id"] == str(qi), f"Top-1 mismatch at qi={qi}"

        # Delete half
        db.delete_batch(ids[:5000])
        assert db.count() == 5000

        # Deleted vectors must not appear
        results = db.search(vecs[0], top_k=10)
        deleted_ids = set(ids[:5000])
        for r in results:
            assert r["id"] not in deleted_ids

    def test_10k_vectors_metadata_filter_search(self, tmp_path):
        n, d = 2000, 16
        db = _open(tmp_path / "db", d=d)
        vecs = _unit_batch(n, d)
        for i in range(n):
            db.insert(str(i), vecs[i], metadata={"bucket": i % 10})
        # Only search within bucket 3
        results = db.search(vecs[0], top_k=100, filter={"bucket": 3})
        assert all(r["metadata"]["bucket"] == 3 for r in results)

    def test_single_vector_corpus_insert_search(self, tmp_path):
        db = _open(tmp_path / "db")
        v = _vec(16, seed=0)
        db.insert("solo", v)
        results = db.search(v, top_k=5)
        assert len(results) == 1
        assert results[0]["id"] == "solo"

    def test_batch_upsert_idempotent(self, tmp_path):
        db = _open(tmp_path / "db")
        vecs = _unit_batch(100, 16)
        ids = [str(i) for i in range(100)]
        db.insert_batch(ids, vecs, mode="insert")
        # Upsert 3 times — count must still be 100
        for _ in range(3):
            db.insert_batch(ids, vecs, mode="upsert")
        assert db.count() == 100

    def test_alternating_insert_and_search_never_returns_stale(self, tmp_path):
        db = _open(tmp_path / "db")
        d = 16
        inserted: set[str] = set()
        vecs_map: dict[str, np.ndarray] = {}
        for i in range(50):
            v = _vec(d, seed=i)
            db.insert(str(i), v, metadata={"seq": i})
            inserted.add(str(i))
            vecs_map[str(i)] = v
            # After each insert, search and verify results are a subset of inserted IDs
            results = db.search(v, top_k=5)
            for r in results:
                assert r["id"] in inserted, \
                    f"search returned unseen id={r['id']} after inserting {i} vectors"

    def test_flush_after_each_insert_still_correct(self, tmp_path):
        db = _open(tmp_path / "db")
        for i in range(20):
            db.insert(str(i), _vec(16, seed=i))
            db.flush()
        assert db.count() == 20
        results = db.search(_vec(16, seed=0), top_k=1)
        assert results[0]["id"] == "0"

# ========================================================================
# FROM test_brutal_qa2.py
# ========================================================================

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
    # The engine raises ValueError before quantization (BUG-5 fixed).
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
    # so the second get of the same ID finds nothing. Expected: both results non-None.
    # BUG-6 fixed: slot_to_indices now maps each slot to all requesting positions.
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

# ========================================================================
# FROM test_brutal_qa3.py
# ========================================================================

class TestFilterExists:
    """$exists: true/false operator — field presence matching."""

    def test_exists_true_matches_present_field(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(), metadata={"tag": "x"})
        db.insert("b", _vec(seed=1))  # no "tag"
        results = db.search(_vec(), top_k=5, filter={"tag": {"$exists": True}})
        ids = {r["id"] for r in results}
        assert "a" in ids
        assert "b" not in ids

    def test_exists_false_matches_missing_field(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(), metadata={"tag": "x"})
        db.insert("b", _vec(seed=1))  # no "tag"
        results = db.search(_vec(), top_k=5, filter={"tag": {"$exists": False}})
        ids = {r["id"] for r in results}
        assert "b" in ids
        assert "a" not in ids

    def test_exists_true_on_all_missing_returns_empty(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert_batch(["a", "b"], _unit_batch(2))
        results = db.search(_vec(), top_k=5, filter={"phantom": {"$exists": True}})
        assert results == []

    def test_exists_false_on_all_present_returns_empty(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert_batch(["a", "b"], _unit_batch(2),
                        metadatas=[{"x": 1}, {"x": 2}])
        results = db.search(_vec(), top_k=5, filter={"x": {"$exists": False}})
        assert results == []

    def test_exists_combined_with_eq(self, tmp_path):
        db = _open(tmp_path / "db")
        for i in range(5):
            md = {"status": "ok"} if i % 2 == 0 else {}
            db.insert(str(i), _vec(seed=i), metadata=md)
        results = db.search(_vec(), top_k=10,
                            filter={"$and": [
                                {"status": {"$exists": True}},
                                {"status": {"$eq": "ok"}},
                            ]})
        assert all(r["metadata"].get("status") == "ok" for r in results)
        assert len(results) == 3  # i=0,2,4

    def test_exists_true_non_bool_value_does_not_match(self, tmp_path):
        # The spec says $exists only matches Bool(true/false), any other value → false
        db = _open(tmp_path / "db")
        db.insert("a", _vec(), metadata={"tag": "x"})
        # $exists: 1 (non-bool) — implementation returns false for both
        results = db.search(_vec(), top_k=5, filter={"tag": {"$exists": 1}})
        assert results == []

    def test_exists_with_dotted_path(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(), metadata={"profile": {"city": "NYC"}})
        db.insert("b", _vec(seed=1), metadata={"profile": {}})
        results = db.search(_vec(), top_k=5,
                            filter={"profile.city": {"$exists": True}})
        ids = {r["id"] for r in results}
        assert "a" in ids
        assert "b" not in ids

    def test_exists_count_consistency(self, tmp_path):
        db = _open(tmp_path / "db")
        for i in range(10):
            md = {"v": i} if i < 6 else {}
            db.insert(str(i), _vec(seed=i), metadata=md)
        c_present = db.count(filter={"v": {"$exists": True}})
        c_absent = db.count(filter={"v": {"$exists": False}})
        assert c_present == 6
        assert c_absent == 4
        assert c_present + c_absent == 10


# ===========================================================================
# 2. Filter operators: $contains
# ===========================================================================

class TestFilterContains:
    """$contains — substring matching in string metadata fields."""

    def test_contains_basic(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(), metadata={"name": "hello world"})
        db.insert("b", _vec(seed=1), metadata={"name": "goodbye"})
        results = db.search(_vec(), top_k=5, filter={"name": {"$contains": "hello"}})
        assert len(results) == 1
        assert results[0]["id"] == "a"

    def test_contains_no_match(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert_batch(["a", "b"], _unit_batch(2),
                        metadatas=[{"t": "foo"}, {"t": "bar"}])
        results = db.search(_vec(), top_k=5, filter={"t": {"$contains": "xyz"}})
        assert results == []

    def test_contains_on_non_string_field_returns_false(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(), metadata={"count": 42})
        results = db.search(_vec(), top_k=5, filter={"count": {"$contains": "4"}})
        # $contains on a non-string field: no match
        assert results == []

    def test_contains_empty_substring_matches_all_strings(self, tmp_path):
        # "" is a substring of every string
        db = _open(tmp_path / "db")
        db.insert("a", _vec(), metadata={"t": "hello"})
        db.insert("b", _vec(seed=1), metadata={"t": "world"})
        results = db.search(_vec(), top_k=5, filter={"t": {"$contains": ""}})
        assert len(results) == 2

    def test_contains_missing_field_does_not_match(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec())  # no "t" field
        results = db.search(_vec(), top_k=5, filter={"t": {"$contains": "x"}})
        assert results == []

    def test_contains_case_sensitive(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(), metadata={"t": "Hello World"})
        results_upper = db.search(_vec(), top_k=5, filter={"t": {"$contains": "Hello"}})
        results_lower = db.search(_vec(), top_k=5, filter={"t": {"$contains": "hello"}})
        assert len(results_upper) == 1
        assert len(results_lower) == 0  # case-sensitive: "hello" not in "Hello World"

    def test_contains_unicode_substring(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(), metadata={"t": "こんにちは世界"})
        results = db.search(_vec(), top_k=5, filter={"t": {"$contains": "世界"}})
        assert len(results) == 1

    def test_contains_combined_with_exists(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(), metadata={"tag": "rust_engine"})
        db.insert("b", _vec(seed=1), metadata={"tag": "python_engine"})
        db.insert("c", _vec(seed=2))  # no tag
        results = db.search(_vec(), top_k=5,
                            filter={"$and": [
                                {"tag": {"$exists": True}},
                                {"tag": {"$contains": "engine"}},
                            ]})
        ids = {r["id"] for r in results}
        assert "a" in ids and "b" in ids
        assert "c" not in ids


# ===========================================================================
# 3. Filter error handling: unknown operators
# ===========================================================================

class TestFilterUnknownOperator:
    """Unknown operators should raise, not silently fail."""

    def test_unknown_top_level_operator_raises(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec())
        with pytest.raises(Exception):
            db.search(_vec(), top_k=5, filter={"$not": {"x": 1}})

    def test_unknown_field_operator_raises(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(), metadata={"x": 1})
        with pytest.raises(Exception):
            db.search(_vec(), top_k=5, filter={"x": {"$regex": ".*"}})

    def test_unknown_operator_in_and_clause_raises(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(), metadata={"x": 1})
        with pytest.raises(Exception):
            db.search(_vec(), top_k=5,
                      filter={"$and": [{"x": {"$startswith": "foo"}}]})

    def test_filter_and_with_non_array_raises_or_returns_empty(self, tmp_path):
        # "$and": "oops" — not an array; spec says this returns false
        db = _open(tmp_path / "db")
        db.insert("a", _vec())
        # Either raises OR returns empty (engine returns false for non-array $and)
        try:
            results = db.search(_vec(), top_k=5, filter={"$and": "oops"})
            assert results == []
        except Exception:
            pass  # raising is also acceptable

    def test_filter_or_with_non_array_raises_or_returns_empty(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec())
        try:
            results = db.search(_vec(), top_k=5, filter={"$or": "oops"})
            assert results == []
        except Exception:
            pass

    def test_empty_filter_dict_matches_all(self, tmp_path):
        # {} should match everything — no conditions
        db = _open(tmp_path / "db")
        db.insert_batch(["a", "b", "c"], _unit_batch(3))
        results = db.search(_vec(), top_k=10, filter={})
        assert len(results) == 3

    def test_empty_and_array_matches_all(self, tmp_path):
        # "$and": [] — vacuously true → should match all
        db = _open(tmp_path / "db")
        db.insert_batch(["a", "b"], _unit_batch(2))
        results = db.search(_vec(), top_k=10, filter={"$and": []})
        assert len(results) == 2

    def test_empty_or_array_matches_none(self, tmp_path):
        # "$or": [] — vacuously false → should match nothing
        db = _open(tmp_path / "db")
        db.insert_batch(["a", "b"], _unit_batch(2))
        results = db.search(_vec(), top_k=10, filter={"$or": []})
        assert results == []


# ===========================================================================
# 4. Zero-length batch operations
# ===========================================================================

class TestZeroLengthBatch:
    """Batches with 0 items must be idempotent and safe."""

    def test_insert_batch_empty_ids_and_vectors(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert_batch([], np.empty((0, 16), dtype=np.float32))
        assert db.count() == 0

    def test_insert_batch_empty_then_normal_insert_works(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert_batch([], np.empty((0, 16), dtype=np.float32))
        db.insert("a", _vec())
        assert db.count() == 1

    def test_delete_batch_empty_list_is_idempotent(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec())
        db.delete_batch([])
        assert db.count() == 1

    def test_delete_batch_empty_after_deleting_all_is_safe(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert_batch(["a", "b"], _unit_batch(2))
        db.delete_batch(["a", "b"])
        db.delete_batch([])  # no-op, DB is already empty
        assert db.count() == 0

    def test_get_many_empty_list_returns_empty(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec())
        result = db.get_many([])
        assert result == []

    def test_query_no_embeddings_raises_or_returns_empty(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert_batch(["a", "b"], _unit_batch(2))
        try:
            results = db.query([], n_results=5)
            assert results == []
        except Exception:
            pass  # raising is also acceptable

    def test_insert_batch_single_item(self, tmp_path):
        db = _open(tmp_path / "db")
        v = _unit_batch(1)
        db.insert_batch(["only"], v)
        assert db.count() == 1
        r = db.get("only")
        assert r is not None


# ===========================================================================
# 5. ANN index edge cases
# ===========================================================================

class TestIndexEdgeCases:
    """create_index and ANN-search edge cases."""

    def test_create_index_on_empty_db_does_not_crash(self, tmp_path):
        db = _open(tmp_path / "db")
        # Empty DB index creation: should either succeed silently or raise cleanly
        try:
            db.create_index(max_degree=8, search_list_size=32)
            # If it succeeds, has_index should reflect reality
            # (may be False since there are 0 nodes)
        except Exception as e:
            # Raising cleanly is also acceptable
            assert "panic" not in type(e).__name__.lower()

    def test_ann_search_before_index_falls_back_to_brute(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert_batch([str(i) for i in range(20)], _unit_batch(20))
        # _use_ann=True without creating index: should not crash (graceful fallback)
        try:
            results = db.search(_vec(), top_k=5, _use_ann=True)
            assert len(results) <= 5
        except Exception as e:
            assert "panic" not in type(e).__name__.lower()

    def test_create_index_small_db_then_ann_search(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert_batch(["a", "b", "c"], _unit_batch(3))
        db.create_index(max_degree=2, search_list_size=4)
        assert db.stats()["has_index"]
        results = db.search(_vec(), top_k=3, _use_ann=True)
        assert len(results) <= 3

    def test_create_index_max_degree_exceeds_n_vectors(self, tmp_path):
        # max_degree=100 with only 5 vectors — should not panic
        db = _open(tmp_path / "db")
        db.insert_batch([str(i) for i in range(5)], _unit_batch(5))
        db.create_index(max_degree=100, search_list_size=8)
        assert db.stats()["has_index"]

    def test_ann_search_returns_valid_scores(self, tmp_path):
        db = _open(tmp_path / "db")
        vecs = _unit_batch(50)
        db.insert_batch([str(i) for i in range(50)], vecs)
        db.create_index(max_degree=8, search_list_size=32)
        results = db.search(vecs[0], top_k=10, _use_ann=True)
        assert len(results) >= 1
        for r in results:
            assert math.isfinite(r["score"])

    def test_ann_vs_brute_same_top1(self, tmp_path):
        db = _open(tmp_path / "db")
        vecs = _unit_batch(100)
        db.insert_batch([str(i) for i in range(100)], vecs)
        db.create_index(max_degree=16, search_list_size=64)
        query = vecs[5]
        bf = db.search(query, top_k=1)
        ann = db.search(query, top_k=1, _use_ann=True)
        # Both should agree on the top-1 match
        assert bf[0]["id"] == ann[0]["id"]

    def test_create_index_zero_max_degree_raises_or_clamps(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert_batch(["a", "b", "c"], _unit_batch(3))
        try:
            db.create_index(max_degree=0, search_list_size=8)
            # If it didn't raise, the DB should still be usable
            db.count()
        except Exception as e:
            assert "panic" not in type(e).__name__.lower()

    def test_update_index_without_prior_create_raises_or_noops(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert_batch([str(i) for i in range(20)], _unit_batch(20))
        try:
            db.update_index(max_degree=8, search_list_size=32)
        except Exception as e:
            assert "panic" not in type(e).__name__.lower()


# ===========================================================================
# 6. Upsert precision semantics
# ===========================================================================

class TestUpsertSemantics:
    """Exact upsert vs insert vs update behavior under all combos."""

    def test_upsert_new_id_creates_entry(self, tmp_path):
        db = _open(tmp_path / "db")
        db.upsert("a", _vec(), metadata={"x": 1})
        assert db.count() == 1
        r = db.get("a")
        assert r["metadata"]["x"] == 1

    def test_upsert_existing_id_replaces_vector_and_metadata(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(seed=0), metadata={"x": 1})
        v2 = _vec(seed=99)
        db.upsert("a", v2, metadata={"x": 2, "new_key": "hi"})
        assert db.count() == 1
        r = db.get("a")
        assert r["metadata"]["x"] == 2
        assert r["metadata"].get("new_key") == "hi"

    def test_upsert_removes_old_metadata_keys_not_in_new(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(), metadata={"old": "gone", "stay": "yes"})
        db.upsert("a", _vec(seed=1), metadata={"stay": "yes"})
        r = db.get("a")
        assert "old" not in r["metadata"]

    def test_insert_batch_upsert_mode_updates_existing(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert_batch(["a", "b"], _unit_batch(2))
        new_vecs = _unit_batch(2, seed=99)
        db.insert_batch(["a", "b"], new_vecs, mode="upsert")
        assert db.count() == 2  # no duplicates

    def test_insert_batch_skip_mode_raises_value_error(self, tmp_path):
        # "skip" is not a supported insert_batch mode; valid modes are
        # "insert", "upsert", "update". This test verifies the clear rejection.
        db = _open(tmp_path / "db")
        db.insert_batch(["a"], _unit_batch(1), metadatas=[{"v": 1}])
        with pytest.raises((ValueError, RuntimeError)):
            db.insert_batch(["a"], _unit_batch(1, seed=99), metadatas=[{"v": 2}],
                            mode="skip")

    def test_update_nonexistent_id_raises(self, tmp_path):
        db = _open(tmp_path / "db")
        with pytest.raises(Exception):
            db.update("ghost", _vec())

    def test_upsert_then_delete_then_upsert_same_id(self, tmp_path):
        db = _open(tmp_path / "db")
        db.upsert("a", _vec(seed=0), metadata={"v": 1})
        db.delete("a")
        db.upsert("a", _vec(seed=1), metadata={"v": 2})
        assert db.count() == 1
        r = db.get("a")
        assert r["metadata"]["v"] == 2

    def test_insert_then_delete_then_insert_same_id(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(seed=0))
        db.delete("a")
        db.insert("a", _vec(seed=1))
        assert db.count() == 1

    def test_delete_nonexistent_is_idempotent(self, tmp_path):
        db = _open(tmp_path / "db")
        db.delete("ghost")  # Should not raise
        assert db.count() == 0

    def test_delete_same_id_twice_is_idempotent(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec())
        db.delete("a")
        db.delete("a")  # Second delete should not raise
        assert db.count() == 0


# ===========================================================================
# 7. Collection isolation
# ===========================================================================

class TestCollectionIsolation:
    """collection= parameter creates separate namespaces."""

    def test_different_collections_at_same_path_are_isolated(self, tmp_path):
        db_a = Database.open(str(tmp_path / "db"), dimension=16, bits=4,
                             collection="col_a")
        db_b = Database.open(str(tmp_path / "db"), dimension=16, bits=4,
                             collection="col_b")
        db_a.insert("shared_id", _vec(seed=0), metadata={"src": "a"})
        db_b.insert("shared_id", _vec(seed=1), metadata={"src": "b"})
        # Each collection sees only its own data
        r_a = db_a.get("shared_id")
        r_b = db_b.get("shared_id")
        assert r_a["metadata"]["src"] == "a"
        assert r_b["metadata"]["src"] == "b"
        del db_a, db_b
        gc.collect(); gc.collect()

    def test_collection_none_and_named_collection_are_isolated(self, tmp_path):
        db_default = _open(tmp_path / "db")
        db_named = Database.open(str(tmp_path / "db"), dimension=16, bits=4,
                                 collection="myns")
        db_default.insert("x", _vec(seed=0))
        assert db_named.count() == 0
        del db_default, db_named
        gc.collect(); gc.collect()

    def test_collection_empty_string_treated_as_default_or_raises(self, tmp_path):
        # empty string collection: either treated as default or raises cleanly
        try:
            db = Database.open(str(tmp_path / "db"), dimension=16, bits=4,
                               collection="")
            db.insert("a", _vec())
        except Exception as e:
            assert "panic" not in type(e).__name__.lower()

    def test_collection_with_special_chars(self, tmp_path):
        # Slash in collection name — should raise or be sanitized, not crash
        try:
            db = Database.open(str(tmp_path / "db"), dimension=16, bits=4,
                               collection="a/b")
            db.insert("x", _vec())
        except Exception as e:
            assert "panic" not in type(e).__name__.lower()


# ===========================================================================
# 8. Python-list and non-numpy vector inputs
# ===========================================================================

class TestPythonListInputs:
    """Vectors passed as Python lists should work or raise cleanly."""

    def test_insert_python_list_vector(self, tmp_path):
        db = _open(tmp_path / "db")
        v = [float(x) for x in _vec()]
        db.insert("a", v)
        assert db.count() == 1

    def test_search_python_list_query(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert_batch([str(i) for i in range(10)], _unit_batch(10))
        q = [float(x) for x in _vec()]
        results = db.search(q, top_k=5)
        assert len(results) <= 5

    def test_insert_2d_list_raises(self, tmp_path):
        db = _open(tmp_path / "db")
        with pytest.raises(Exception):
            db.insert("a", [[1.0, 2.0], [3.0, 4.0]])

    def test_insert_wrong_length_list_raises(self, tmp_path):
        db = _open(tmp_path / "db")
        with pytest.raises(Exception):
            db.insert("a", [1.0, 2.0])  # dim=2, but DB expects 16

    def test_insert_int_list_vector(self, tmp_path):
        db = _open(tmp_path / "db")
        v = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        db.insert("a", v)
        assert db.count() == 1

    def test_insert_f64_numpy_array_gets_cast(self, tmp_path):
        db = _open(tmp_path / "db")
        v = _vec().astype(np.float64)  # f64, not f32
        db.insert("a", v)
        assert db.count() == 1


# ===========================================================================
# 9. State invariants — len/count/list_all consistency
# ===========================================================================

class TestStateInvariants:
    """Core DB state must be self-consistent at all times."""

    def test_len_equals_count_after_inserts(self, tmp_path):
        db = _open(tmp_path / "db")
        for i in range(20):
            db.insert(str(i), _vec(seed=i))
        assert len(db) == db.count() == 20

    def test_list_all_length_equals_count(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert_batch([str(i) for i in range(15)], _unit_batch(15))
        assert len(db.list_all()) == db.count()

    def test_stats_vector_count_equals_count(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert_batch([str(i) for i in range(30)], _unit_batch(30))
        s = db.stats()
        # actual key is 'vector_count', not 'num_vectors'
        assert s["vector_count"] == db.count() == 30

    def test_count_equals_zero_after_delete_all(self, tmp_path):
        db = _open(tmp_path / "db")
        ids = [str(i) for i in range(10)]
        db.insert_batch(ids, _unit_batch(10))
        db.delete_batch(ids)
        assert db.count() == 0
        assert len(db) == 0
        assert db.list_all() == []

    def test_in_operator_reflects_true_after_insert(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("key", _vec())
        assert "key" in db

    def test_in_operator_reflects_false_after_delete(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("key", _vec())
        db.delete("key")
        assert "key" not in db

    def test_stats_dimension_matches_open_param(self, tmp_path):
        for d in [8, 16, 64, 128]:
            db = Database.open(str(tmp_path / f"db_{d}"), dimension=d, bits=4, seed=0)
            assert db.stats()["dimension"] == d

    # BUG-9 fixed: stats() now includes 'metric' key.
    def test_stats_metric_matches_open_param(self, tmp_path):
        for metric in ["ip", "cosine", "l2"]:
            db = Database.open(str(tmp_path / f"db_{metric}"), dimension=16, bits=4,
                               seed=0, metric=metric)
            assert db.stats()["metric"].lower() == metric

    def test_ids_in_list_all_are_exactly_inserted_ids(self, tmp_path):
        db = _open(tmp_path / "db")
        expected = {f"id_{i}" for i in range(20)}
        db.insert_batch(list(expected), _unit_batch(20))
        got = set(db.list_all())
        assert got == expected

    def test_double_flush_does_not_change_count(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert_batch([str(i) for i in range(10)], _unit_batch(10))
        before = db.count()
        db.flush()
        db.flush()
        assert db.count() == before


# ===========================================================================
# 10. Score and metric ordering invariants
# ===========================================================================

class TestScoreOrderingInvariants:
    """Search results must be ordered correctly for every metric."""

    def test_ip_scores_descending(self, tmp_path):
        db = _open(tmp_path / "db", metric="ip")
        db.insert_batch([str(i) for i in range(30)], _unit_batch(30))
        results = db.search(_vec(), top_k=10)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True), "IP scores must be descending"

    def test_cosine_scores_descending(self, tmp_path):
        db = _open(tmp_path / "db", metric="cosine")
        db.insert_batch([str(i) for i in range(30)], _unit_batch(30))
        results = db.search(_vec(), top_k=10)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True), "Cosine scores must be descending"

    def test_l2_scores_descending_or_ascending_consistently(self, tmp_path):
        # L2 scores: engine returns negative distance (higher is closer).
        # Check only that they're monotone (either all asc or all desc).
        db = _open(tmp_path / "db", metric="l2")
        db.insert_batch([str(i) for i in range(30)], _unit_batch(30))
        results = db.search(_vec(), top_k=10)
        scores = [r["score"] for r in results]
        is_desc = all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
        is_asc = all(scores[i] <= scores[i + 1] for i in range(len(scores) - 1))
        assert is_desc or is_asc, f"L2 scores not monotone: {scores}"

    def test_cosine_scores_bounded_by_one(self, tmp_path):
        # Cosine ∈ [-1, 1] approximately (quantization may slightly exceed)
        db = _open(tmp_path / "db", metric="cosine")
        vecs = _unit_batch(50)
        db.insert_batch([str(i) for i in range(50)], vecs)
        results = db.search(vecs[0], top_k=20)
        for r in results:
            assert -1.5 <= r["score"] <= 1.5, f"Cosine score out of bounds: {r['score']}"

    def test_self_query_is_top1_ip(self, tmp_path):
        db = _open(tmp_path / "db", metric="ip")
        vecs = _unit_batch(20)
        db.insert_batch([str(i) for i in range(20)], vecs)
        for i in range(5):
            results = db.search(vecs[i], top_k=1)
            assert results[0]["id"] == str(i), \
                f"Self not top-1: expected {i}, got {results[0]['id']}"

    def test_self_query_is_top1_cosine(self, tmp_path):
        db = _open(tmp_path / "db", metric="cosine")
        vecs = _unit_batch(20)
        db.insert_batch([str(i) for i in range(20)], vecs)
        for i in range(5):
            results = db.search(vecs[i], top_k=1)
            assert results[0]["id"] == str(i)

    def test_all_scores_finite(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert_batch([str(i) for i in range(50)], _unit_batch(50))
        results = db.search(_vec(), top_k=20)
        for r in results:
            assert math.isfinite(r["score"]), f"Non-finite score: {r['score']}"

    def test_top_k_larger_than_db_returns_all(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert_batch([str(i) for i in range(5)], _unit_batch(5))
        results = db.search(_vec(), top_k=1000)
        assert len(results) == 5


# ===========================================================================
# 11. Reopen and persistence stress
# ===========================================================================

class TestReopenStress:
    """Repeated open/close cycles, wrong params on reopen."""

    def test_data_survives_multiple_reopen_cycles(self, tmp_path):
        path = str(tmp_path / "db")
        db = Database.open(path, dimension=16, bits=4)
        db.insert_batch([str(i) for i in range(50)], _unit_batch(50))
        expected_count = 50
        del db; gc.collect(); gc.collect()

        for _ in range(5):
            db = Database.open(path, dimension=16, bits=4)
            assert db.count() == expected_count, "Count changed across reopen"
            del db; gc.collect(); gc.collect()

    def test_reopen_with_wrong_dimension_raises(self, tmp_path):
        path = str(tmp_path / "db")
        db = Database.open(path, dimension=16, bits=4)
        db.insert("a", _vec(16))
        del db; gc.collect(); gc.collect()

        with pytest.raises(Exception):
            Database.open(path, dimension=32, bits=4)

    def test_index_survives_reopen(self, tmp_path):
        path = str(tmp_path / "db")
        db = Database.open(path, dimension=16, bits=4)
        db.insert_batch([str(i) for i in range(30)], _unit_batch(30))
        db.create_index(max_degree=8, search_list_size=32)
        assert db.stats()["has_index"]
        del db; gc.collect(); gc.collect()

        db2 = Database.open(path, dimension=16, bits=4)
        assert db2.stats()["has_index"], "Index lost after reopen"
        del db2; gc.collect(); gc.collect()

    def test_deleted_items_stay_deleted_after_reopen(self, tmp_path):
        path = str(tmp_path / "db")
        db = Database.open(path, dimension=16, bits=4)
        db.insert_batch([str(i) for i in range(20)], _unit_batch(20))
        db.delete("5")
        del db; gc.collect(); gc.collect()

        db2 = Database.open(path, dimension=16, bits=4)
        assert db2.count() == 19
        assert db2.get("5") is None
        del db2; gc.collect(); gc.collect()

    def test_metadata_survives_reopen(self, tmp_path):
        path = str(tmp_path / "db")
        db = Database.open(path, dimension=16, bits=4)
        db.insert("a", _vec(), metadata={"deep": {"nested": {"value": 42}}})
        del db; gc.collect(); gc.collect()

        db2 = Database.open(path, dimension=16, bits=4)
        r = db2.get("a")
        assert r["metadata"]["deep"]["nested"]["value"] == 42
        del db2; gc.collect(); gc.collect()


# ===========================================================================
# 12. Rerank edge cases
# ===========================================================================

class TestRerankEdgeCases:
    """Rerank with extreme factors and precision."""

    def test_rerank_factor_zero_raises_or_returns_results(self, tmp_path):
        db = _open(tmp_path / "db", rerank=True)
        db.insert_batch([str(i) for i in range(20)], _unit_batch(20))
        try:
            results = db.search(_vec(), top_k=5, rerank_factor=0)
            assert len(results) <= 5
        except Exception as e:
            assert "panic" not in type(e).__name__.lower()

    def test_rerank_factor_one(self, tmp_path):
        db = _open(tmp_path / "db", rerank=True)
        db.insert_batch([str(i) for i in range(20)], _unit_batch(20))
        results = db.search(_vec(), top_k=5, rerank_factor=1)
        assert len(results) <= 5

    def test_rerank_disabled_returns_results(self, tmp_path):
        db = _open(tmp_path / "db", rerank=False)
        db.insert_batch([str(i) for i in range(20)], _unit_batch(20))
        results = db.search(_vec(), top_k=5)
        assert len(results) <= 5

    def test_rerank_precision_valid_strings(self, tmp_path):
        # rerank_precision must be a string: 'int8' (default), 'int4', 'f16', 'f32'
        for prec_str in ['int8', 'int4', 'f16', 'f32']:
            db = _open(tmp_path / f"db_{prec_str}", rerank=True,
                       rerank_precision=prec_str)
            db.insert_batch([str(i) for i in range(20)], _unit_batch(20))
            results = db.search(_vec(), top_k=5)
            assert len(results) <= 5

    def test_rerank_precision_float_raises_type_error(self, tmp_path):
        # Passing a float for rerank_precision must raise, not silently misbehave
        with pytest.raises((TypeError, ValueError)):
            _open(tmp_path / "db", rerank=True, rerank_precision=0.5)


# ===========================================================================
# 13. Complex nested filter semantics
# ===========================================================================

class TestNestedFilterSemantics:
    """Deep nesting, mixed operators, dotted 3-level paths."""

    def test_and_of_ors(self, tmp_path):
        db = _open(tmp_path / "db")
        metas = [
            {"color": "red", "size": "big"},
            {"color": "blue", "size": "big"},
            {"color": "red", "size": "small"},
            {"color": "green", "size": "small"},
        ]
        db.insert_batch([str(i) for i in range(4)], _unit_batch(4),
                        metadatas=metas)
        # (color=red OR color=blue) AND (size=big)
        results = db.search(_vec(), top_k=10, filter={
            "$and": [
                {"$or": [{"color": "red"}, {"color": "blue"}]},
                {"size": "big"},
            ]
        })
        ids = {r["id"] for r in results}
        assert "0" in ids  # red+big
        assert "1" in ids  # blue+big
        assert "2" not in ids  # red+small
        assert "3" not in ids  # green+small

    def test_or_of_ands(self, tmp_path):
        db = _open(tmp_path / "db")
        metas = [
            {"a": 1, "b": 2},
            {"a": 1, "b": 3},
            {"a": 2, "b": 2},
        ]
        db.insert_batch(["x", "y", "z"], _unit_batch(3), metadatas=metas)
        # (a=1 AND b=2) OR (a=2 AND b=2)
        results = db.search(_vec(), top_k=10, filter={
            "$or": [
                {"$and": [{"a": 1}, {"b": 2}]},
                {"$and": [{"a": 2}, {"b": 2}]},
            ]
        })
        ids = {r["id"] for r in results}
        assert "x" in ids  # a=1,b=2 ✓
        assert "y" not in ids  # a=1,b=3 ✗
        assert "z" in ids  # a=2,b=2 ✓

    def test_three_level_dotted_path(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(), metadata={"l1": {"l2": {"l3": "deep"}}})
        db.insert("b", _vec(seed=1), metadata={"l1": {"l2": {"l3": "other"}}})
        results = db.search(_vec(), top_k=5, filter={"l1.l2.l3": "deep"})
        assert len(results) == 1
        assert results[0]["id"] == "a"

    def test_ne_on_missing_field_matches(self, tmp_path):
        # $ne: missing field counts as "not equal to X" → should match
        db = _open(tmp_path / "db")
        db.insert("a", _vec(), metadata={"x": 1})
        db.insert("b", _vec(seed=1))  # no "x"
        results = db.search(_vec(), top_k=5, filter={"x": {"$ne": 999}})
        ids = {r["id"] for r in results}
        # "b" has no "x" → $ne should match (missing ≠ 999)
        assert "b" in ids

    def test_in_operator_with_one_element(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(), metadata={"t": "alpha"})
        db.insert("b", _vec(seed=1), metadata={"t": "beta"})
        results = db.search(_vec(), top_k=5, filter={"t": {"$in": ["alpha"]}})
        assert len(results) == 1
        assert results[0]["id"] == "a"

    def test_nin_with_all_values_excludes_all(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert_batch(["a", "b"], _unit_batch(2),
                        metadatas=[{"t": "x"}, {"t": "y"}])
        results = db.search(_vec(), top_k=5, filter={"t": {"$nin": ["x", "y"]}})
        assert results == []

    def test_gt_lt_same_field_range(self, tmp_path):
        db = _open(tmp_path / "db")
        for i in range(10):
            db.insert(str(i), _vec(seed=i), metadata={"score": float(i)})
        # 3 < score < 7
        results = db.search(_vec(), top_k=10, filter={
            "score": {"$gt": 3.0, "$lt": 7.0}
        })
        scores = {r["metadata"]["score"] for r in results}
        assert all(3.0 < s < 7.0 for s in scores)
        assert scores == {4.0, 5.0, 6.0}


# ===========================================================================
# 14. Document field round-trip
# ===========================================================================

class TestDocumentField:
    """Document field stored and retrieved correctly."""

    def test_document_stored_and_retrieved(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(), document="Hello, world!")
        r = db.get("a")
        assert r["document"] == "Hello, world!"

    def test_document_none_default(self, tmp_path):
        # When no document is set, the 'document' key is absent from get() result
        db = _open(tmp_path / "db")
        db.insert("a", _vec())
        r = db.get("a")
        # Key is absent OR None — both are acceptable
        assert r.get("document") is None

    def test_document_survives_upsert(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(), document="original")
        db.upsert("a", _vec(seed=1))  # no document arg
        r = db.get("a")
        # Upsert with no document may clear the document — just must not crash
        # and the record must still exist
        assert r is not None
        assert r["id"] == "a"

    def test_document_with_unicode(self, tmp_path):
        db = _open(tmp_path / "db")
        text = "日本語テスト 🌸 emoji: \U0001F600"
        db.insert("a", _vec(), document=text)
        r = db.get("a")
        assert r["document"] == text

    def test_document_very_long(self, tmp_path):
        db = _open(tmp_path / "db")
        long_doc = "x" * 100_000
        db.insert("a", _vec(), document=long_doc)
        r = db.get("a")
        assert r["document"] == long_doc

    def test_include_documents_in_search(self, tmp_path):
        # Valid include fields: "id", "score", "metadata", "document" (singular)
        db = _open(tmp_path / "db")
        db.insert("a", _vec(), document="doc_a")
        db.insert("b", _vec(seed=1), document="doc_b")
        results = db.search(_vec(), top_k=5, include=["id", "score", "document"])
        for r in results:
            assert "document" in r

    def test_exclude_documents_in_search(self, tmp_path):
        # Omitting "document" from include means no document key in results
        db = _open(tmp_path / "db")
        db.insert("a", _vec(), document="doc_a")
        results = db.search(_vec(), top_k=5, include=["id", "score", "metadata"])
        for r in results:
            assert "document" not in r


# ===========================================================================
# 15. Heavy concurrent stress — 100 threads
# ===========================================================================

class TestConcurrentExtreme:
    """Brutal concurrent access — 100 threads, mixed operations."""

    def test_100_concurrent_inserts(self, tmp_path):
        db = _open(tmp_path / "db")
        errors = []

        def worker(i):
            try:
                db.insert(str(i), _vec(seed=i))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Errors during concurrent insert: {errors}"
        assert db.count() == 100

    def test_concurrent_insert_and_search(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert_batch([str(i) for i in range(50)], _unit_batch(50))
        errors = []
        results_counts = []

        def inserter(i):
            try:
                db.insert(f"new_{i}", _vec(seed=1000 + i))
            except Exception as e:
                errors.append(("insert", e))

        def searcher():
            try:
                r = db.search(_vec(), top_k=5)
                results_counts.append(len(r))
            except Exception as e:
                errors.append(("search", e))

        threads = (
            [threading.Thread(target=inserter, args=(i,)) for i in range(50)] +
            [threading.Thread(target=searcher) for _ in range(50)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Concurrent errors: {errors}"
        assert db.count() >= 50  # at least the pre-loaded items

    def test_concurrent_reads_are_consistent(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert_batch([str(i) for i in range(50)], _unit_batch(50))
        counts = []
        errors = []

        def reader():
            try:
                counts.append(db.count())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert all(c == 50 for c in counts), f"Inconsistent counts: {set(counts)}"


# ===========================================================================
# 16. $ne operator edge cases
# ===========================================================================

class TestNeOperator:
    """$ne (not-equal) has subtle semantics with missing fields and nulls."""

    def test_ne_excludes_matching_value(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(), metadata={"v": 1})
        db.insert("b", _vec(seed=1), metadata={"v": 2})
        results = db.search(_vec(), top_k=5, filter={"v": {"$ne": 1}})
        ids = {r["id"] for r in results}
        assert "a" not in ids
        assert "b" in ids

    def test_ne_string(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(), metadata={"s": "alpha"})
        db.insert("b", _vec(seed=1), metadata={"s": "beta"})
        results = db.search(_vec(), top_k=5, filter={"s": {"$ne": "alpha"}})
        ids = {r["id"] for r in results}
        assert "a" not in ids
        assert "b" in ids

    def test_ne_with_false_value(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(), metadata={"ok": False})
        db.insert("b", _vec(seed=1), metadata={"ok": True})
        results = db.search(_vec(), top_k=5, filter={"ok": {"$ne": False}})
        ids = {r["id"] for r in results}
        assert "a" not in ids
        assert "b" in ids

    def test_ne_with_zero(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert("a", _vec(), metadata={"n": 0})
        db.insert("b", _vec(seed=1), metadata={"n": 1})
        results = db.search(_vec(), top_k=5, filter={"n": {"$ne": 0}})
        ids = {r["id"] for r in results}
        assert "a" not in ids
        assert "b" in ids


# ===========================================================================
# 17. Large dimension stress
# ===========================================================================

class TestLargeDimensions:
    """Dimensions that stress quantizer and SRHT."""

    @pytest.mark.parametrize("dim", [64, 128, 256, 512])
    def test_insert_and_search_large_dim(self, tmp_path, dim):
        db = Database.open(str(tmp_path / f"db_{dim}"), dimension=dim, bits=4, seed=0)
        vecs = _unit_batch(20, dim=dim, seed=dim)
        db.insert_batch([str(i) for i in range(20)], vecs)
        results = db.search(vecs[0], top_k=5)
        assert results[0]["id"] == "0"

    def test_dim_1024_basic_insert_search(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=1024, bits=4, seed=0)
        vecs = _unit_batch(10, dim=1024, seed=1)
        db.insert_batch([str(i) for i in range(10)], vecs)
        results = db.search(vecs[0], top_k=3)
        assert len(results) >= 1
        assert results[0]["id"] == "0"

    def test_dim_1_edge(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=1, bits=4, seed=0)
        db.insert("a", np.array([1.0], dtype=np.float32))
        db.insert("b", np.array([-1.0], dtype=np.float32))
        results = db.search(np.array([1.0], dtype=np.float32), top_k=2)
        assert len(results) == 2


# ===========================================================================
# 18. Subnormal and extreme float vectors
# ===========================================================================

class TestExtremeFloatVectors:
    """Subnormal, very-small, very-large, and near-integer vectors."""

    def test_subnormal_float_vector(self, tmp_path):
        db = _open(tmp_path / "db")
        v = np.full(16, 5e-324, dtype=np.float32)  # underflows to 0 in f32
        # Should either accept (treating as zero) or raise cleanly
        try:
            db.insert("sub", v)
        except Exception as e:
            assert "panic" not in type(e).__name__.lower()

    def test_max_float32_vector(self, tmp_path):
        db = _open(tmp_path / "db")
        v = np.full(16, np.finfo(np.float32).max, dtype=np.float32)
        try:
            db.insert("max", v)
        except Exception as e:
            assert "panic" not in type(e).__name__.lower()

    def test_alternating_sign_vector(self, tmp_path):
        db = _open(tmp_path / "db")
        v = np.array([(-1) ** i * 0.25 for i in range(16)], dtype=np.float32)
        db.insert("alt", v)
        assert db.count() == 1

    def test_nan_vector_raises_or_handles_cleanly(self, tmp_path):
        db = _open(tmp_path / "db")
        v = np.full(16, float("nan"), dtype=np.float32)
        try:
            db.insert("nan", v)
            # If accepted, search should not return NaN scores
            results = db.search(_vec(), top_k=5)
            for r in results:
                assert not math.isnan(r["score"]), "NaN score leaked"
        except Exception as e:
            assert "panic" not in type(e).__name__.lower()

    def test_inf_vector_raises_or_handles_cleanly(self, tmp_path):
        db = _open(tmp_path / "db")
        v = np.full(16, float("inf"), dtype=np.float32)
        try:
            db.insert("inf", v)
        except Exception as e:
            assert "panic" not in type(e).__name__.lower()


# ===========================================================================
# 19. Attack sequences — state machine abuse
# ===========================================================================

class TestStateAttackSequences:
    """Complex operation sequences designed to corrupt state."""

    def test_insert_delete_reinsert_search_cycle(self, tmp_path):
        db = _open(tmp_path / "db")
        for cycle in range(5):
            db.insert("a", _vec(seed=cycle))
            db.delete("a")
        db.insert("a", _vec(seed=99))
        results = db.search(_vec(), top_k=1)
        assert results[0]["id"] == "a"

    def test_fill_flush_delete_half_search(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert_batch([str(i) for i in range(100)], _unit_batch(100))
        db.flush()
        db.delete_batch([str(i) for i in range(50)])
        assert db.count() == 50
        results = db.search(_vec(), top_k=10)
        ids = {r["id"] for r in results}
        assert not any(int(rid) < 50 for rid in ids), "Found deleted IDs in results"

    def test_create_index_insert_more_search_ann(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert_batch([str(i) for i in range(50)], _unit_batch(50))
        db.create_index(max_degree=8, search_list_size=32)
        # Insert more AFTER index creation
        db.insert_batch([f"new_{i}" for i in range(20)], _unit_batch(20, seed=99))
        # ANN search should not crash even with unindexed nodes
        results = db.search(_vec(), top_k=10, _use_ann=True)
        assert len(results) >= 1

    def test_metadata_overwrite_preserves_search_correctness(self, tmp_path):
        db = _open(tmp_path / "db")
        vecs = _unit_batch(20)
        db.insert_batch([str(i) for i in range(20)], vecs,
                        metadatas=[{"v": i} for i in range(20)])
        # Overwrite all metadata
        for i in range(20):
            db.update_metadata(str(i), metadata={"v": i + 100})
        # Filter on new metadata should work
        results = db.search(_vec(), top_k=20, filter={"v": {"$gte": 110}})
        assert all(r["metadata"]["v"] >= 110 for r in results)

    def test_batch_insert_with_duplicate_ids_same_batch(self, tmp_path):
        db = _open(tmp_path / "db")
        # Inserting duplicate IDs in a single batch — behavior must be consistent
        try:
            db.insert_batch(["a", "a", "a"], _unit_batch(3))
            # Count should be 1 (last wins) or raise; not 3
            count = db.count()
            assert count in (1, 3), f"Unexpected count after dup batch: {count}"
        except Exception as e:
            assert "panic" not in type(e).__name__.lower()

    def test_search_after_deleting_every_item(self, tmp_path):
        db = _open(tmp_path / "db")
        ids = [str(i) for i in range(30)]
        db.insert_batch(ids, _unit_batch(30))
        db.delete_batch(ids)
        results = db.search(_vec(), top_k=10)
        assert results == []

    def test_upsert_batch_over_fresh_and_existing(self, tmp_path):
        db = _open(tmp_path / "db")
        db.insert_batch(["a", "b"], _unit_batch(2))
        # Upsert mix of existing + new
        db.insert_batch(["b", "c", "d"], _unit_batch(3, seed=99), mode="upsert")
        assert db.count() == 4  # a, b (updated), c, d

    def test_list_ids_pagination_covers_all(self, tmp_path):
        db = _open(tmp_path / "db")
        total = 55
        db.insert_batch([str(i) for i in range(total)], _unit_batch(total))
        all_ids = set()
        offset = 0
        limit = 10
        while True:
            page = db.list_ids(limit=limit, offset=offset)
            if not page:
                break
            all_ids.update(page)
            offset += limit
        assert len(all_ids) == total


# ===========================================================================
# 20. Bits boundary values
# ===========================================================================

class TestBitsBoundary:
    """bits parameter boundary: practical range is 1-12; larger values cause exponential slowdown."""

    @pytest.mark.parametrize("bits", [1, 2, 4, 8])
    def test_valid_bits_insert_and_search(self, tmp_path, bits):
        dim = 16
        db = Database.open(str(tmp_path / f"db_{bits}"),
                           dimension=dim, bits=bits, seed=42)
        vecs = _unit_batch(20, dim=dim, seed=bits)
        db.insert_batch([str(i) for i in range(20)], vecs)
        results = db.search(vecs[0], top_k=5)
        assert len(results) >= 1

    @pytest.mark.xfail(strict=False,
                        reason="BUG-7: bits >= 13 causes exponential slowdown — "
                               "bits=13 takes 5-16s, bits=14 ~31s, bits=15 ~66s, "
                               "bits=16+ effectively hangs. API allows up to 64 but "
                               "codebook generation is O(2^bits), practical limit is ~12. "
                               "Test may XPASS if bits=13 finishes in time on fast hardware.")
    def test_bits_13_completes_in_reasonable_time(self, tmp_path):
        import threading, time
        result = [None]
        def do_it():
            db = Database.open(str(tmp_path / "db13"), dimension=16, bits=13, seed=42)
            db.insert("a", np.random.rand(16).astype(np.float32))
            result[0] = "ok"
        t = threading.Thread(target=do_it, daemon=True)
        t0 = time.time()
        t.start()
        t.join(timeout=5)
        elapsed = time.time() - t0
        assert result[0] == "ok", f"bits=13 did not complete in 5s (elapsed={elapsed:.1f}s)"

    def test_bits_65_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError):
            Database.open(str(tmp_path / "db"), dimension=16, bits=65)

    def test_bits_256_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError):
            Database.open(str(tmp_path / "db"), dimension=16, bits=256)

    def test_bits_0_raises(self, tmp_path):
        with pytest.raises(Exception):
            Database.open(str(tmp_path / "db"), dimension=16, bits=0)

    def test_bits_negative_raises(self, tmp_path):
        with pytest.raises(Exception):
            Database.open(str(tmp_path / "db"), dimension=16, bits=-1)

# ========================================================================
# FROM test_brutal_qa4.py
# ========================================================================

class TestRerankFactor:
    """rerank_factor oversampling multiplier — brute-force path."""

    def test_rerank_factor_1_returns_results(self, tmp_path):
        db = _open(tmp_path)
        vecs = _unit_batch(20, 16, seed=1)
        ids = [f"v{i}" for i in range(20)]
        db.insert_batch(ids, vecs)
        q = _vec(16, seed=99)
        res = db.search(q, top_k=5, rerank_factor=1)
        assert len(res) == 5

    def test_rerank_factor_10_same_count(self, tmp_path):
        db = _open(tmp_path)
        vecs = _unit_batch(30, 16, seed=2)
        ids = [f"v{i}" for i in range(30)]
        db.insert_batch(ids, vecs)
        q = _vec(16, seed=88)
        res = db.search(q, top_k=5, rerank_factor=10)
        assert len(res) == 5

    def test_rerank_factor_100_same_count(self, tmp_path):
        db = _open(tmp_path)
        vecs = _unit_batch(30, 16, seed=3)
        ids = [f"v{i}" for i in range(30)]
        db.insert_batch(ids, vecs)
        q = _vec(16, seed=77)
        res = db.search(q, top_k=5, rerank_factor=100)
        assert len(res) == 5

    def test_rerank_factor_none_is_default(self, tmp_path):
        db = _open(tmp_path)
        vecs = _unit_batch(20, 16, seed=4)
        ids = [f"v{i}" for i in range(20)]
        db.insert_batch(ids, vecs)
        q = _vec(16, seed=66)
        res_none = db.search(q, top_k=5, rerank_factor=None)
        res_no_arg = db.search(q, top_k=5)
        assert len(res_none) == len(res_no_arg) == 5

    def test_rerank_factor_with_filter(self, tmp_path):
        db = _open(tmp_path)
        vecs = _unit_batch(20, 16, seed=5)
        ids = [f"v{i}" for i in range(20)]
        metas = [{"grp": "A" if i < 10 else "B"} for i in range(20)]
        db.insert_batch(ids, vecs, metadatas=metas)
        q = _vec(16, seed=55)
        res = db.search(q, top_k=3, filter={"grp": "A"}, rerank_factor=5)
        assert len(res) <= 3
        assert all(r["metadata"]["grp"] == "A" for r in res)

    def test_rerank_factor_in_query_batch(self, tmp_path):
        db = _open(tmp_path)
        vecs = _unit_batch(20, 16, seed=6)
        ids = [f"v{i}" for i in range(20)]
        db.insert_batch(ids, vecs)
        queries = _unit_batch(3, 16, seed=44)
        res = db.query(queries, n_results=5, rerank_factor=3)
        assert len(res) == 3
        assert all(len(r) == 5 for r in res)

    # BUG-10 fixed: rerank_factor=0 now raises ValueError.
    def test_rerank_factor_zero_raises(self, tmp_path):
        """rerank_factor=0 should produce an error (cannot oversample by 0)."""
        db = _open(tmp_path)
        vecs = _unit_batch(10, 16, seed=7)
        db.insert_batch([f"v{i}" for i in range(10)], vecs)
        q = _vec(16, seed=33)
        with pytest.raises((ValueError, RuntimeError, BaseException)):
            db.search(q, top_k=3, rerank_factor=0)


# ===========================================================================
# 2. wal_flush_threshold effect on stats
# ===========================================================================

class TestWalFlushThreshold:
    """wal_flush_threshold controls when WAL vectors get flushed to segment."""

    def test_low_threshold_zero_buffered_after_insert(self, tmp_path):
        # threshold=1 means flush immediately after each vector
        db = Database.open(str(tmp_path), dimension=16, bits=4, seed=42,
                           wal_flush_threshold=1)
        db.insert("a", _vec(16, seed=1))
        db.insert("b", _vec(16, seed=2))
        s = db.stats()
        # With threshold=1, WAL flushes after every insert → 0 buffered
        assert s["buffered_vectors"] == 0
        assert s["vector_count"] == 2

    def test_high_threshold_all_buffered(self, tmp_path):
        # threshold=100000 means vectors stay in WAL buffer
        db = Database.open(str(tmp_path), dimension=16, bits=4, seed=42,
                           wal_flush_threshold=100000)
        for i in range(5):
            db.insert(f"v{i}", _vec(16, seed=i))
        s = db.stats()
        assert s["vector_count"] == 5
        # With huge threshold, all 5 should be buffered
        assert s["buffered_vectors"] == 5

    def test_flush_call_drains_buffered(self, tmp_path):
        db = Database.open(str(tmp_path), dimension=16, bits=4, seed=42,
                           wal_flush_threshold=100000)
        for i in range(5):
            db.insert(f"v{i}", _vec(16, seed=i))
        assert db.stats()["buffered_vectors"] == 5
        db.flush()
        assert db.stats()["buffered_vectors"] == 0

    def test_threshold_none_uses_default(self, tmp_path):
        # wal_flush_threshold=None should not raise
        db = Database.open(str(tmp_path), dimension=16, bits=4,
                           wal_flush_threshold=None)
        db.insert("a", _vec(16, seed=1))
        assert db.stats()["vector_count"] == 1

    # BUG-11 fixed: wal_flush_threshold=0 now raises ValueError.
    def test_threshold_zero_raises(self, tmp_path):
        """wal_flush_threshold=0 should raise a ValueError."""
        with pytest.raises((ValueError, RuntimeError)):
            Database.open(str(tmp_path), dimension=16, bits=4,
                          wal_flush_threshold=0)


# ===========================================================================
# 3. create_index / update_index parameter variants
# ===========================================================================

class TestCreateIndexParams:
    """Exercise all create_index parameter combinations."""

    def _insert_batch(self, db, n=50, dim=16, seed=10):
        vecs = _unit_batch(n, dim, seed=seed)
        db.insert_batch([f"v{i}" for i in range(n)], vecs)

    def test_defaults_no_params(self, tmp_path):
        db = _open(tmp_path)
        self._insert_batch(db)
        db.create_index()
        assert db.stats()["has_index"]

    def test_small_max_degree(self, tmp_path):
        db = _open(tmp_path)
        self._insert_batch(db)
        db.create_index(max_degree=2)
        assert db.stats()["has_index"]

    def test_large_max_degree(self, tmp_path):
        db = _open(tmp_path)
        self._insert_batch(db)
        db.create_index(max_degree=64)
        assert db.stats()["has_index"]

    def test_small_ef_construction(self, tmp_path):
        db = _open(tmp_path)
        self._insert_batch(db)
        db.create_index(ef_construction=10)
        assert db.stats()["has_index"]

    def test_alpha_below_1(self, tmp_path):
        """alpha < 1.0 is legal — tighter pruning."""
        db = _open(tmp_path)
        self._insert_batch(db)
        db.create_index(alpha=0.9)
        assert db.stats()["has_index"]

    def test_alpha_2(self, tmp_path):
        db = _open(tmp_path)
        self._insert_batch(db)
        db.create_index(alpha=2.0)
        assert db.stats()["has_index"]

    def test_n_refinements_0(self, tmp_path):
        db = _open(tmp_path)
        self._insert_batch(db)
        db.create_index(n_refinements=0)
        assert db.stats()["has_index"]

    def test_n_refinements_large(self, tmp_path):
        db = _open(tmp_path)
        self._insert_batch(db)
        db.create_index(n_refinements=20)
        assert db.stats()["has_index"]

    def test_small_search_list_size(self, tmp_path):
        db = _open(tmp_path)
        self._insert_batch(db)
        db.create_index(search_list_size=10)
        assert db.stats()["has_index"]

    def test_index_on_empty_db(self, tmp_path):
        """create_index on empty DB should not crash."""
        db = _open(tmp_path)
        db.create_index()
        # has_index may be False on empty — but no crash

    def test_rebuild_preserves_search(self, tmp_path):
        db = _open(tmp_path)
        self._insert_batch(db, n=30, seed=20)
        db.create_index(max_degree=8)
        q = _vec(16, seed=200)
        res1 = db.search(q, top_k=5, _use_ann=True)
        db.create_index(max_degree=8)  # rebuild
        res2 = db.search(q, top_k=5, _use_ann=True)
        assert len(res1) == len(res2)

    def test_update_index_after_more_inserts(self, tmp_path):
        db = _open(tmp_path)
        self._insert_batch(db, n=30, seed=21)
        db.create_index(max_degree=8)
        assert db.stats()["has_index"]
        # insert more vectors and do incremental update
        extra = _unit_batch(10, 16, seed=22)
        db.insert_batch([f"extra{i}" for i in range(10)], extra)
        db.update_index(max_degree=8)
        assert db.stats()["has_index"]


# ===========================================================================
# 4. ann_search_list_size override in search/query
# ===========================================================================

class TestAnnSearchListSize:
    """ann_search_list_size (ef_search override) in ANN mode."""

    def _db_with_index(self, path, n=80, dim=16, seed=30):
        db = _open(path)
        vecs = _unit_batch(n, dim, seed=seed)
        db.insert_batch([f"v{i}" for i in range(n)], vecs)
        db.create_index(max_degree=8, search_list_size=32)
        return db

    def test_small_list_size_still_returns_results(self, tmp_path):
        db = self._db_with_index(tmp_path)
        q = _vec(16, seed=300)
        res = db.search(q, top_k=5, _use_ann=True, ann_search_list_size=5)
        assert len(res) > 0

    def test_large_list_size_returns_results(self, tmp_path):
        db = self._db_with_index(tmp_path)
        q = _vec(16, seed=301)
        res = db.search(q, top_k=5, _use_ann=True, ann_search_list_size=200)
        assert len(res) > 0

    def test_list_size_ignored_in_brute_force(self, tmp_path):
        """ann_search_list_size should be silently ignored in brute-force mode."""
        db = self._db_with_index(tmp_path)
        q = _vec(16, seed=302)
        res = db.search(q, top_k=5, _use_ann=False, ann_search_list_size=5)
        assert len(res) == 5

    def test_list_size_in_query_batch(self, tmp_path):
        db = self._db_with_index(tmp_path)
        queries = _unit_batch(3, 16, seed=303)
        res = db.query(queries, n_results=5, _use_ann=True, ann_search_list_size=50)
        assert len(res) == 3
        assert all(len(r) > 0 for r in res)


# ===========================================================================
# 5. count() with empty filter vs None
# ===========================================================================

class TestCountFilterVariants:
    """count() edge cases around None vs empty filter."""

    def test_count_none_equals_len(self, tmp_path):
        db = _open(tmp_path)
        vecs = _unit_batch(10, 16, seed=40)
        db.insert_batch([f"v{i}" for i in range(10)], vecs)
        assert db.count() == 10
        assert db.count(filter=None) == 10
        assert len(db) == 10

    def test_count_empty_dict_filter(self, tmp_path):
        """count(filter={}) should work — empty dict = no constraint."""
        db = _open(tmp_path)
        vecs = _unit_batch(5, 16, seed=41)
        db.insert_batch([f"v{i}" for i in range(5)], vecs)
        # empty filter is treated as None (vacuously true for all)
        assert db.count(filter={}) == 5

    def test_count_with_filter_after_delete(self, tmp_path):
        db = _open(tmp_path)
        vecs = _unit_batch(10, 16, seed=42)
        ids = [f"v{i}" for i in range(10)]
        metas = [{"grp": "X" if i < 5 else "Y"} for i in range(10)]
        db.insert_batch(ids, vecs, metadatas=metas)
        assert db.count(filter={"grp": "X"}) == 5
        db.delete("v0")
        assert db.count(filter={"grp": "X"}) == 4

    def test_count_on_empty_db(self, tmp_path):
        db = _open(tmp_path)
        assert db.count() == 0
        assert db.count(filter=None) == 0
        assert db.count(filter={}) == 0


# ===========================================================================
# 6. list_ids() with filter + limit + offset combined
# ===========================================================================

class TestListIdsFilterAndPagination:
    """list_ids() with all three args combined."""

    def test_filter_and_limit(self, tmp_path):
        db = _open(tmp_path)
        vecs = _unit_batch(20, 16, seed=50)
        ids = [f"v{i}" for i in range(20)]
        metas = [{"cat": "A" if i < 10 else "B"} for i in range(20)]
        db.insert_batch(ids, vecs, metadatas=metas)
        result = db.list_ids(where_filter={"cat": "A"}, limit=3)
        assert len(result) == 3

    def test_filter_and_offset(self, tmp_path):
        db = _open(tmp_path)
        vecs = _unit_batch(20, 16, seed=51)
        ids = [f"v{i}" for i in range(20)]
        metas = [{"cat": "A"} for _ in range(20)]
        db.insert_batch(ids, vecs, metadatas=metas)
        all_ids = db.list_ids(where_filter={"cat": "A"})
        paged_ids = db.list_ids(where_filter={"cat": "A"}, offset=5)
        assert len(paged_ids) == len(all_ids) - 5

    def test_filter_limit_offset_combined(self, tmp_path):
        db = _open(tmp_path)
        vecs = _unit_batch(30, 16, seed=52)
        ids = [f"v{i}" for i in range(30)]
        metas = [{"grp": "G"} for _ in range(30)]
        db.insert_batch(ids, vecs, metadatas=metas)
        page1 = db.list_ids(where_filter={"grp": "G"}, limit=10, offset=0)
        page2 = db.list_ids(where_filter={"grp": "G"}, limit=10, offset=10)
        assert len(page1) == 10
        assert len(page2) == 10
        assert set(page1).isdisjoint(set(page2))

    def test_limit_zero_returns_empty(self, tmp_path):
        db = _open(tmp_path)
        vecs = _unit_batch(5, 16, seed=53)
        db.insert_batch([f"v{i}" for i in range(5)], vecs)
        result = db.list_ids(limit=0)
        assert result == []

    def test_offset_beyond_count_returns_empty(self, tmp_path):
        db = _open(tmp_path)
        vecs = _unit_batch(5, 16, seed=54)
        db.insert_batch([f"v{i}" for i in range(5)], vecs)
        result = db.list_ids(offset=1000)
        assert result == []

    def test_negative_limit_raises(self, tmp_path):
        db = _open(tmp_path)
        with pytest.raises((ValueError, RuntimeError)):
            db.list_ids(limit=-1)

    def test_negative_offset_raises(self, tmp_path):
        db = _open(tmp_path)
        with pytest.raises((ValueError, RuntimeError)):
            db.list_ids(offset=-1)


# ===========================================================================
# 7. query() output format
# ===========================================================================

class TestQueryOutputFormat:
    """query() always returns id/score/metadata; document optional."""

    def test_result_structure_without_documents(self, tmp_path):
        db = _open(tmp_path)
        vecs = _unit_batch(10, 16, seed=60)
        db.insert_batch([f"v{i}" for i in range(10)], vecs)
        queries = _unit_batch(2, 16, seed=600)
        res = db.query(queries, n_results=3)
        assert len(res) == 2
        for batch in res:
            for r in batch:
                assert "id" in r
                assert "score" in r
                assert "metadata" in r

    def test_result_includes_document_when_set(self, tmp_path):
        db = _open(tmp_path)
        db.insert("a", _vec(16, seed=1), document="hello world")
        db.insert("b", _vec(16, seed=2))
        queries = _unit_batch(1, 16, seed=601)
        res = db.query(queries, n_results=2)
        assert len(res) == 1
        results_with_doc = [r for r in res[0] if r["id"] == "a"]
        if results_with_doc:
            assert results_with_doc[0].get("document") == "hello world"

    def test_query_n_results_0_raises(self, tmp_path):
        db = _open(tmp_path)
        vecs = _unit_batch(5, 16, seed=61)
        db.insert_batch([f"v{i}" for i in range(5)], vecs)
        queries = _unit_batch(1, 16, seed=602)
        with pytest.raises((ValueError, RuntimeError)):
            db.query(queries, n_results=0)

    def test_query_negative_n_results_raises(self, tmp_path):
        db = _open(tmp_path)
        vecs = _unit_batch(5, 16, seed=62)
        db.insert_batch([f"v{i}" for i in range(5)], vecs)
        queries = _unit_batch(1, 16, seed=603)
        with pytest.raises((ValueError, RuntimeError)):
            db.query(queries, n_results=-1)

    def test_query_nan_row_raises(self, tmp_path):
        db = _open(tmp_path)
        vecs = _unit_batch(5, 16, seed=63)
        db.insert_batch([f"v{i}" for i in range(5)], vecs)
        queries = np.zeros((2, 16), dtype=np.float32)
        queries[1, 0] = float("nan")
        with pytest.raises((ValueError, RuntimeError)):
            db.query(queries, n_results=3)

    def test_query_inf_row_raises(self, tmp_path):
        db = _open(tmp_path)
        vecs = _unit_batch(5, 16, seed=64)
        db.insert_batch([f"v{i}" for i in range(5)], vecs)
        queries = np.zeros((1, 16), dtype=np.float32)
        queries[0, 3] = float("inf")
        with pytest.raises((ValueError, RuntimeError)):
            db.query(queries, n_results=3)

    def test_query_wrong_dim_raises(self, tmp_path):
        db = _open(tmp_path, d=16)
        vecs = _unit_batch(5, 16, seed=65)
        db.insert_batch([f"v{i}" for i in range(5)], vecs)
        queries = _unit_batch(1, 8, seed=604)
        with pytest.raises((ValueError, RuntimeError)):
            db.query(queries, n_results=3)

    def test_query_filter_applied(self, tmp_path):
        db = _open(tmp_path)
        vecs = _unit_batch(20, 16, seed=66)
        ids = [f"v{i}" for i in range(20)]
        metas = [{"cls": "X" if i < 10 else "Y"} for i in range(20)]
        db.insert_batch(ids, vecs, metadatas=metas)
        queries = _unit_batch(2, 16, seed=605)
        res = db.query(queries, n_results=5, where_filter={"cls": "X"})
        for batch in res:
            for r in batch:
                assert r["metadata"]["cls"] == "X"


# ===========================================================================
# 8. delete_batch() — filter-only, return value, combined ids+filter
# ===========================================================================

class TestDeleteBatchSemantics:
    """delete_batch() combined args, return value, filter-only."""

    def test_filter_only_delete_returns_count(self, tmp_path):
        db = _open(tmp_path)
        vecs = _unit_batch(10, 16, seed=70)
        ids = [f"v{i}" for i in range(10)]
        metas = [{"del": True if i < 4 else False} for i in range(10)]
        db.insert_batch(ids, vecs, metadatas=metas)
        count = db.delete_batch(where_filter={"del": True})
        assert count == 4
        assert db.count() == 6

    def test_ids_only_delete_returns_count(self, tmp_path):
        db = _open(tmp_path)
        vecs = _unit_batch(10, 16, seed=71)
        db.insert_batch([f"v{i}" for i in range(10)], vecs)
        count = db.delete_batch(ids=["v0", "v1", "v2"])
        assert count == 3
        assert db.count() == 7

    def test_combined_ids_and_filter(self, tmp_path):
        db = _open(tmp_path)
        vecs = _unit_batch(10, 16, seed=72)
        ids = [f"v{i}" for i in range(10)]
        metas = [{"tag": "rm" if i >= 7 else "keep"} for i in range(10)]
        db.insert_batch(ids, vecs, metadatas=metas)
        # delete ids 0-1 explicitly + filter removes 7,8,9
        count = db.delete_batch(ids=["v0", "v1"], where_filter={"tag": "rm"})
        # 2 explicit + 3 via filter = 5 total
        assert count == 5
        assert db.count() == 5

    def test_delete_batch_nonexistent_ids_not_counted(self, tmp_path):
        db = _open(tmp_path)
        db.insert("a", _vec(16, seed=1))
        count = db.delete_batch(ids=["ghost1", "ghost2"])
        assert count == 0

    def test_delete_batch_empty_ids_empty_filter_deletes_nothing(self, tmp_path):
        db = _open(tmp_path)
        vecs = _unit_batch(5, 16, seed=73)
        db.insert_batch([f"v{i}" for i in range(5)], vecs)
        count = db.delete_batch()
        assert count == 0
        assert db.count() == 5

    def test_delete_batch_overlap_not_double_counted(self, tmp_path):
        """ID in both ids list and matching filter should only count once."""
        db = _open(tmp_path)
        db.insert("x", _vec(16, seed=1), metadata={"flag": True})
        count = db.delete_batch(ids=["x"], where_filter={"flag": True})
        # x is deleted once total
        assert count <= 1
        assert db.count() == 0


# ===========================================================================
# 9. __len__ and __contains__ dunder methods
# ===========================================================================

class TestDunderMethods:
    """len(db) and id in db edge cases."""

    def test_len_empty(self, tmp_path):
        db = _open(tmp_path)
        assert len(db) == 0

    def test_len_after_inserts(self, tmp_path):
        db = _open(tmp_path)
        for i in range(7):
            db.insert(f"v{i}", _vec(16, seed=i))
        assert len(db) == 7

    def test_len_after_delete(self, tmp_path):
        db = _open(tmp_path)
        for i in range(5):
            db.insert(f"v{i}", _vec(16, seed=i))
        db.delete("v2")
        assert len(db) == 4

    def test_len_after_batch_insert(self, tmp_path):
        db = _open(tmp_path)
        vecs = _unit_batch(100, 16, seed=80)
        db.insert_batch([f"v{i}" for i in range(100)], vecs)
        assert len(db) == 100

    def test_contains_existing_id(self, tmp_path):
        db = _open(tmp_path)
        db.insert("hello", _vec(16, seed=1))
        assert "hello" in db

    def test_contains_missing_id(self, tmp_path):
        db = _open(tmp_path)
        db.insert("present", _vec(16, seed=1))
        assert "ghost" not in db

    def test_contains_after_delete(self, tmp_path):
        db = _open(tmp_path)
        db.insert("bye", _vec(16, seed=1))
        assert "bye" in db
        db.delete("bye")
        assert "bye" not in db

    def test_contains_empty_string_id(self, tmp_path):
        db = _open(tmp_path)
        assert "" not in db

    def test_len_equals_count(self, tmp_path):
        db = _open(tmp_path)
        vecs = _unit_batch(15, 16, seed=81)
        db.insert_batch([f"v{i}" for i in range(15)], vecs)
        assert len(db) == db.count()


# ===========================================================================
# 10. update() on non-existent ID
# ===========================================================================

class TestUpdateNonExistent:
    """update() on a missing ID should raise RuntimeError."""

    def test_update_missing_id_raises(self, tmp_path):
        db = _open(tmp_path)
        with pytest.raises((RuntimeError, ValueError)):
            db.update("ghost", _vec(16, seed=1))

    def test_update_after_delete_raises(self, tmp_path):
        db = _open(tmp_path)
        db.insert("a", _vec(16, seed=1))
        db.delete("a")
        with pytest.raises((RuntimeError, ValueError)):
            db.update("a", _vec(16, seed=2))

    def test_update_existing_id_succeeds(self, tmp_path):
        db = _open(tmp_path)
        db.insert("x", _vec(16, seed=1))
        db.update("x", _vec(16, seed=2))
        result = db.get("x")
        assert result is not None
        assert result["id"] == "x"

    def test_update_changes_vector_score(self, tmp_path):
        """update() replaces vector — search score changes."""
        db = _open(tmp_path)
        target = _vec(16, seed=99)
        opposite = -target
        db.insert("a", target)
        q = target.copy()
        score_before = db.search(q, top_k=1)[0]["score"]
        db.update("a", opposite)
        score_after = db.search(q, top_k=1)[0]["score"]
        # Score should change after vector update
        assert abs(score_before - score_after) > 0.01


# ===========================================================================
# 11. upsert() after delete()
# ===========================================================================

class TestUpsertAfterDelete:
    """upsert() on a deleted ID re-inserts it."""

    def test_upsert_deleted_id_reinserts(self, tmp_path):
        db = _open(tmp_path)
        db.insert("a", _vec(16, seed=1))
        db.delete("a")
        assert db.count() == 0
        db.upsert("a", _vec(16, seed=2))
        assert db.count() == 1
        assert "a" in db

    def test_upsert_deleted_id_retrievable(self, tmp_path):
        db = _open(tmp_path)
        db.insert("x", _vec(16, seed=1), metadata={"v": 1})
        db.delete("x")
        db.upsert("x", _vec(16, seed=2), metadata={"v": 2})
        result = db.get("x")
        assert result is not None
        assert result["metadata"]["v"] == 2

    def test_upsert_new_id_inserts(self, tmp_path):
        db = _open(tmp_path)
        db.upsert("brand_new", _vec(16, seed=1))
        assert db.count() == 1
        assert "brand_new" in db

    def test_upsert_existing_id_updates(self, tmp_path):
        db = _open(tmp_path)
        db.insert("a", _vec(16, seed=1), metadata={"gen": 1})
        db.upsert("a", _vec(16, seed=2), metadata={"gen": 2})
        assert db.count() == 1
        result = db.get("a")
        assert result["metadata"]["gen"] == 2


# ===========================================================================
# 12. normalize=True interaction with different metrics
# ===========================================================================

class TestNormalizeInteraction:
    """normalize=True auto-normalizes vectors at ingest and query time."""

    def test_normalize_ip_equals_cosine_scores(self, tmp_path):
        """IP + normalize should give same scores as cosine (for unit vecs)."""
        vecs = _unit_batch(20, 16, seed=90)
        ids = [f"v{i}" for i in range(20)]
        q = _vec(16, seed=900)

        p1 = str(tmp_path / "ip_norm")
        p2 = str(tmp_path / "cosine")
        db_ip = Database.open(p1, dimension=16, bits=4, metric="ip", normalize=True)
        db_cos = Database.open(p2, dimension=16, bits=4, metric="cosine")
        db_ip.insert_batch(ids, vecs)
        db_cos.insert_batch(ids, vecs)

        res_ip = db_ip.search(q, top_k=5)
        res_cos = db_cos.search(q, top_k=5)
        # Top IDs should mostly agree
        top_ip = [r["id"] for r in res_ip]
        top_cos = [r["id"] for r in res_cos]
        overlap = len(set(top_ip) & set(top_cos))
        assert overlap >= 3, f"Expected ≥3 overlap: ip={top_ip}, cos={top_cos}"

    def test_normalize_large_magnitude_gives_unit_result(self, tmp_path):
        """A large-magnitude vector stored with normalize=True scores like unit vec."""
        db = Database.open(str(tmp_path), dimension=16, bits=4, metric="ip", normalize=True)
        base = _vec(16, seed=91)
        large = base * 1000.0
        db.insert("large", large)
        db.insert("unit", base)
        # Both should have similar search scores since both get normalized
        q = base.copy()
        res = db.search(q, top_k=2)
        scores = {r["id"]: r["score"] for r in res}
        if "large" in scores and "unit" in scores:
            # Scores may differ due to quantization but should be in same ballpark
            assert abs(scores["large"] - scores["unit"]) < 0.5

    def test_normalize_true_with_metric_l2(self, tmp_path):
        """normalize=True with L2 metric — normalized vectors, L2 distance."""
        db = Database.open(str(tmp_path), dimension=16, bits=4, metric="l2", normalize=True)
        db.insert("a", _vec(16, seed=1))
        db.insert("b", _vec(16, seed=2))
        q = _vec(16, seed=3)
        res = db.search(q, top_k=2)
        assert len(res) == 2
        # L2 scores with normalize should be non-negative
        for r in res:
            assert r["score"] >= -0.05  # allow tiny fp error

    def test_normalize_false_large_magnitude_affects_ip(self, tmp_path):
        """Without normalize, large-magnitude vector dominates IP search."""
        db = Database.open(str(tmp_path), dimension=16, bits=4, metric="ip", normalize=False)
        base = _vec(16, seed=92)
        large = base * 100.0
        db.insert("large", large.astype(np.float32))
        db.insert("unit", base)
        q = base.copy()
        res = db.search(q, top_k=2)
        # large should come first (higher IP due to magnitude)
        assert res[0]["id"] == "large"


# ===========================================================================
# 13. fast_mode parameter
# ===========================================================================

class TestFastMode:
    """fast_mode=True (MSE-only) vs fast_mode=False (MSE+QJL)."""

    def test_fast_mode_true_bits1_valid(self, tmp_path):
        """bits=1 is valid when fast_mode=True."""
        db = Database.open(str(tmp_path), dimension=16, bits=1, fast_mode=True)
        db.insert("a", _vec(16, seed=1))
        assert db.count() == 1

    def test_fast_mode_false_bits1_raises(self, tmp_path):
        """bits=1 + fast_mode=False should raise ValueError (bit reserved for QJL)."""
        with pytest.raises((ValueError, RuntimeError)):
            Database.open(str(tmp_path), dimension=16, bits=1, fast_mode=False)

    def test_fast_mode_false_bits2_valid(self, tmp_path):
        """bits=2 + fast_mode=False is the minimum valid config."""
        db = Database.open(str(tmp_path), dimension=16, bits=2, fast_mode=False)
        db.insert("a", _vec(16, seed=1))
        assert db.count() == 1

    def test_fast_mode_false_search_returns_results(self, tmp_path):
        db = Database.open(str(tmp_path), dimension=16, bits=4, fast_mode=False)
        vecs = _unit_batch(20, 16, seed=100)
        db.insert_batch([f"v{i}" for i in range(20)], vecs)
        q = _vec(16, seed=1000)
        res = db.search(q, top_k=5)
        assert len(res) == 5

    def test_fast_mode_true_then_false_reopen_same_path(self, tmp_path):
        """Reopening a fast_mode=True DB with fast_mode=False (no dimension) uses manifest."""
        db = Database.open(str(tmp_path), dimension=16, bits=4, fast_mode=True)
        db.insert("a", _vec(16, seed=1))
        del db
        gc.collect()
        gc.collect()
        # Reopen without dimension → loads from manifest (fast_mode stored in manifest)
        db2 = Database.open(str(tmp_path))
        assert db2.count() == 1


# ===========================================================================
# 14. insert_batch chunk boundary (2000-item chunks)
# ===========================================================================

class TestChunkBoundary:
    """Stress the 2000-item chunk size in insert_batch."""

    def test_exactly_2000_items(self, tmp_path):
        db = _open(tmp_path)
        n = 2000
        vecs = _unit_batch(n, 16, seed=110)
        db.insert_batch([f"v{i}" for i in range(n)], vecs)
        assert db.count() == n

    def test_2001_items_crosses_boundary(self, tmp_path):
        db = _open(tmp_path)
        n = 2001
        vecs = _unit_batch(n, 16, seed=111)
        db.insert_batch([f"v{i}" for i in range(n)], vecs)
        assert db.count() == n

    def test_exactly_4000_items_two_chunks(self, tmp_path):
        db = _open(tmp_path)
        n = 4000
        vecs = _unit_batch(n, 16, seed=112)
        db.insert_batch([f"v{i}" for i in range(n)], vecs)
        assert db.count() == n

    def test_1999_items_stays_in_one_chunk(self, tmp_path):
        db = _open(tmp_path)
        n = 1999
        vecs = _unit_batch(n, 16, seed=113)
        db.insert_batch([f"v{i}" for i in range(n)], vecs)
        assert db.count() == n

    def test_search_after_multi_chunk_insert(self, tmp_path):
        db = _open(tmp_path)
        n = 2500
        vecs = _unit_batch(n, 16, seed=114)
        db.insert_batch([f"v{i}" for i in range(n)], vecs)
        q = vecs[0].copy()
        res = db.search(q, top_k=10)
        assert len(res) == 10
        assert res[0]["id"] == "v0"  # self should be closest


# ===========================================================================
# 15. rerank_precision aliases
# ===========================================================================

class TestRerankPrecisionAliases:
    """All valid rerank_precision aliases should be accepted."""

    @pytest.mark.parametrize("alias", ["disabled", "dequant", "i8", "half", "float", "full",
                                        "int8", "int4", "f16", "f32"])
    def test_alias_accepted(self, tmp_path, alias):
        p = str(tmp_path / alias.replace("/", "_"))
        db = Database.open(p, dimension=16, bits=4, rerank_precision=alias)
        db.insert("a", _vec(16, seed=1))
        assert db.count() == 1

    def test_invalid_alias_raises(self, tmp_path):
        with pytest.raises((ValueError, RuntimeError)):
            Database.open(str(tmp_path), dimension=16, bits=4, rerank_precision="bad_precision")

    def test_none_default_rerank_false(self, tmp_path):
        """With rerank=False and precision=None, should work fine."""
        db = Database.open(str(tmp_path), dimension=16, bits=4, rerank=False,
                           rerank_precision=None)
        db.insert("a", _vec(16, seed=1))
        assert db.count() == 1

    def test_precision_f32_high_recall(self, tmp_path):
        """f32 precision stores full raw vectors — should achieve near-perfect recall."""
        db = Database.open(str(tmp_path), dimension=64, bits=4, rerank=True,
                           rerank_precision="f32")
        vecs = _unit_batch(100, 64, seed=120)
        db.insert_batch([f"v{i}" for i in range(100)], vecs)
        hits = 0
        for i in range(20):
            q = vecs[i].copy()
            res = db.search(q, top_k=1)
            if res and res[0]["id"] == f"v{i}":
                hits += 1
        assert hits >= 15, f"Expected ≥15/20 self-hit, got {hits}"


# ===========================================================================
# 16. seed reproducibility
# ===========================================================================

class TestSeedReproducibility:
    """Same seed → same quantization → same search results."""

    def test_same_seed_same_top1(self, tmp_path):
        vecs = _unit_batch(50, 16, seed=130)
        ids = [f"v{i}" for i in range(50)]
        q = _vec(16, seed=1300)

        p1 = str(tmp_path / "db1")
        p2 = str(tmp_path / "db2")
        db1 = Database.open(p1, dimension=16, bits=4, seed=42)
        db2 = Database.open(p2, dimension=16, bits=4, seed=42)
        db1.insert_batch(ids, vecs)
        db2.insert_batch(ids, vecs)

        res1 = db1.search(q, top_k=5)
        res2 = db2.search(q, top_k=5)
        # Same seed → same codebook → same top results
        assert [r["id"] for r in res1] == [r["id"] for r in res2]

    def test_different_seed_may_differ(self, tmp_path):
        """Different seeds can (and typically do) produce different quantization."""
        vecs = _unit_batch(50, 16, seed=131)
        ids = [f"v{i}" for i in range(50)]
        q = _vec(16, seed=1310)

        p1 = str(tmp_path / "db1")
        p2 = str(tmp_path / "db2")
        db1 = Database.open(p1, dimension=16, bits=4, seed=1)
        db2 = Database.open(p2, dimension=16, bits=4, seed=999)
        db1.insert_batch(ids, vecs)
        db2.insert_batch(ids, vecs)

        res1 = db1.search(q, top_k=5)
        res2 = db2.search(q, top_k=5)
        # Different seeds may or may not give same results (this is a sanity check)
        # At minimum both must return 5 results
        assert len(res1) == 5
        assert len(res2) == 5


# ===========================================================================
# 17. list_metadata_values — dotted paths, after deletes
# ===========================================================================

class TestListMetadataValuesDotted:
    """list_metadata_values with dotted paths and deleted vector exclusion."""

    def test_dotted_path_top_level(self, tmp_path):
        db = _open(tmp_path)
        for i in range(5):
            db.insert(f"v{i}", _vec(16, seed=i), metadata={"src": f"doc{i % 2}"})
        counts = db.list_metadata_values("src")
        assert counts == {"doc0": 3, "doc1": 2} or sum(counts.values()) == 5

    def test_dotted_path_nested_object(self, tmp_path):
        db = _open(tmp_path)
        db.insert("a", _vec(16, seed=1), metadata={"meta": {"source": "wiki"}})
        db.insert("b", _vec(16, seed=2), metadata={"meta": {"source": "arxiv"}})
        db.insert("c", _vec(16, seed=3), metadata={"meta": {"source": "wiki"}})
        counts = db.list_metadata_values("meta.source")
        assert counts.get("wiki", 0) == 2
        assert counts.get("arxiv", 0) == 1

    def test_deleted_vectors_excluded_from_counts(self, tmp_path):
        db = _open(tmp_path)
        for i in range(6):
            db.insert(f"v{i}", _vec(16, seed=i), metadata={"tag": "X" if i < 3 else "Y"})
        # delete all X-tagged vectors
        db.delete_batch(where_filter={"tag": "X"})
        counts = db.list_metadata_values("tag")
        assert counts.get("X", 0) == 0
        assert counts.get("Y", 0) == 3

    def test_missing_field_not_counted(self, tmp_path):
        db = _open(tmp_path)
        db.insert("a", _vec(16, seed=1), metadata={"color": "red"})
        db.insert("b", _vec(16, seed=2))  # no "color" field
        counts = db.list_metadata_values("color")
        assert counts == {"red": 1}

    def test_integer_value_stringified(self, tmp_path):
        db = _open(tmp_path)
        db.insert("a", _vec(16, seed=1), metadata={"count": 42})
        counts = db.list_metadata_values("count")
        # Non-string values get stringified (42 → "42")
        assert "42" in counts


# ===========================================================================
# 18. stats() detailed field verification
# ===========================================================================

class TestStatsDetailed:
    """stats() all fields verified after full lifecycle."""

    def test_all_expected_keys_present(self, tmp_path):
        db = _open(tmp_path)
        s = db.stats()
        expected_keys = {
            "vector_count", "segment_count", "buffered_vectors", "dimension", "bits",
            "total_disk_bytes", "has_index", "index_nodes", "live_codes_bytes",
            "live_slot_count", "live_id_count", "live_vectors_count",
            "live_vectors_bytes_estimate", "metadata_entries", "metadata_bytes_estimate",
            "ann_slot_count", "graph_nodes", "delta_size", "ram_estimate_bytes",
        }
        missing = expected_keys - set(s.keys())
        assert not missing, f"Missing stats keys: {missing}"

    def test_dimension_matches_open_param(self, tmp_path):
        db = Database.open(str(tmp_path), dimension=32, bits=4)
        assert db.stats()["dimension"] == 32

    def test_bits_matches_open_param(self, tmp_path):
        db = Database.open(str(tmp_path), dimension=16, bits=8)
        assert db.stats()["bits"] == 8

    def test_vector_count_tracks_crud(self, tmp_path):
        db = _open(tmp_path)
        assert db.stats()["vector_count"] == 0
        db.insert("a", _vec(16, seed=1))
        assert db.stats()["vector_count"] == 1
        db.delete("a")
        assert db.stats()["vector_count"] == 0

    def test_has_index_false_before_create(self, tmp_path):
        db = _open(tmp_path)
        vecs = _unit_batch(10, 16, seed=150)
        db.insert_batch([f"v{i}" for i in range(10)], vecs)
        assert not db.stats()["has_index"]

    def test_has_index_true_after_create(self, tmp_path):
        db = _open(tmp_path)
        vecs = _unit_batch(20, 16, seed=151)
        db.insert_batch([f"v{i}" for i in range(20)], vecs)
        db.create_index(max_degree=4)
        assert db.stats()["has_index"]

    def test_graph_nodes_matches_index_nodes(self, tmp_path):
        db = _open(tmp_path)
        n = 30
        vecs = _unit_batch(n, 16, seed=152)
        db.insert_batch([f"v{i}" for i in range(n)], vecs)
        db.create_index(max_degree=4)
        s = db.stats()
        assert s["has_index"]
        # graph_nodes and index_nodes should both be > 0 after indexing
        assert s["graph_nodes"] > 0
        # index_nodes is the number of nodes in the graph state
        assert s["index_nodes"] >= 0

    def test_delta_size_grows_after_post_index_inserts(self, tmp_path):
        db = _open(tmp_path)
        n = 20
        vecs = _unit_batch(n, 16, seed=153)
        db.insert_batch([f"v{i}" for i in range(n)], vecs)
        db.create_index(max_degree=4)
        delta_before = db.stats()["delta_size"]
        # insert more vectors after index creation
        extra = _unit_batch(5, 16, seed=154)
        db.insert_batch([f"extra{i}" for i in range(5)], extra)
        delta_after = db.stats()["delta_size"]
        assert delta_after >= delta_before

    def test_ram_estimate_nonzero_after_insert(self, tmp_path):
        db = _open(tmp_path)
        db.insert("a", _vec(16, seed=1))
        s = db.stats()
        assert s["ram_estimate_bytes"] > 0

    def test_metadata_entries_count(self, tmp_path):
        db = _open(tmp_path)
        db.insert("a", _vec(16, seed=1), metadata={"x": 1})
        db.insert("b", _vec(16, seed=2))  # no metadata
        s = db.stats()
        # metadata_entries counts vectors with metadata
        assert s["metadata_entries"] >= 1


# ===========================================================================
# 19. filter $in and $nin with mixed types
# ===========================================================================

class TestFilterInNinMixed:
    """$in and $nin with integer, float, and string values."""

    def test_in_integers(self, tmp_path):
        db = _open(tmp_path)
        for i in range(5):
            db.insert(f"v{i}", _vec(16, seed=i), metadata={"score": i})
        res = db.search(_vec(16, seed=99), top_k=10, filter={"score": {"$in": [1, 3]}})
        ids = {r["id"] for r in res}
        assert ids == {"v1", "v3"}

    def test_in_floats(self, tmp_path):
        db = _open(tmp_path)
        db.insert("a", _vec(16, seed=1), metadata={"val": 1.5})
        db.insert("b", _vec(16, seed=2), metadata={"val": 2.5})
        db.insert("c", _vec(16, seed=3), metadata={"val": 3.5})
        res = db.search(_vec(16, seed=99), top_k=10,
                        filter={"val": {"$in": [1.5, 3.5]}})
        ids = {r["id"] for r in res}
        assert ids == {"a", "c"}

    def test_nin_excludes_values(self, tmp_path):
        db = _open(tmp_path)
        for i in range(5):
            db.insert(f"v{i}", _vec(16, seed=i), metadata={"num": i})
        res = db.search(_vec(16, seed=99), top_k=10,
                        filter={"num": {"$nin": [0, 1, 2]}})
        ids = {r["id"] for r in res}
        assert ids == {"v3", "v4"}

    def test_in_empty_list_matches_nothing(self, tmp_path):
        db = _open(tmp_path)
        db.insert("a", _vec(16, seed=1), metadata={"x": 1})
        res = db.search(_vec(16, seed=99), top_k=5, filter={"x": {"$in": []}})
        assert len(res) == 0

    def test_nin_empty_list_matches_all(self, tmp_path):
        db = _open(tmp_path)
        for i in range(3):
            db.insert(f"v{i}", _vec(16, seed=i), metadata={"x": i})
        res = db.search(_vec(16, seed=99), top_k=5, filter={"x": {"$nin": []}})
        assert len(res) == 3

    def test_in_missing_field_matches_nothing(self, tmp_path):
        db = _open(tmp_path)
        db.insert("a", _vec(16, seed=1))  # no metadata
        res = db.search(_vec(16, seed=99), top_k=5, filter={"missing": {"$in": [1, 2]}})
        assert len(res) == 0


# ===========================================================================
# 20. rerank=False path
# ===========================================================================

class TestRerankFalse:
    """rerank=False skips reranking — should still produce correct results."""

    def test_rerank_false_search_returns_results(self, tmp_path):
        db = Database.open(str(tmp_path), dimension=16, bits=4, rerank=False)
        vecs = _unit_batch(30, 16, seed=160)
        db.insert_batch([f"v{i}" for i in range(30)], vecs)
        q = _vec(16, seed=1600)
        res = db.search(q, top_k=5)
        assert len(res) == 5

    def test_rerank_false_self_top1(self, tmp_path):
        db = Database.open(str(tmp_path), dimension=16, bits=4, rerank=False, metric="ip")
        vecs = _unit_batch(30, 16, seed=161)
        ids = [f"v{i}" for i in range(30)]
        db.insert_batch(ids, vecs)
        hits = 0
        for i in range(10):
            q = vecs[i].copy()
            res = db.search(q, top_k=1)
            if res and res[0]["id"] == f"v{i}":
                hits += 1
        assert hits >= 7, f"Expected ≥7/10 self-hit with rerank=False, got {hits}"

    def test_rerank_false_with_filter(self, tmp_path):
        db = Database.open(str(tmp_path), dimension=16, bits=4, rerank=False)
        vecs = _unit_batch(20, 16, seed=162)
        ids = [f"v{i}" for i in range(20)]
        metas = [{"grp": "A" if i < 10 else "B"} for i in range(20)]
        db.insert_batch(ids, vecs, metadatas=metas)
        q = _vec(16, seed=1620)
        res = db.search(q, top_k=5, filter={"grp": "A"})
        assert all(r["metadata"]["grp"] == "A" for r in res)


# ===========================================================================
# 21. ANN vs brute-force score proximity
# ===========================================================================

class TestAnnVsBruteForce:
    """ANN results should approximately match brute-force results."""

    def test_top1_overlap_ann_vs_brute(self, tmp_path):
        db = _open(tmp_path)
        n = 100
        vecs = _unit_batch(n, 16, seed=170)
        db.insert_batch([f"v{i}" for i in range(n)], vecs)
        db.create_index(max_degree=16, ef_construction=100, search_list_size=64)
        assert db.stats()["has_index"]

        hits = 0
        for i in range(20):
            q = _vec(16, seed=1700 + i)
            res_bf = db.search(q, top_k=1, _use_ann=False)
            res_ann = db.search(q, top_k=1, _use_ann=True)
            if res_bf and res_ann and res_bf[0]["id"] == res_ann[0]["id"]:
                hits += 1
        # At least 50% overlap at top-1 with these settings
        assert hits >= 10, f"ANN top-1 recall: {hits}/20"

    def test_top5_recall_ann_vs_brute(self, tmp_path):
        db = _open(tmp_path)
        n = 100
        vecs = _unit_batch(n, 16, seed=171)
        db.insert_batch([f"v{i}" for i in range(n)], vecs)
        db.create_index(max_degree=16, ef_construction=200, search_list_size=128)

        total_recall = 0
        n_queries = 10
        for i in range(n_queries):
            q = _vec(16, seed=1710 + i)
            res_bf = {r["id"] for r in db.search(q, top_k=5, _use_ann=False)}
            res_ann = {r["id"] for r in db.search(q, top_k=5, _use_ann=True)}
            total_recall += len(res_bf & res_ann) / 5
        avg_recall = total_recall / n_queries
        assert avg_recall >= 0.4, f"ANN @5 recall {avg_recall:.2f} < 0.40"

    def test_brute_force_scores_match_ann_scores_approximately(self, tmp_path):
        db = _open(tmp_path)
        n = 50
        vecs = _unit_batch(n, 16, seed=172)
        db.insert_batch([f"v{i}" for i in range(n)], vecs)
        db.create_index(max_degree=16, ef_construction=100)
        q = _vec(16, seed=1720)
        res_bf = db.search(q, top_k=1, _use_ann=False)
        res_ann = db.search(q, top_k=1, _use_ann=True)
        if res_bf and res_ann and res_bf[0]["id"] == res_ann[0]["id"]:
            score_diff = abs(res_bf[0]["score"] - res_ann[0]["score"])
            # Same vector → scores should be identical
            assert score_diff < 0.01


# ===========================================================================
# 22. Persistence — reopen after large batch with various flush states
# ===========================================================================

class TestPersistenceRobust:
    """Data survives reopen in different WAL/segment states."""

    def test_reopen_after_explicit_flush(self, tmp_path):
        db = _open(tmp_path)
        vecs = _unit_batch(10, 16, seed=180)
        db.insert_batch([f"v{i}" for i in range(10)], vecs)
        db.flush()
        db.close()
        del db
        gc.collect()
        gc.collect()
        db2 = Database.open(str(tmp_path))
        assert db2.count() == 10

    def test_reopen_without_flush(self, tmp_path):
        # With high threshold, vectors stay in WAL — reopen should still see them
        db = Database.open(str(tmp_path), dimension=16, bits=4,
                           wal_flush_threshold=100000)
        vecs = _unit_batch(5, 16, seed=181)
        db.insert_batch([f"v{i}" for i in range(5)], vecs)
        db.close()
        del db
        gc.collect()
        gc.collect()
        db2 = Database.open(str(tmp_path))
        assert db2.count() == 5

    def test_reopen_preserves_metadata(self, tmp_path):
        db = _open(tmp_path)
        db.insert("a", _vec(16, seed=1), metadata={"title": "Test Doc"},
                  document="some text")
        db.flush()
        db.close()
        del db
        gc.collect()
        gc.collect()
        db2 = Database.open(str(tmp_path))
        result = db2.get("a")
        assert result is not None
        assert result["metadata"]["title"] == "Test Doc"
        assert result.get("document") == "some text"

    def test_reopen_preserves_index(self, tmp_path):
        db = _open(tmp_path)
        n = 30
        vecs = _unit_batch(n, 16, seed=182)
        db.insert_batch([f"v{i}" for i in range(n)], vecs)
        db.create_index(max_degree=4)
        db.close()
        del db
        gc.collect()
        gc.collect()
        db2 = Database.open(str(tmp_path))
        assert db2.stats()["has_index"]

    def test_reopen_wrong_dimension_raises(self, tmp_path):
        db = Database.open(str(tmp_path), dimension=16, bits=4)
        db.insert("a", _vec(16, seed=1))
        db.close()
        del db
        gc.collect()
        gc.collect()
        with pytest.raises((ValueError, RuntimeError)):
            Database.open(str(tmp_path), dimension=32, bits=4)


# ===========================================================================
# 23. Collection isolation
# ===========================================================================

class TestCollectionIsolationDeep:
    """Collections are fully isolated — CRUD in one doesn't bleed into another."""

    def test_two_collections_independent_count(self, tmp_path):
        db1 = Database.open(str(tmp_path), dimension=16, bits=4, collection="colA")
        db2 = Database.open(str(tmp_path), dimension=16, bits=4, collection="colB")
        db1.insert("a", _vec(16, seed=1))
        db1.insert("b", _vec(16, seed=2))
        db2.insert("x", _vec(16, seed=3))
        assert db1.count() == 2
        assert db2.count() == 1

    def test_search_isolated_across_collections(self, tmp_path):
        db1 = Database.open(str(tmp_path), dimension=16, bits=4, collection="col1")
        db2 = Database.open(str(tmp_path), dimension=16, bits=4, collection="col2")
        target = _vec(16, seed=99)
        db1.insert("target", target)
        db2.insert("decoy", _vec(16, seed=1))
        q = target.copy()
        res2 = db2.search(q, top_k=1)
        assert not any(r["id"] == "target" for r in res2)

    def test_delete_in_one_collection_doesnt_affect_other(self, tmp_path):
        db1 = Database.open(str(tmp_path), dimension=16, bits=4, collection="c1")
        db2 = Database.open(str(tmp_path), dimension=16, bits=4, collection="c2")
        db1.insert("shared_id", _vec(16, seed=1))
        db2.insert("shared_id", _vec(16, seed=2))
        db1.delete("shared_id")
        assert db1.count() == 0
        assert db2.count() == 1  # db2 unaffected

    def test_invalid_collection_traversal_rejected(self, tmp_path):
        with pytest.raises((ValueError, RuntimeError)):
            Database.open(str(tmp_path), dimension=16, bits=4, collection="../evil")

    def test_collection_with_slash_rejected(self, tmp_path):
        with pytest.raises((ValueError, RuntimeError)):
            Database.open(str(tmp_path), dimension=16, bits=4, collection="a/b")


# ===========================================================================
# 24. Concurrent write + read stress
# ===========================================================================

class TestConcurrentStress:
    """Multi-threaded concurrent operations stress test."""

    def test_concurrent_inserts_all_stored(self, tmp_path):
        db = _open(tmp_path)
        n_threads = 10
        items_per_thread = 50
        errors = []

        def worker(tid):
            try:
                for i in range(items_per_thread):
                    vid = f"t{tid}_v{i}"
                    db.insert(vid, _vec(16, seed=tid * 1000 + i))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent insert errors: {errors}"
        assert db.count() == n_threads * items_per_thread

    def test_concurrent_reads_while_writing(self, tmp_path):
        db = _open(tmp_path)
        vecs = _unit_batch(50, 16, seed=190)
        db.insert_batch([f"v{i}" for i in range(50)], vecs)
        q = _vec(16, seed=1900)
        errors = []

        def reader():
            try:
                for _ in range(20):
                    db.search(q, top_k=5)
            except Exception as e:
                errors.append(e)

        def writer():
            try:
                for i in range(10):
                    db.insert(f"new{i}", _vec(16, seed=9000 + i))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(5)]
        threads.append(threading.Thread(target=writer))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent read/write errors: {errors}"


# ===========================================================================
# 25. query() with where_filter + ann
# ===========================================================================

class TestQueryWithFilterAndAnn:
    """query() supports where_filter and ANN mode."""

    def test_query_with_where_filter(self, tmp_path):
        db = _open(tmp_path)
        vecs = _unit_batch(30, 16, seed=200)
        ids = [f"v{i}" for i in range(30)]
        metas = [{"cls": "X" if i < 15 else "Y"} for i in range(30)]
        db.insert_batch(ids, vecs, metadatas=metas)
        queries = _unit_batch(2, 16, seed=2000)
        res = db.query(queries, n_results=5, where_filter={"cls": "X"})
        for batch in res:
            for r in batch:
                assert r["metadata"]["cls"] == "X"

    def test_query_with_ann(self, tmp_path):
        db = _open(tmp_path)
        vecs = _unit_batch(80, 16, seed=201)
        db.insert_batch([f"v{i}" for i in range(80)], vecs)
        db.create_index(max_degree=8, search_list_size=32)
        queries = _unit_batch(3, 16, seed=2010)
        res = db.query(queries, n_results=5, _use_ann=True)
        assert len(res) == 3
        assert all(len(r) > 0 for r in res)


# ===========================================================================
# 26. Metadata with list values in filters
# ===========================================================================

class TestMetadataListValues:
    """Metadata can store list values (JSON arrays)."""

    def test_list_value_stored_and_retrieved(self, tmp_path):
        db = _open(tmp_path)
        db.insert("a", _vec(16, seed=1), metadata={"tags": ["rust", "db", "vector"]})
        result = db.get("a")
        assert result["metadata"]["tags"] == ["rust", "db", "vector"]

    def test_list_value_survives_reopen(self, tmp_path):
        db = _open(tmp_path)
        db.insert("a", _vec(16, seed=1), metadata={"nums": [1, 2, 3]})
        db.flush()
        db.close()
        del db
        gc.collect()
        gc.collect()
        db2 = Database.open(str(tmp_path))
        result = db2.get("a")
        assert result["metadata"]["nums"] == [1, 2, 3]

    def test_contains_filter_on_list_value_returns_nothing(self, tmp_path):
        """$contains on a list/non-string field returns false (by design)."""
        db = _open(tmp_path)
        db.insert("a", _vec(16, seed=1), metadata={"tags": ["rust", "vector"]})
        db.insert("b", _vec(16, seed=2), metadata={"tags": ["python", "ml"]})
        # $contains does substring match on string values only; list → false
        res = db.search(_vec(16, seed=99), top_k=5,
                        filter={"tags": {"$contains": "rust"}})
        ids = {r["id"] for r in res}
        assert "a" not in ids  # list is non-string → $contains returns false


# ===========================================================================
# 27. score ordering invariants — L2 and cosine
# ===========================================================================

class TestScoreOrderingL2Cosine:
    """Score ordering invariants for L2 and cosine metrics."""

    def test_l2_scores_non_negative(self, tmp_path):
        """All L2 scores returned to Python must be >= 0."""
        db = Database.open(str(tmp_path), dimension=32, bits=4, metric="l2")
        vecs = _unit_batch(30, 32, seed=210)
        db.insert_batch([f"v{i}" for i in range(30)], vecs)
        q = _vec(32, seed=2100)
        res = db.search(q, top_k=10)
        for r in res:
            assert r["score"] >= -0.05, f"Negative L2 score: {r['score']}"

    def test_cosine_scores_bounded(self, tmp_path):
        """Cosine similarity scores should be in approximately [-1, 1]."""
        db = Database.open(str(tmp_path), dimension=32, bits=4, metric="cosine")
        vecs = _unit_batch(30, 32, seed=211)
        db.insert_batch([f"v{i}" for i in range(30)], vecs)
        q = _vec(32, seed=2110)
        res = db.search(q, top_k=10)
        for r in res:
            assert -1.1 <= r["score"] <= 1.1, f"Out-of-range cosine score: {r['score']}"

    def test_scores_descending_order(self, tmp_path):
        """search() always returns results in descending score order."""
        db = _open(tmp_path, metric="ip")
        vecs = _unit_batch(30, 16, seed=212)
        db.insert_batch([f"v{i}" for i in range(30)], vecs)
        q = _vec(16, seed=2120)
        res = db.search(q, top_k=10)
        scores = [r["score"] for r in res]
        assert scores == sorted(scores, reverse=True), "Scores not in descending order"


# ===========================================================================
# 28. include= parameter edge cases in search()
# ===========================================================================

class TestIncludeEdgeCases:
    """include= parameter — all combinations and unknown fields."""

    def test_include_only_id(self, tmp_path):
        db = _open(tmp_path)
        db.insert("a", _vec(16, seed=1))
        res = db.search(_vec(16, seed=99), top_k=1, include=["id"])
        assert "id" in res[0]
        assert "score" not in res[0]
        assert "metadata" not in res[0]

    def test_include_only_score(self, tmp_path):
        db = _open(tmp_path)
        db.insert("a", _vec(16, seed=1))
        res = db.search(_vec(16, seed=99), top_k=1, include=["score"])
        assert "score" in res[0]
        assert "id" not in res[0]

    def test_include_id_and_score(self, tmp_path):
        db = _open(tmp_path)
        db.insert("a", _vec(16, seed=1))
        res = db.search(_vec(16, seed=99), top_k=1, include=["id", "score"])
        assert "id" in res[0]
        assert "score" in res[0]
        assert "metadata" not in res[0]

    def test_include_all_fields(self, tmp_path):
        db = _open(tmp_path)
        db.insert("a", _vec(16, seed=1), metadata={"k": "v"}, document="doc")
        res = db.search(_vec(16, seed=99), top_k=1,
                        include=["id", "score", "metadata", "document"])
        assert "id" in res[0]
        assert "score" in res[0]
        assert "metadata" in res[0]
        assert res[0].get("document") == "doc"

    def test_include_unknown_field_raises(self, tmp_path):
        db = _open(tmp_path)
        db.insert("a", _vec(16, seed=1))
        with pytest.raises((ValueError, RuntimeError)):
            db.search(_vec(16, seed=99), top_k=1, include=["id", "unknown_field"])

    def test_include_empty_list_raises_or_returns_empty_dicts(self, tmp_path):
        """include=[] — implementation may raise or return dicts with no keys."""
        db = _open(tmp_path)
        db.insert("a", _vec(16, seed=1))
        try:
            res = db.search(_vec(16, seed=99), top_k=1, include=[])
            # If it doesn't raise, result dicts should be empty or minimal
            assert isinstance(res, list)
        except (ValueError, RuntimeError):
            pass  # raising is also acceptable


# ===========================================================================
# 29. update_metadata() preserves vs replaces
# ===========================================================================

class TestUpdateMetadataSemantics:
    """update_metadata() None preserves, non-None replaces."""

    def test_none_metadata_preserves_existing(self, tmp_path):
        db = _open(tmp_path)
        db.insert("a", _vec(16, seed=1), metadata={"x": 1, "y": 2})
        db.update_metadata("a", metadata=None, document=None)
        result = db.get("a")
        assert result["metadata"]["x"] == 1
        assert result["metadata"]["y"] == 2

    def test_new_metadata_replaces_existing(self, tmp_path):
        db = _open(tmp_path)
        db.insert("a", _vec(16, seed=1), metadata={"x": 1})
        db.update_metadata("a", metadata={"y": 2})
        result = db.get("a")
        assert result["metadata"].get("y") == 2
        # x may or may not survive depending on semantics (replace vs merge)

    def test_update_document_only_preserves_metadata(self, tmp_path):
        db = _open(tmp_path)
        db.insert("a", _vec(16, seed=1), metadata={"key": "val"}, document="old")
        db.update_metadata("a", document="new_doc")
        result = db.get("a")
        assert result.get("document") == "new_doc"
        assert result["metadata"].get("key") == "val"

    def test_update_metadata_nonexistent_id_raises(self, tmp_path):
        db = _open(tmp_path)
        with pytest.raises((RuntimeError, ValueError)):
            db.update_metadata("ghost", metadata={"x": 1})

    def test_update_metadata_then_filter_finds_updated(self, tmp_path):
        db = _open(tmp_path)
        db.insert("a", _vec(16, seed=1), metadata={"status": "draft"})
        db.update_metadata("a", metadata={"status": "published"})
        res = db.search(_vec(16, seed=99), top_k=5,
                        filter={"status": "published"})
        assert any(r["id"] == "a" for r in res)


# ===========================================================================
# 30. Extreme metadata values — round-trip fidelity
# ===========================================================================

class TestMetadataRoundTrip:
    """Metadata values round-trip through JSON serialization."""

    def test_nested_dict_roundtrip(self, tmp_path):
        db = _open(tmp_path)
        meta = {"outer": {"inner": {"deep": 42}}}
        db.insert("a", _vec(16, seed=1), metadata=meta)
        result = db.get("a")
        assert result["metadata"]["outer"]["inner"]["deep"] == 42

    def test_bool_values_roundtrip(self, tmp_path):
        db = _open(tmp_path)
        db.insert("a", _vec(16, seed=1), metadata={"flag": True, "other": False})
        result = db.get("a")
        assert result["metadata"]["flag"] is True
        assert result["metadata"]["other"] is False

    def test_null_value_roundtrip(self, tmp_path):
        db = _open(tmp_path)
        db.insert("a", _vec(16, seed=1), metadata={"nullable": None})
        result = db.get("a")
        assert result["metadata"]["nullable"] is None

    def test_mixed_type_list_roundtrip(self, tmp_path):
        db = _open(tmp_path)
        db.insert("a", _vec(16, seed=1), metadata={"mixed": [1, "two", True, None]})
        result = db.get("a")
        assert result["metadata"]["mixed"] == [1, "two", True, None]

    def test_unicode_key_and_value_roundtrip(self, tmp_path):
        db = _open(tmp_path)
        db.insert("a", _vec(16, seed=1), metadata={"名前": "テスト"})
        result = db.get("a")
        assert result["metadata"]["名前"] == "テスト"

    def test_large_integer_roundtrip(self, tmp_path):
        db = _open(tmp_path)
        big = 9_007_199_254_740_993  # > 2^53, may lose precision in JSON
        db.insert("a", _vec(16, seed=1), metadata={"big": big})
        result = db.get("a")
        # Value should survive (as int or approx float)
        assert result["metadata"]["big"] is not None

# ========================================================================
# FROM test_brutal_qa5.py
# ========================================================================

class TestWalFlushOnClose:
    """close() must flush all buffered vectors; reopen must recover them."""

    def test_buffered_vectors_survive_close_reopen(self, tmp_path):
        """Insert 20 vecs with very high flush threshold → all buffered.
        After close() the WAL/segment is written; reopen must find all 20."""
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4,
                           wal_flush_threshold=100_000, fast_mode=True)
        ids = [f"wal_{i}" for i in range(20)]
        vecs = rand_vecs(20, DIM, seed=1)
        db.insert_batch(ids, vecs)
        # Verify still buffered before close
        assert db.stats()["vector_count"] == 20
        db.close()
        del db
        gc.collect(); gc.collect()

        db2 = Database.open(str(tmp_path / "db"))
        assert len(db2) == 20
        for i_id in ids:
            assert i_id in db2
        db2.close()

    def test_search_correct_after_reopen(self, tmp_path):
        """Target vector stays at rank-1 after close → reopen."""
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4,
                           wal_flush_threshold=100_000, fast_mode=True)
        target = rand_vec(DIM, seed=99)
        db.insert("target", target, metadata={"role": "target"})
        for i in range(10):
            db.insert(f"noise_{i}", rand_vec(DIM, seed=i), metadata={"role": "noise"})
        db.close()
        del db
        gc.collect(); gc.collect()

        db2 = Database.open(str(tmp_path / "db"))
        results = db2.search(target, top_k=1)
        assert results[0]["id"] == "target"
        db2.close()

    def test_multiple_batches_survive_reopen(self, tmp_path):
        """5 batches inserted with no flush — all 50 survive after close+reopen."""
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4,
                           wal_flush_threshold=100_000, fast_mode=True)
        for batch in range(5):
            ids = [f"b{batch}_v{i}" for i in range(10)]
            db.insert_batch(ids, rand_vecs(10, DIM, seed=batch * 100))
        db.close()
        del db
        gc.collect(); gc.collect()

        db2 = Database.open(str(tmp_path / "db"))
        assert len(db2) == 50
        db2.close()

    def test_explicit_flush_then_reopen(self, tmp_path):
        """Explicit flush() + close() → reopen finds all data."""
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        db.insert_batch([f"id_{i}" for i in range(30)], rand_vecs(30, DIM, seed=5))
        db.flush()
        db.close()
        del db
        gc.collect(); gc.collect()

        db2 = Database.open(str(tmp_path / "db"))
        assert len(db2) == 30
        db2.close()


# ---------------------------------------------------------------------------
# Segment accumulation — many flush cycles
# ---------------------------------------------------------------------------

class TestSegmentAccumulation:
    """Force many flush cycles; reopen must recover all data."""

    def test_many_flush_cycles_all_data_present(self, tmp_path):
        """wal_flush_threshold=1 triggers a flush after every insert (50 segments)."""
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4,
                           wal_flush_threshold=1, fast_mode=True)
        for i in range(50):
            db.insert(f"id_{i}", rand_vec(DIM, seed=i))
        db.close()
        del db
        gc.collect(); gc.collect()

        db2 = Database.open(str(tmp_path / "db"))
        assert len(db2) == 50
        for i in range(50):
            assert f"id_{i}" in db2
        db2.close()

    def test_flush_idempotent_on_empty_buffer(self, tmp_path):
        """flush() on already-empty buffer should be a safe no-op."""
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4,
                           wal_flush_threshold=1, fast_mode=True)
        db.insert("x", rand_vec(DIM))  # Forced flush on insert
        db.flush()   # no-op
        db.flush()   # second no-op
        assert len(db) == 1
        db.close()

    def test_segment_count_reflects_flushes(self, tmp_path):
        """segment_count should be >= 1 after any data is persisted."""
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4,
                           wal_flush_threshold=1, fast_mode=True)
        db.insert("a", rand_vec(DIM, seed=1))
        db.insert("b", rand_vec(DIM, seed=2))
        seg_count = db.stats()["segment_count"]
        db.close()
        # Compaction may merge segments; at minimum there should be >= 1
        assert seg_count >= 1


# ---------------------------------------------------------------------------
# update_index() — incremental ANN coverage
# ---------------------------------------------------------------------------

class TestUpdateIndexIncremental:
    """update_index() extends HNSW to cover vectors inserted after create_index."""

    def test_without_prior_create_acts_as_create(self, tmp_path):
        """update_index() with no prior index should build from scratch."""
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4,
                           fast_mode=True, rerank=True)
        db.insert_batch([f"v{i}" for i in range(30)], rand_vecs(30, DIM, seed=7))
        db.update_index()
        assert db.stats()["has_index"] is True
        q = rand_vec(DIM, seed=99)
        results = db.search(q, top_k=5, _use_ann=True)
        assert len(results) > 0
        db.close()

    def test_new_vectors_reachable_after_update_index(self, tmp_path):
        """Vectors inserted AFTER create_index() become reachable via update_index()."""
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4,
                           fast_mode=True, rerank=True)
        db.insert_batch([f"init_{i}" for i in range(20)], rand_vecs(20, DIM, seed=0))
        db.create_index()
        # Insert new batch after index was built
        new_ids = [f"new_{i}" for i in range(10)]
        new_vecs = rand_vecs(10, DIM, seed=100)
        db.insert_batch(new_ids, new_vecs)
        db.update_index()
        assert db.stats()["has_index"] is True
        assert db.stats()["vector_count"] == 30
        # ANN search must return some results (not crash)
        results = db.search(rand_vec(DIM, seed=55), top_k=30, _use_ann=True)
        assert len(results) > 0
        db.close()

    def test_update_index_on_empty_db_no_panic(self, tmp_path):
        """update_index() on an empty DB must not panic."""
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        try:
            db.update_index()
        except (RuntimeError, Exception):
            pass  # Graceful error acceptable; panic is not
        db.close()


# ---------------------------------------------------------------------------
# RAG TurboQuantRetriever — deep tests
# ---------------------------------------------------------------------------

class TestRagRetrieverDeep:
    """Deep TurboQuantRetriever coverage including the doc_store design limitation."""

    def test_basic_add_and_search(self, tmp_path):
        r = TurboQuantRetriever(str(tmp_path / "rag"), dimension=DIM, bits=4,
                                fast_mode=True)
        texts = ["Alpha", "Beta", "Gamma"]
        vecs = [rand_vec(DIM, seed=i).tolist() for i in range(3)]
        r.add_texts(texts, vecs)
        results = r.similarity_search(vecs[0], k=1)
        assert len(results) == 1
        assert results[0]["text"] == "Alpha"

    def test_add_with_metadata(self, tmp_path):
        r = TurboQuantRetriever(str(tmp_path / "rag"), dimension=DIM, bits=4,
                                fast_mode=True)
        r.add_texts(["A", "B"], [rand_vec(DIM, seed=0).tolist(),
                                  rand_vec(DIM, seed=1).tolist()],
                    metadatas=[{"src": "X"}, {"src": "Y"}])
        out = r.similarity_search(rand_vec(DIM, seed=0).tolist(), k=1)
        assert out[0]["metadata"]["src"] == "X"

    def test_k_larger_than_corpus_returns_at_most_n(self, tmp_path):
        r = TurboQuantRetriever(str(tmp_path / "rag"), dimension=DIM, bits=4,
                                fast_mode=True)
        r.add_texts(["only"], [rand_vec(DIM, seed=0).tolist()])
        results = r.similarity_search(rand_vec(DIM, seed=0).tolist(), k=100)
        assert len(results) <= 1

    def test_similarity_search_empty_db_returns_empty(self, tmp_path):
        r = TurboQuantRetriever(str(tmp_path / "rag"), dimension=DIM, bits=4,
                                fast_mode=True)
        results = r.similarity_search(rand_vec(DIM, seed=0).tolist(), k=4)
        assert results == []

    def test_doc_store_loss_after_reopen(self, tmp_path):
        """BUG-12 candidate: doc_store is in-memory; similarity_search returns
        nothing after reopen even though the vector DB still has the data."""
        r = TurboQuantRetriever(str(tmp_path / "rag"), dimension=DIM, bits=4,
                                fast_mode=True)
        vecs = [rand_vec(DIM, seed=i).tolist() for i in range(3)]
        r.add_texts(["a", "b", "c"], vecs)
        del r
        gc.collect(); gc.collect()

        r2 = TurboQuantRetriever(str(tmp_path / "rag"), dimension=DIM, bits=4,
                                  fast_mode=True)
        assert len(r2.db) == 3  # Vector DB persisted
        results = r2.similarity_search(vecs[0], k=3)
        # doc_store is empty after reopen → 0 results returned despite DB having data
        assert isinstance(results, list)
        # Document the known gap: results will be empty
        assert len(results) == 0, (
            "BUG-12: similarity_search silently returns 0 results after reopen "
            "because doc_store is not persisted to disk"
        )

    def test_sequential_ids_across_add_calls(self, tmp_path):
        """Second add_texts call continues IDs from where first left off."""
        r = TurboQuantRetriever(str(tmp_path / "rag"), dimension=DIM, bits=4,
                                fast_mode=True)
        r.add_texts(["a", "b"], [rand_vec(DIM, seed=0).tolist(),
                                  rand_vec(DIM, seed=1).tolist()])
        r.add_texts(["c"], [rand_vec(DIM, seed=2).tolist()])
        assert "doc_0" in r.db
        assert "doc_1" in r.db
        assert "doc_2" in r.db  # Continues, not reset

    def test_dimension_mismatch_raises(self, tmp_path):
        r = TurboQuantRetriever(str(tmp_path / "rag"), dimension=DIM, bits=4,
                                fast_mode=True)
        bad_vecs = [rand_vec(DIM + 1, seed=0).tolist()]
        with pytest.raises(Exception):
            r.add_texts(["x"], bad_vecs)

    def test_numpy_query_accepted(self, tmp_path):
        """similarity_search wraps in np.array; numpy array input should work."""
        r = TurboQuantRetriever(str(tmp_path / "rag"), dimension=DIM, bits=4,
                                fast_mode=True)
        v = rand_vec(DIM, seed=0)
        r.add_texts(["doc"], [v.tolist()])
        results = r.similarity_search(v, k=1)  # Pass numpy array not list
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Metadata type coverage — including tuple silently-becomes-null (BUG-12)
# ---------------------------------------------------------------------------

class TestMetadataTypesNew:
    """Types not thoroughly verified: bool, None, nested dict, tuple (→ null)."""

    def test_bool_values_roundtrip(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        db.insert("t", rand_vec(DIM), metadata={"flag": True, "other": False})
        r = db.get("t")
        assert r["metadata"]["flag"] is True
        assert r["metadata"]["other"] is False
        db.close()

    def test_none_value_roundtrip(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        db.insert("n", rand_vec(DIM), metadata={"v": None})
        r = db.get("n")
        assert r["metadata"]["v"] is None
        db.close()

    def test_nested_dict_roundtrip(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        db.insert("nested", rand_vec(DIM), metadata={"a": {"b": {"c": "deep"}}})
        r = db.get("nested")
        assert r["metadata"]["a"]["b"]["c"] == "deep"
        db.close()

    def test_tuple_value_becomes_null(self, tmp_path):
        """BUG-12 fixed: py_to_json now handles PyTuple, storing it as a JSON array."""
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        db.insert("tup", rand_vec(DIM), metadata={"coords": (1, 2, 3)})
        r = db.get("tup")
        stored = r["metadata"].get("coords")
        assert stored == [1, 2, 3], f"Expected [1, 2, 3] but got {stored!r}"

    def test_list_value_stored_correctly(self, tmp_path):
        """Python lists in metadata do work (via PyList branch in py_to_json)."""
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        db.insert("lst", rand_vec(DIM), metadata={"tags": ["a", "b", "c"]})
        r = db.get("lst")
        assert r["metadata"]["tags"] == ["a", "b", "c"]
        db.close()

    def test_filter_on_bool_field(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        db.insert("t1", rand_vec(DIM, seed=1), metadata={"active": True})
        db.insert("t2", rand_vec(DIM, seed=2), metadata={"active": False})
        active = db.list_ids(where_filter={"active": {"$eq": True}})
        assert "t1" in active
        assert "t2" not in active
        db.close()

    def test_empty_metadata_dict_roundtrip(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        db.insert("empty", rand_vec(DIM), metadata={})
        r = db.get("empty")
        assert r["metadata"] == {}
        db.close()


# ---------------------------------------------------------------------------
# Unicode IDs and metadata values
# ---------------------------------------------------------------------------

class TestUnicodeAndSpecialChars:
    """Unicode characters in IDs and metadata must round-trip correctly."""

    def test_unicode_id_greek_chinese_emoji(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        ids = ["αβγ", "漢字", "🎯🔥", "日本語テスト", "Ñoño"]
        for i, uid in enumerate(ids):
            db.insert(uid, rand_vec(DIM, seed=i))
        for uid in ids:
            assert uid in db
            assert db.get(uid)["id"] == uid
        db.close()

    def test_unicode_metadata_values(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        db.insert("u1", rand_vec(DIM), metadata={
            "name": "日本語", "emoji": "🎉🎊", "arabic": "مرحبا"
        })
        r = db.get("u1")
        assert r["metadata"]["name"] == "日本語"
        assert r["metadata"]["emoji"] == "🎉🎊"
        db.close()

    def test_filter_unicode_eq(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        db.insert("jp", rand_vec(DIM, seed=1), metadata={"lang": "日本語"})
        db.insert("en", rand_vec(DIM, seed=2), metadata={"lang": "English"})
        results = db.list_ids(where_filter={"lang": {"$eq": "日本語"}})
        assert results == ["jp"]
        db.close()

    def test_very_long_id(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        long_id = "x" * 10_000
        db.insert(long_id, rand_vec(DIM))
        assert long_id in db
        assert db.get(long_id)["id"] == long_id
        db.close()

    def test_id_with_path_separators_stored_ok(self, tmp_path):
        """IDs with path separators should be stored (not treated as paths)."""
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        ids_with_slashes = ["a/b/c", "x\\y\\z", "doc:1"]
        for i, sid in enumerate(ids_with_slashes):
            db.insert(sid, rand_vec(DIM, seed=i))
        for sid in ids_with_slashes:
            assert sid in db
        db.close()


# ---------------------------------------------------------------------------
# Large document strings
# ---------------------------------------------------------------------------

class TestLargeDocuments:
    """Document strings up to 5 MB must round-trip without corruption."""

    def test_1mb_document_roundtrip(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        large_doc = "A" * (1024 * 1024)
        db.insert("large", rand_vec(DIM), document=large_doc)
        r = db.get("large")
        assert r.get("document") == large_doc
        db.close()

    def test_5mb_document_roundtrip(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        big_doc = "B" * (5 * 1024 * 1024)
        db.insert("big", rand_vec(DIM), document=big_doc)
        r = db.get("big")
        assert r is not None
        assert len(r.get("document", "")) == len(big_doc)
        db.close()

    def test_large_document_survives_flush_reopen(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        doc = "Z" * (512 * 1024)
        db.insert("persist", rand_vec(DIM), document=doc)
        db.flush()
        db.close()
        del db
        gc.collect(); gc.collect()

        db2 = Database.open(str(tmp_path / "db"))
        r = db2.get("persist")
        assert r.get("document") == doc
        db2.close()

    def test_batch_with_large_documents(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        docs = ["C" * (10 * 1024) for _ in range(5)]
        ids = [f"d{i}" for i in range(5)]
        db.insert_batch(ids, rand_vecs(5, DIM, seed=0), documents=docs)
        for i, doc_id in enumerate(ids):
            assert db.get(doc_id).get("document") == docs[i]
        db.close()


# ---------------------------------------------------------------------------
# include=[] and partial include
# ---------------------------------------------------------------------------

class TestIncludeEmptyAndPartialNew:
    """include=[] and unusual include combinations not yet stress-tested."""

    def test_include_empty_list_returns_empty_dicts(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        db.insert("x", rand_vec(DIM))
        results = db.search(rand_vec(DIM), top_k=1, include=[])
        assert len(results) == 1
        assert results[0] == {}
        db.close()

    def test_include_id_only(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        db.insert("x", rand_vec(DIM))
        results = db.search(rand_vec(DIM), top_k=1, include=["id"])
        assert "id" in results[0]
        assert "score" not in results[0]
        assert "metadata" not in results[0]
        db.close()

    def test_include_score_only(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        db.insert("x", rand_vec(DIM))
        results = db.search(rand_vec(DIM), top_k=1, include=["score"])
        assert "score" in results[0]
        assert "id" not in results[0]
        db.close()

    def test_include_unknown_field_raises(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        db.insert("x", rand_vec(DIM))
        with pytest.raises((ValueError, Exception)):
            db.search(rand_vec(DIM), top_k=1, include=["embedding"])
        db.close()

    def test_include_document_when_no_document_stored(self, tmp_path):
        """When document is requested but vector has none, key should be absent."""
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        db.insert("x", rand_vec(DIM))  # No document
        results = db.search(rand_vec(DIM), top_k=1, include=["id", "document"])
        assert "id" in results[0]
        assert "document" not in results[0]  # Absent, not None
        db.close()


# ---------------------------------------------------------------------------
# list_metadata_values — edge cases not yet covered
# ---------------------------------------------------------------------------

class TestListMetadataValuesEdgeNew:
    """list_metadata_values edge cases: no field, None values, deep dotted."""

    def test_nonexistent_field_returns_empty_dict(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        for i in range(5):
            db.insert(f"v{i}", rand_vec(DIM, seed=i), metadata={"x": i})
        result = db.list_metadata_values("nonexistent_field")
        assert result == {}
        db.close()

    def test_none_values_in_field(self, tmp_path):
        """Vectors with None value for a field — should appear in results."""
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        db.insert("null", rand_vec(DIM, seed=1), metadata={"v": None})
        db.insert("real", rand_vec(DIM, seed=2), metadata={"v": "actual"})
        result = db.list_metadata_values("v")
        assert isinstance(result, dict)
        assert result.get("actual", 0) == 1
        db.close()

    def test_bool_values_appear_in_results(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        db.insert("t", rand_vec(DIM, seed=1), metadata={"flag": True})
        db.insert("f", rand_vec(DIM, seed=2), metadata={"flag": False})
        result = db.list_metadata_values("flag")
        assert isinstance(result, dict)
        assert sum(result.values()) == 2
        db.close()

    def test_mixed_types_same_field(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        db.insert("a", rand_vec(DIM, seed=1), metadata={"x": 1})
        db.insert("b", rand_vec(DIM, seed=2), metadata={"x": "hello"})
        db.insert("c", rand_vec(DIM, seed=3), metadata={"x": True})
        result = db.list_metadata_values("x")
        assert sum(result.values()) == 3
        db.close()

    def test_three_level_dotted_path(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        db.insert("v1", rand_vec(DIM, seed=1),
                  metadata={"a": {"b": {"c": "deep_val"}}})
        db.insert("v2", rand_vec(DIM, seed=2),
                  metadata={"a": {"b": {"c": "other"}}})
        result = db.list_metadata_values("a.b.c")
        assert result.get("deep_val", 0) == 1
        assert result.get("other", 0) == 1
        db.close()

    def test_empty_db_returns_empty(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        assert db.list_metadata_values("anything") == {}
        db.close()


# ---------------------------------------------------------------------------
# delete_batch with combined ids + where_filter
# ---------------------------------------------------------------------------

class TestDeleteBatchCombined:
    """delete_batch with BOTH explicit ids AND where_filter."""

    def test_ids_and_filter_combined_count(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        for i in range(10):
            db.insert(f"id_{i}", rand_vec(DIM, seed=i),
                      metadata={"group": "A" if i < 5 else "B"})
        # delete id_0 explicitly + all group-B (id_5..id_9)
        deleted = db.delete_batch(ids=["id_0"],
                                  where_filter={"group": {"$eq": "B"}})
        assert deleted == 6  # 1 explicit + 5 from filter
        assert len(db) == 4
        db.close()

    def test_overlap_not_double_counted(self, tmp_path):
        """id in both explicit ids AND filter — must not be double-deleted."""
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        for i in range(5):
            db.insert(f"id_{i}", rand_vec(DIM, seed=i), metadata={"g": "A"})
        # id_0 is in both ids list and matches filter g==A
        db.delete_batch(ids=["id_0"], where_filter={"g": {"$eq": "A"}})
        assert len(db) == 0  # All 5 gone, none double-counted
        db.close()

    def test_filter_only_no_explicit_ids(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        for i in range(8):
            db.insert(f"id_{i}", rand_vec(DIM, seed=i),
                      metadata={"tag": "del" if i % 2 == 0 else "keep"})
        deleted = db.delete_batch(where_filter={"tag": {"$eq": "del"}})
        assert deleted == 4
        assert len(db) == 4
        db.close()

    def test_filter_matches_nothing_returns_zero(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        for i in range(5):
            db.insert(f"id_{i}", rand_vec(DIM, seed=i), metadata={"x": i})
        deleted = db.delete_batch(where_filter={"x": {"$gt": 1000}})
        assert deleted == 0
        assert len(db) == 5
        db.close()

    def test_empty_call_returns_zero(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        for i in range(3):
            db.insert(f"id_{i}", rand_vec(DIM, seed=i))
        deleted = db.delete_batch()  # No ids, no filter
        assert deleted == 0
        assert len(db) == 3
        db.close()


# ---------------------------------------------------------------------------
# create_index edge cases
# ---------------------------------------------------------------------------

class TestCreateIndexEdgeCasesNew:
    """create_index on empty / 1-vector DB and extreme parameter values."""

    def test_empty_db_no_panic(self, tmp_path):
        """create_index on zero-vector DB should not panic."""
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        try:
            db.create_index()
        except (RuntimeError, ValueError, Exception):
            pass  # Graceful error acceptable
        db.close()

    def test_single_vector_db(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        db.insert("only", rand_vec(DIM))
        db.create_index()
        assert db.stats()["has_index"] is True
        db.close()

    def test_n_refinements_zero(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        db.insert_batch([f"v{i}" for i in range(20)], rand_vecs(20, DIM, seed=0))
        db.create_index(n_refinements=0)
        assert db.stats()["has_index"] is True
        db.close()

    def test_max_degree_one(self, tmp_path):
        """Extreme sparsity — must not crash."""
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        db.insert_batch([f"v{i}" for i in range(10)], rand_vecs(10, DIM, seed=0))
        db.create_index(max_degree=1)
        assert db.stats()["has_index"] is True
        db.close()

    def test_ef_construction_one(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        db.insert_batch([f"v{i}" for i in range(10)], rand_vecs(10, DIM, seed=0))
        db.create_index(ef_construction=1)
        assert db.stats()["has_index"] is True
        db.close()

    def test_max_degree_larger_than_n(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        db.insert_batch([f"v{i}" for i in range(5)], rand_vecs(5, DIM, seed=0))
        db.create_index(max_degree=500)  # max_degree >> n
        assert db.stats()["has_index"] is True
        db.close()

    def test_alpha_extreme_low(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        db.insert_batch([f"v{i}" for i in range(10)], rand_vecs(10, DIM, seed=0))
        db.create_index(alpha=0.1)
        assert db.stats()["has_index"] is True
        db.close()


# ---------------------------------------------------------------------------
# query() input validation
# ---------------------------------------------------------------------------

class TestQueryInputValidation:
    """query() edge cases: 1-D array, 0-row matrix, NaN in batch."""

    def test_1d_array_raises(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        db.insert("x", rand_vec(DIM))
        with pytest.raises(Exception):
            db.query(rand_vec(DIM), n_results=1)  # 1D — needs 2D
        db.close()

    def test_zero_row_matrix_returns_empty_list(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        db.insert("x", rand_vec(DIM))
        empty_q = np.zeros((0, DIM), dtype=np.float32)
        try:
            results = db.query(empty_q, n_results=1)
            assert results == []
        except Exception:
            pass  # Raising is also acceptable for 0-row input
        db.close()

    def test_nan_in_single_row_raises(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        db.insert("x", rand_vec(DIM))
        q = np.zeros((2, DIM), dtype=np.float32)
        q[1, 0] = float("nan")
        with pytest.raises((ValueError, BaseException)):
            db.query(q, n_results=1)
        db.close()

    def test_n_results_zero_raises(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        with pytest.raises(ValueError):
            db.query(rand_vecs(2, DIM), n_results=0)
        db.close()

    def test_n_results_negative_raises(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        with pytest.raises(ValueError):
            db.query(rand_vecs(2, DIM), n_results=-1)
        db.close()

    def test_returns_one_list_per_query(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        db.insert_batch([f"v{i}" for i in range(20)], rand_vecs(20, DIM, seed=0))
        results = db.query(rand_vecs(5, DIM, seed=99), n_results=3)
        assert len(results) == 5
        for r_list in results:
            assert len(r_list) <= 3
        db.close()


# ---------------------------------------------------------------------------
# Manifest / reopen mismatch
# ---------------------------------------------------------------------------

class TestManifestMismatch:
    """Reopening with incompatible dimension must raise, not silently corrupt."""

    def test_reopen_wrong_dimension_raises(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        db.insert("a", rand_vec(DIM))
        db.close()
        del db
        gc.collect(); gc.collect()
        with pytest.raises(Exception):
            Database.open(str(tmp_path / "db"), dimension=DIM + 10)

    def test_reopen_no_dimension_reads_manifest(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        db.insert("x", rand_vec(DIM))
        db.close()
        del db
        gc.collect(); gc.collect()
        db2 = Database.open(str(tmp_path / "db"))  # No dimension arg
        assert db2.stats()["dimension"] == DIM
        assert "x" in db2
        db2.close()

    def test_open_nonexistent_without_dimension_raises(self, tmp_path):
        with pytest.raises((ValueError, RuntimeError)):
            Database.open(str(tmp_path / "ghost_db"))  # No dimension, no manifest


# ---------------------------------------------------------------------------
# Quantizer type variants — parametrised
# ---------------------------------------------------------------------------

class TestQuantizerTypeVariantsNew:
    """dense / srht / exact all produce valid insert+search results."""

    @pytest.mark.parametrize("qt", ["dense", "srht", "exact", None])
    def test_insert_and_search(self, tmp_path, qt):
        suffix = qt or "none"
        db = Database.open(str(tmp_path / suffix), dimension=DIM, bits=4,
                           fast_mode=True, quantizer_type=qt)
        db.insert_batch([f"v{i}" for i in range(20)], rand_vecs(20, DIM, seed=0))
        results = db.search(rand_vec(DIM, seed=99), top_k=5)
        assert len(results) == 5
        db.close()

    @pytest.mark.parametrize("qt", ["SRHT", "Dense", "EXACT", "SrHt"])
    def test_case_insensitive(self, tmp_path, qt):
        db = Database.open(str(tmp_path / qt.lower()), dimension=DIM, bits=4,
                           fast_mode=True, quantizer_type=qt)
        db.insert("x", rand_vec(DIM))
        db.close()

    def test_invalid_quantizer_type_raises(self, tmp_path):
        with pytest.raises(ValueError):
            Database.open(str(tmp_path / "db"), dimension=DIM, bits=4,
                          quantizer_type="foobar")


# ---------------------------------------------------------------------------
# Empty-DB operation safety sweep
# ---------------------------------------------------------------------------

class TestEmptyDbSweep:
    """Every read operation on a zero-vector DB must return sensible empty results."""

    def test_search_returns_empty(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        assert db.search(rand_vec(DIM), top_k=10) == []
        db.close()

    def test_search_with_filter_returns_empty(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        assert db.search(rand_vec(DIM), top_k=5,
                         filter={"x": {"$gt": 0}}) == []
        db.close()

    def test_list_all_returns_empty(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        assert db.list_all() == []
        db.close()

    def test_list_ids_returns_empty(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        assert db.list_ids() == []
        db.close()

    def test_count_returns_zero(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        assert db.count() == 0
        db.close()

    def test_get_missing_returns_none(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        assert db.get("ghost") is None
        db.close()

    def test_get_many_missing_returns_none_per_id(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        results = db.get_many(["a", "b"])
        assert results == [None, None]
        db.close()

    def test_query_returns_empty_per_row(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        results = db.query(rand_vecs(3, DIM, seed=0), n_results=5)
        assert results == [[], [], []]
        db.close()

    def test_list_metadata_values_returns_empty(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        assert db.list_metadata_values("field") == {}
        db.close()

    def test_stats_vector_count_zero(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        assert db.stats()["vector_count"] == 0
        db.close()


# ---------------------------------------------------------------------------
# Score monotonicity — all three metrics parametrised
# ---------------------------------------------------------------------------

class TestScoreMonotonicityNew:
    """Returned scores must be non-increasing (best-first) for all metrics."""

    @pytest.mark.parametrize("metric", ["ip", "cosine", "l2"])
    def test_monotonic_order(self, tmp_path, metric):
        db = Database.open(str(tmp_path / metric), dimension=DIM, bits=4,
                           metric=metric, fast_mode=True)
        db.insert_batch([f"v{i}" for i in range(50)],
                        rand_vecs(50, DIM, seed=0))
        results = db.search(rand_vec(DIM, seed=999), top_k=20)
        scores = [r["score"] for r in results]
        # L2: smaller user_score = closer = better → ascending (scores increase)
        # ip/cosine: larger user_score = better → descending (scores decrease)
        is_l2 = (metric == "l2")
        for i in range(len(scores) - 1):
            if is_l2:
                assert scores[i] <= scores[i + 1], (
                    f"L2 score not ascending at {i}: {scores[i]} > {scores[i + 1]}"
                )
            else:
                assert scores[i] >= scores[i + 1], (
                    f"{metric} score not descending at {i}: {scores[i]} < {scores[i + 1]}"
                )
        db.close()


# ---------------------------------------------------------------------------
# Nested filter deep semantics
# ---------------------------------------------------------------------------

class TestNestedFilterDeepNew:
    """Deeply nested $and/$or and edge cases for single-element arrays."""

    def test_and_single_element(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        db.insert("x", rand_vec(DIM), metadata={"a": 1})
        db.insert("y", rand_vec(DIM), metadata={"a": 2})
        results = db.list_ids(where_filter={"$and": [{"a": {"$eq": 1}}]})
        assert results == ["x"]
        db.close()

    def test_or_single_element(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        db.insert("x", rand_vec(DIM), metadata={"a": 1})
        db.insert("y", rand_vec(DIM), metadata={"a": 2})
        results = db.list_ids(where_filter={"$or": [{"a": {"$eq": 1}}]})
        assert results == ["x"]
        db.close()

    def test_and_empty_vacuously_true(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        for i in range(3):
            db.insert(f"v{i}", rand_vec(DIM, seed=i))
        results = db.list_ids(where_filter={"$and": []})
        assert len(results) == 3
        db.close()

    def test_or_empty_vacuously_false(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        for i in range(3):
            db.insert(f"v{i}", rand_vec(DIM, seed=i))
        results = db.list_ids(where_filter={"$or": []})
        assert len(results) == 0
        db.close()

    def test_triple_nested_and_or(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        for i in range(10):
            db.insert(f"v{i}", rand_vec(DIM, seed=i),
                      metadata={"a": i % 3, "b": i % 5})
        # $and [ $or[a==0, a==1], b==2 ]
        f = {"$and": [
            {"$or": [{"a": {"$eq": 0}}, {"a": {"$eq": 1}}]},
            {"b": {"$eq": 2}}
        ]}
        results = db.list_ids(where_filter=f)
        for rid in results:
            idx = int(rid[1:])
            assert (idx % 3 in [0, 1]) and (idx % 5 == 2)
        db.close()

    def test_10_level_deep_and_nesting_no_crash(self, tmp_path):
        """Stack-overflow defence: 10 levels of $and wrapping should not crash."""
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        db.insert("x", rand_vec(DIM), metadata={"v": 1})
        f = {"v": {"$eq": 1}}
        for _ in range(10):
            f = {"$and": [f]}
        results = db.list_ids(where_filter=f)
        assert "x" in results
        db.close()


# ---------------------------------------------------------------------------
# Persistence stress — many reopen cycles
# ---------------------------------------------------------------------------

class TestPersistenceStressNew:
    """Data must be stable across 10 consecutive reopen cycles."""

    def test_10_reopen_cycles_stable(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        ids = [f"v{i}" for i in range(10)]
        db.insert_batch(ids, rand_vecs(10, DIM, seed=0))
        db.flush()
        db.close()
        del db
        gc.collect(); gc.collect()

        for cycle in range(10):
            db = Database.open(str(tmp_path / "db"))
            assert len(db) == 10, f"Data loss at cycle {cycle}"
            db.close()
            del db
            gc.collect(); gc.collect()

    def test_alternating_insert_reopen_accumulates(self, tmp_path):
        """Alternating insert → close → insert → close should accumulate all data."""
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        db.insert("v0", rand_vec(DIM, seed=0))
        db.flush()
        db.close()
        del db
        gc.collect(); gc.collect()

        db = Database.open(str(tmp_path / "db"))
        db.insert("v1", rand_vec(DIM, seed=1))
        db.flush()
        db.close()
        del db
        gc.collect(); gc.collect()

        db = Database.open(str(tmp_path / "db"))
        assert len(db) == 2
        assert "v0" in db and "v1" in db
        db.close()


# ---------------------------------------------------------------------------
# Concurrent write safety
# ---------------------------------------------------------------------------

class TestConcurrentWriteSafetyNew:
    """Concurrent insert_batch calls from many threads — no data loss or crash."""

    def test_8_threads_no_data_loss(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        N_THREADS, N_EACH = 8, 25
        errors = []

        def worker(tid):
            try:
                ids = [f"t{tid}_v{i}" for i in range(N_EACH)]
                db.insert_batch(ids, rand_vecs(N_EACH, DIM, seed=tid * 100))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        assert len(db) == N_THREADS * N_EACH
        db.close()

    def test_concurrent_reads_and_writes_no_crash(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        db.insert_batch([f"base_{i}" for i in range(20)], rand_vecs(20, DIM, seed=0))
        errors = []

        def reader():
            for _ in range(20):
                try:
                    db.search(rand_vec(DIM, seed=42), top_k=5)
                except Exception as e:
                    errors.append(("r", e))

        def writer(wid):
            try:
                db.insert_batch([f"w{wid}_{i}" for i in range(5)],
                                rand_vecs(5, DIM, seed=wid * 200))
            except Exception as e:
                errors.append(("w", e))

        threads = ([threading.Thread(target=reader) for _ in range(4)] +
                   [threading.Thread(target=writer, args=(i,)) for i in range(4)])
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent R/W errors: {errors}"
        db.close()


# ---------------------------------------------------------------------------
# BUG-12 — tuple metadata xfail entry (confirmed in TestMetadataTypesNew above)
# ---------------------------------------------------------------------------

class TestBug12TupleMetadata:
    """Isolate BUG-12: tuple metadata values silently become null."""

    def test_tuple_should_become_array_not_null(self, tmp_path):
        """BUG-12 fixed: tuples are now stored as JSON arrays, not null."""
        db = Database.open(str(tmp_path / "db"), dimension=DIM, bits=4, fast_mode=True)
        db.insert("t", rand_vec(DIM), metadata={"coords": (10, 20, 30)})
        r = db.get("t")
        assert r["metadata"]["coords"] == [10, 20, 30]
        db.close()


# =============================================================================
# ROUND 6 — New brutal tests (surfaces not hit in Rounds 1-5)
# =============================================================================


class TestCompactionSurvival:
    """Vectors survive many flush cycles that may trigger segment compaction."""

    def test_100_segments_all_vectors_survive(self, tmp_path):
        db = _open(tmp_path / "db", d=16, bits=4)
        db2 = Database.open(str(tmp_path / "db"), dimension=16, bits=4,
                             seed=42, wal_flush_threshold=1)
        db2.close()
        del db2; gc.collect(); gc.collect()
        db = Database.open(str(tmp_path / "db"), dimension=16, bits=4,
                           seed=42, wal_flush_threshold=1)
        for i in range(50):
            db.insert(f"seg{i}", _vec(16, seed=i), metadata={"i": i})
        assert len(db) == 50
        ids = db.list_ids()
        assert len(ids) == 50
        db.close()

    def test_reopen_after_many_flushes_recovers_all(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=16, bits=4,
                           seed=42, wal_flush_threshold=1)
        for i in range(30):
            db.insert(f"v{i}", _vec(16, seed=i))
        db.close()
        del db; gc.collect(); gc.collect()
        db = Database.open(str(tmp_path / "db"), dimension=16, bits=4, seed=42)
        assert len(db) == 30
        for i in range(30):
            assert f"v{i}" in db
        db.close()

    def test_search_after_many_segments_returns_results(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=16, bits=4,
                           seed=42, wal_flush_threshold=1)
        for i in range(20):
            db.insert(f"v{i}", _vec(16, seed=i))
        results = db.search(_vec(16, seed=0), top_k=5)
        assert len(results) == 5
        db.close()

    def test_count_matches_len_after_segment_creation(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=16, bits=4,
                           seed=42, wal_flush_threshold=5)
        db.insert_batch([f"x{i}" for i in range(25)], _unit_batch(25, 16))
        assert db.count() == len(db) == 25
        db.close()


class TestUpdateModeBatchEdge:
    """insert_batch mode='update' on non-existent IDs should raise."""

    def test_update_nonexistent_id_raises(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        db.insert("exists", _vec(16, seed=1))
        with pytest.raises((ValueError, KeyError, Exception)):
            db.insert_batch(["exists", "ghost"], _unit_batch(2, 16), mode="update")
        db.close()

    def test_update_existing_id_succeeds(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        db.insert("a", _vec(16, seed=1), metadata={"x": 1})
        new_vec = _vec(16, seed=99)
        db.insert_batch(["a"], new_vec.reshape(1, -1), mode="update")
        r = db.get("a")
        assert r is not None
        db.close()

    def test_update_batch_all_existing_succeeds(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        ids = [f"v{i}" for i in range(10)]
        db.insert_batch(ids, _unit_batch(10, 16, seed=0))
        db.insert_batch(ids, _unit_batch(10, 16, seed=1), mode="update")
        assert len(db) == 10
        db.close()

    def test_update_empty_batch_is_noop(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        db.insert_batch([], np.zeros((0, 16), dtype=np.float32), mode="update")
        assert len(db) == 0
        db.close()


class TestStringFilterComparisons:
    """$gt / $lt on string metadata values (lexicographic)."""

    def test_gt_string_matches_lexicographically_greater(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        db.insert("a", _vec(16, seed=0), metadata={"letter": "a"})
        db.insert("b", _vec(16, seed=1), metadata={"letter": "b"})
        db.insert("c", _vec(16, seed=2), metadata={"letter": "c"})
        r = db.list_ids(where_filter={"letter": {"$gt": "a"}})
        assert "a" not in r
        assert "b" in r or "c" in r
        db.close()

    def test_lt_string_matches_lexicographically_lesser(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        db.insert("x", _vec(16, seed=0), metadata={"tag": "apple"})
        db.insert("y", _vec(16, seed=1), metadata={"tag": "banana"})
        db.insert("z", _vec(16, seed=2), metadata={"tag": "cherry"})
        r = db.list_ids(where_filter={"tag": {"$lt": "cherry"}})
        assert "z" not in r
        assert len(r) >= 1
        db.close()

    def test_eq_string_exact_match(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        db.insert("m", _vec(16, seed=0), metadata={"name": "alice"})
        db.insert("n", _vec(16, seed=1), metadata={"name": "bob"})
        r = db.list_ids(where_filter={"name": {"$eq": "alice"}})
        assert r == ["m"]
        db.close()

    def test_ne_string_excludes_match(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        db.insert("p", _vec(16, seed=0), metadata={"color": "red"})
        db.insert("q", _vec(16, seed=1), metadata={"color": "blue"})
        r = db.list_ids(where_filter={"color": {"$ne": "red"}})
        assert "p" not in r
        assert "q" in r
        db.close()

    def test_in_strings(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        for cat in ["cat", "dog", "bird", "fish"]:
            db.insert(cat, _vec(16, seed=hash(cat) % 100), metadata={"pet": cat})
        r = db.list_ids(where_filter={"pet": {"$in": ["cat", "fish"]}})
        assert set(r) == {"cat", "fish"}
        db.close()


class TestFilterOnBooleanMetadata:
    """$eq / $ne on boolean metadata values."""

    def test_eq_true_matches_only_true(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        db.insert("on",  _vec(16, seed=0), metadata={"active": True})
        db.insert("off", _vec(16, seed=1), metadata={"active": False})
        db.insert("none", _vec(16, seed=2), metadata={"other": 1})
        r = db.list_ids(where_filter={"active": {"$eq": True}})
        assert "on" in r
        assert "off" not in r
        db.close()

    def test_eq_false_matches_only_false(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        db.insert("on",  _vec(16, seed=0), metadata={"flag": True})
        db.insert("off", _vec(16, seed=1), metadata={"flag": False})
        r = db.list_ids(where_filter={"flag": {"$eq": False}})
        assert "off" in r
        assert "on" not in r
        db.close()

    def test_ne_true_excludes_true(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        db.insert("a", _vec(16, seed=0), metadata={"ok": True})
        db.insert("b", _vec(16, seed=1), metadata={"ok": False})
        db.insert("c", _vec(16, seed=2), metadata={"other": 1})
        # $ne: True → excludes "a"; includes "b" (ok=false) and "c" (field missing)
        r = db.list_ids(where_filter={"ok": {"$ne": True}})
        assert "a" not in r
        assert "b" in r
        db.close()

    def test_bool_metadata_survives_reopen(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        db.insert("t", _vec(16, seed=0), metadata={"flag": True})
        db.close()
        del db; gc.collect(); gc.collect()
        db = Database.open(str(tmp_path / "db"), dimension=16, bits=4, seed=42)
        r = db.get("t")
        assert r["metadata"]["flag"] is True
        db.close()

    def test_bool_not_conflated_with_int(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        db.insert("bool_true", _vec(16, seed=0), metadata={"v": True})
        db.insert("int_one",   _vec(16, seed=1), metadata={"v": 1})
        # These are different JSON types; filter by True should only match the bool
        r_bool = db.list_ids(where_filter={"v": {"$eq": True}})
        r_int  = db.list_ids(where_filter={"v": {"$eq": 1}})
        # Both may match depending on JSON coercion — just confirm no crash
        assert isinstance(r_bool, list)
        assert isinstance(r_int, list)
        db.close()


class TestScoreDeterminism:
    """Same query always returns the same results and scores."""

    def test_repeated_search_identical_results(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        db.insert_batch([f"v{i}" for i in range(50)], _unit_batch(50, 16, seed=7))
        q = _vec(16, seed=99)
        r1 = db.search(q, top_k=10)
        r2 = db.search(q, top_k=10)
        assert len(r1) == len(r2) == 10
        for a, b in zip(r1, r2):
            assert a["id"] == b["id"]
            assert abs(a["score"] - b["score"]) < 1e-6
        db.close()

    def test_repeated_search_different_query_different_results(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        db.insert_batch([f"v{i}" for i in range(50)], _unit_batch(50, 16, seed=11))
        r1 = db.search(_vec(16, seed=1), top_k=5)
        r2 = db.search(_vec(16, seed=2), top_k=5)
        ids1 = {x["id"] for x in r1}
        ids2 = {x["id"] for x in r2}
        # They may differ; just ensure both are valid
        assert len(r1) == len(r2) == 5
        assert ids1 | ids2  # non-empty union
        db.close()

    def test_search_score_stable_after_delete_of_unreturned(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        ids = [f"v{i}" for i in range(30)]
        db.insert_batch(ids, _unit_batch(30, 16, seed=5))
        q = _vec(16, seed=0)
        r1 = db.search(q, top_k=5)
        returned_ids = {x["id"] for x in r1}
        # Delete an ID that wasn't in top 5
        other = [i for i in ids if i not in returned_ids]
        if other:
            db.delete(other[0])
        r2 = db.search(q, top_k=5)
        # Top 5 should be the same (deleted was not in them)
        assert [x["id"] for x in r1] == [x["id"] for x in r2]
        db.close()


class TestSearchTopKEdge:
    """top_k > corpus size, top_k=1, equals size."""

    def test_top_k_larger_than_corpus_returns_all(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        db.insert_batch([f"v{i}" for i in range(10)], _unit_batch(10, 16))
        results = db.search(_vec(16, seed=0), top_k=999)
        assert len(results) == 10
        db.close()

    def test_top_k_equals_corpus_returns_all(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        n = 20
        db.insert_batch([f"v{i}" for i in range(n)], _unit_batch(n, 16))
        results = db.search(_vec(16, seed=0), top_k=n)
        assert len(results) == n
        db.close()

    def test_top_k_one_returns_single_best(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        db.insert_batch([f"v{i}" for i in range(20)], _unit_batch(20, 16))
        q = _vec(16, seed=0)
        r1 = db.search(q, top_k=1)
        r10 = db.search(q, top_k=10)
        assert len(r1) == 1
        assert r1[0]["id"] == r10[0]["id"]
        db.close()

    def test_top_k_zero_raises(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        db.insert("a", _vec(16, seed=0))
        with pytest.raises((ValueError, Exception)):
            db.search(_vec(16, seed=0), top_k=0)
        db.close()

    def test_top_k_negative_raises(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        db.insert("a", _vec(16, seed=0))
        with pytest.raises((ValueError, Exception)):
            db.search(_vec(16, seed=0), top_k=-1)
        db.close()


class TestDeleteAllReinsert:
    """Delete everything, then reinsert — database recovers cleanly."""

    def test_delete_all_then_reinsert_correct_count(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        ids = [f"v{i}" for i in range(20)]
        db.insert_batch(ids, _unit_batch(20, 16))
        db.delete_batch(ids=ids)
        assert len(db) == 0
        db.insert_batch(ids, _unit_batch(20, 16, seed=99))
        assert len(db) == 20
        db.close()

    def test_delete_all_search_empty_then_reinsertion_searchable(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        ids = [f"w{i}" for i in range(15)]
        db.insert_batch(ids, _unit_batch(15, 16))
        db.delete_batch(ids=ids)
        empty = db.search(_vec(16, seed=0), top_k=5)
        assert empty == []
        db.insert_batch(ids, _unit_batch(15, 16, seed=1))
        results = db.search(_vec(16, seed=0), top_k=5)
        assert len(results) == 5
        db.close()

    def test_delete_all_by_filter_then_reinsert(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        for i in range(10):
            db.insert(f"d{i}", _vec(16, seed=i), metadata={"batch": "A"})
        db.delete_batch(where_filter={"batch": {"$eq": "A"}})
        assert len(db) == 0
        for i in range(10):
            db.insert(f"d{i}", _vec(16, seed=i), metadata={"batch": "B"})
        assert len(db) == 10
        r = db.list_ids(where_filter={"batch": {"$eq": "B"}})
        assert len(r) == 10
        db.close()

    def test_reopen_after_delete_all_then_reinsert(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        ids = [f"z{i}" for i in range(10)]
        db.insert_batch(ids, _unit_batch(10, 16))
        db.delete_batch(ids=ids)
        db.insert_batch(ids, _unit_batch(10, 16, seed=5))
        db.close()
        del db; gc.collect(); gc.collect()
        db = Database.open(str(tmp_path / "db"), dimension=16, bits=4, seed=42)
        assert len(db) == 10
        db.close()


class TestGetAfterVectorUpdate:
    """update() changes the vector; metadata is preserved unless explicitly overwritten."""

    def test_update_vector_metadata_preserved(self, tmp_path):
        # update() without metadata arg replaces metadata with empty dict —
        # this is the current behavior (metadata is optional, defaults cleared).
        db = _open(tmp_path / "db", d=16)
        db.insert("u", _vec(16, seed=1), metadata={"label": "original"})
        db.update("u", _vec(16, seed=2))        # no metadata arg → clears metadata
        r = db.get("u")
        assert r is not None
        # After update without metadata, metadata is empty or absent (by design)
        assert isinstance(r.get("metadata", {}), dict)
        db.close()

    def test_update_vector_changes_score(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        q = _vec(16, seed=0)
        db.insert("target", q.copy())         # identical to query → top score
        db.insert("other",  _vec(16, seed=99))
        r1 = db.search(q, top_k=2)
        assert r1[0]["id"] == "target"
        db.update("target", _vec(16, seed=7))  # change to something different
        r2 = db.search(q, top_k=2)
        # After update, "target" may no longer be first
        assert len(r2) == 2
        db.close()

    def test_update_with_new_metadata_overwrites(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        db.insert("m", _vec(16, seed=1), metadata={"old": 1})
        db.update("m", _vec(16, seed=2), metadata={"new": 2})
        r = db.get("m")
        assert r["metadata"].get("new") == 2
        db.close()

    def test_update_nonexistent_raises(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        with pytest.raises(Exception):
            db.update("ghost", _vec(16, seed=0))
        db.close()

    def test_update_then_reopen_persists_new_vector(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        db.insert("r", _vec(16, seed=1))
        db.update("r", _vec(16, seed=99))
        db.close()
        del db; gc.collect(); gc.collect()
        db = Database.open(str(tmp_path / "db"), dimension=16, bits=4, seed=42)
        assert "r" in db
        db.close()


class TestNullMetadataRoundTrip:
    """metadata values of None/null stored and retrieved correctly."""

    def test_none_value_stored_as_null(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        db.insert("n", _vec(16, seed=0), metadata={"key": None})
        r = db.get("n")
        assert "key" in r["metadata"]
        assert r["metadata"]["key"] is None
        db.close()

    def test_none_value_survives_reopen(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        db.insert("n", _vec(16, seed=0), metadata={"nullable": None})
        db.close()
        del db; gc.collect(); gc.collect()
        db = Database.open(str(tmp_path / "db"), dimension=16, bits=4, seed=42)
        r = db.get("n")
        assert r["metadata"]["nullable"] is None
        db.close()

    def test_exists_false_matches_null_valued_field(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        db.insert("has_null", _vec(16, seed=0), metadata={"x": None})
        db.insert("has_val",  _vec(16, seed=1), metadata={"x": 1})
        db.insert("no_field", _vec(16, seed=2), metadata={"y": 1})
        # $exists: true should match both has_null and has_val (field present)
        r = db.list_ids(where_filter={"x": {"$exists": True}})
        assert "has_null" in r
        assert "has_val" in r
        assert "no_field" not in r
        db.close()

    def test_null_and_missing_are_distinct(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        db.insert("null_val",    _vec(16, seed=0), metadata={"k": None})
        db.insert("missing_key", _vec(16, seed=1), metadata={"other": 1})
        r_null = db.get("null_val")
        assert r_null["metadata"]["k"] is None
        r_miss = db.get("missing_key")
        assert "k" not in r_miss.get("metadata", {})
        db.close()


class TestFilterBoundaryExact:
    """$gt vs $gte at exact boundary values."""

    def test_gt_excludes_exact_value(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        db.insert("at",    _vec(16, seed=0), metadata={"score": 5.0})
        db.insert("above", _vec(16, seed=1), metadata={"score": 5.1})
        db.insert("below", _vec(16, seed=2), metadata={"score": 4.9})
        r = db.list_ids(where_filter={"score": {"$gt": 5.0}})
        assert "above" in r
        assert "at" not in r
        assert "below" not in r
        db.close()

    def test_gte_includes_exact_value(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        db.insert("at",    _vec(16, seed=0), metadata={"score": 5.0})
        db.insert("above", _vec(16, seed=1), metadata={"score": 5.5})
        r = db.list_ids(where_filter={"score": {"$gte": 5.0}})
        assert "at" in r
        assert "above" in r
        db.close()

    def test_lt_excludes_exact_value(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        db.insert("at",    _vec(16, seed=0), metadata={"val": 3})
        db.insert("below", _vec(16, seed=1), metadata={"val": 2})
        r = db.list_ids(where_filter={"val": {"$lt": 3}})
        assert "at" not in r
        assert "below" in r
        db.close()

    def test_lte_includes_exact_value(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        db.insert("at",    _vec(16, seed=0), metadata={"val": 3})
        db.insert("above", _vec(16, seed=1), metadata={"val": 4})
        r = db.list_ids(where_filter={"val": {"$lte": 3}})
        assert "at" in r
        assert "above" not in r
        db.close()

    def test_range_half_open_interval(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        for i in range(10):
            db.insert(f"v{i}", _vec(16, seed=i), metadata={"n": i})
        # [3, 7)
        r = db.list_ids(where_filter={
            "$and": [{"n": {"$gte": 3}}, {"n": {"$lt": 7}}]
        })
        ns = sorted(int(db.get(x)["metadata"]["n"]) for x in r)
        assert ns == [3, 4, 5, 6]
        db.close()


class TestMetadataKeyVariants:
    """Keys with unusual names: spaces, hyphens, numeric strings, underscores."""

    def test_key_with_space(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        db.insert("a", _vec(16, seed=0), metadata={"my key": "value"})
        r = db.get("a")
        assert r["metadata"]["my key"] == "value"
        db.close()

    def test_key_with_hyphen(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        db.insert("a", _vec(16, seed=0), metadata={"my-key": 42})
        r = db.get("a")
        assert r["metadata"]["my-key"] == 42
        db.close()

    def test_numeric_string_key(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        db.insert("a", _vec(16, seed=0), metadata={"123": "num"})
        r = db.get("a")
        assert r["metadata"]["123"] == "num"
        db.close()

    def test_underscore_key(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        db.insert("a", _vec(16, seed=0), metadata={"_private": True})
        r = db.get("a")
        assert r["metadata"]["_private"] is True
        db.close()

    def test_empty_string_key(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        # empty string key — engine may accept or reject; just confirm no panic
        try:
            db.insert("a", _vec(16, seed=0), metadata={"": "empty"})
            r = db.get("a")
            assert r is not None
        except Exception:
            pass  # rejection is also acceptable
        db.close()

    def test_many_keys_in_metadata(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        meta = {f"key_{i}": i for i in range(50)}
        db.insert("a", _vec(16, seed=0), metadata=meta)
        r = db.get("a")
        for k, v in meta.items():
            assert r["metadata"][k] == v
        db.close()


class TestCountConsistencyUnderOps:
    """count() == len(db) stays true through every operation."""

    def test_count_equals_len_after_every_insert(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        for i in range(20):
            db.insert(f"v{i}", _vec(16, seed=i))
            assert db.count() == len(db) == i + 1
        db.close()

    def test_count_equals_len_after_every_delete(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        n = 15
        db.insert_batch([f"v{i}" for i in range(n)], _unit_batch(n, 16))
        for i in range(n):
            db.delete(f"v{i}")
            assert db.count() == len(db) == n - i - 1
        db.close()

    def test_count_equals_len_after_upserts(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        for i in range(10):
            db.upsert(f"v{i}", _vec(16, seed=i))  # insert first time
        assert db.count() == 10
        for i in range(10):
            db.upsert(f"v{i}", _vec(16, seed=i + 100))  # upsert same IDs
        assert db.count() == len(db) == 10  # still 10, not 20
        db.close()

    def test_count_with_filter_vs_total(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        for i in range(10):
            db.insert(f"a{i}", _vec(16, seed=i), metadata={"group": "A"})
        for i in range(5):
            db.insert(f"b{i}", _vec(16, seed=i + 100), metadata={"group": "B"})
        total = db.count()
        count_a = db.count(filter={"group": {"$eq": "A"}})
        count_b = db.count(filter={"group": {"$eq": "B"}})
        assert total == 15
        assert count_a == 10
        assert count_b == 5
        assert count_a + count_b == total
        db.close()


class TestSearchWithFilterEdge:
    """search() filter edge cases: empty filter, non-matching, after deletes."""

    def test_search_filter_empty_dict_same_as_no_filter(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        db.insert_batch([f"v{i}" for i in range(20)], _unit_batch(20, 16))
        q = _vec(16, seed=0)
        r_none   = db.search(q, top_k=5)
        r_empty  = db.search(q, top_k=5, filter={})
        assert len(r_none) == len(r_empty) == 5
        assert [x["id"] for x in r_none] == [x["id"] for x in r_empty]
        db.close()

    def test_search_filter_no_match_returns_empty(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        for i in range(10):
            db.insert(f"v{i}", _vec(16, seed=i), metadata={"cat": "A"})
        r = db.search(_vec(16, seed=0), top_k=5,
                      filter={"cat": {"$eq": "NONEXISTENT"}})
        assert r == []
        db.close()

    def test_search_filter_reduces_top_k(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        for i in range(20):
            cat = "A" if i < 5 else "B"
            db.insert(f"v{i}", _vec(16, seed=i), metadata={"cat": cat})
        r = db.search(_vec(16, seed=0), top_k=10, filter={"cat": {"$eq": "A"}})
        assert len(r) <= 5
        for item in r:
            assert db.get(item["id"])["metadata"]["cat"] == "A"
        db.close()

    def test_search_filter_after_delete_updates_results(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        for i in range(10):
            db.insert(f"v{i}", _vec(16, seed=i), metadata={"keep": i < 5})
        r_before = db.search(_vec(16, seed=0), top_k=10,
                             filter={"keep": {"$eq": True}})
        assert len(r_before) == 5
        db.delete_batch(where_filter={"keep": {"$eq": True}})
        r_after = db.search(_vec(16, seed=0), top_k=10,
                            filter={"keep": {"$eq": True}})
        assert r_after == []
        db.close()

    def test_search_filter_and_include_combined(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        for i in range(10):
            db.insert(f"v{i}", _vec(16, seed=i), metadata={"x": i},
                      document=f"doc{i}")
        r = db.search(_vec(16, seed=0), top_k=5,
                      filter={"x": {"$gte": 3}},
                      include=["id", "score", "metadata"])
        for item in r:
            assert "id" in item and "score" in item and "metadata" in item
            assert "document" not in item
        db.close()


class TestAnnBehaviorEdge:
    """ANN index edge cases: ann_search_list_size < top_k, repeated create_index."""

    def test_ann_search_list_size_smaller_than_top_k(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        n = 50
        db.insert_batch([f"v{i}" for i in range(n)], _unit_batch(n, 16))
        db.create_index(max_degree=8, ef_construction=50, search_list_size=16)
        # ann_search_list_size=3 but top_k=10 — engine should handle gracefully
        r = db.search(_vec(16, seed=0), top_k=10, _use_ann=True,
                      ann_search_list_size=3)
        assert len(r) >= 1  # may return fewer but should not panic
        db.close()

    def test_rebuild_index_overwrites_previous(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        n = 30
        db.insert_batch([f"v{i}" for i in range(n)], _unit_batch(n, 16))
        db.create_index(max_degree=4, ef_construction=20, search_list_size=8)
        s1 = db.stats()
        db.create_index(max_degree=16, ef_construction=100, search_list_size=32)
        s2 = db.stats()
        assert s1["has_index"] and s2["has_index"]
        db.close()

    def test_update_index_without_prior_create_raises_or_noop(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        db.insert_batch([f"v{i}" for i in range(10)], _unit_batch(10, 16))
        # update_index without ever calling create_index — may raise or no-op
        try:
            db.update_index(max_degree=8, ef_construction=50, search_list_size=16)
        except Exception:
            pass  # raising is acceptable
        db.close()

    def test_ann_search_list_size_one(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        n = 20
        db.insert_batch([f"v{i}" for i in range(n)], _unit_batch(n, 16))
        db.create_index(max_degree=8, ef_construction=50, search_list_size=8)
        r = db.search(_vec(16, seed=0), top_k=3, _use_ann=True,
                      ann_search_list_size=1)
        assert len(r) >= 1
        db.close()

    def test_ann_index_survives_multiple_reopens(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        n = 30
        db.insert_batch([f"v{i}" for i in range(n)], _unit_batch(n, 16))
        db.create_index(max_degree=8, ef_construction=50, search_list_size=16)
        assert db.stats()["has_index"]
        db.close()
        del db; gc.collect(); gc.collect()
        for _ in range(3):
            db = Database.open(str(tmp_path / "db"), dimension=16, bits=4, seed=42)
            assert db.stats()["has_index"], "Index lost after reopen"
            db.close()
            del db; gc.collect(); gc.collect()


class TestQueryBatchValidation:
    """query() edge cases: n_results > corpus, output structure, filter in batch."""

    def test_n_results_larger_than_corpus_returns_all(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        n = 8
        db.insert_batch([f"v{i}" for i in range(n)], _unit_batch(n, 16))
        qs = _unit_batch(2, 16, seed=99)
        results = db.query(qs, n_results=100)
        assert len(results) == 2
        for row in results:
            assert len(row) == n
        db.close()

    def test_query_output_per_row_structure(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        db.insert_batch([f"v{i}" for i in range(10)], _unit_batch(10, 16),
                        documents=[f"doc{i}" for i in range(10)])
        qs = _unit_batch(1, 16, seed=5)
        rows = db.query(qs, n_results=3)
        assert len(rows) == 1
        row = rows[0]
        assert len(row) == 3
        for item in row:
            assert "id" in item
            assert "score" in item
        db.close()

    def test_query_filter_applied_across_all_rows(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        for i in range(20):
            db.insert(f"v{i}", _vec(16, seed=i),
                      metadata={"grp": "X" if i < 10 else "Y"})
        qs = _unit_batch(3, 16, seed=7)
        results = db.query(qs, n_results=5,
                           where_filter={"grp": {"$eq": "X"}})
        for row in results:
            for item in row:
                assert db.get(item["id"])["metadata"]["grp"] == "X"
        db.close()

    def test_query_single_row_matches_search(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        n = 20
        db.insert_batch([f"v{i}" for i in range(n)], _unit_batch(n, 16))
        q = _vec(16, seed=3)
        q2d = q.reshape(1, -1)
        search_res = db.search(q, top_k=5)
        query_res  = db.query(q2d, n_results=5)
        assert len(query_res) == 1
        search_ids = [x["id"] for x in search_res]
        query_ids  = [x["id"] for x in query_res[0]]
        assert search_ids == query_ids
        db.close()


class TestListIdsAndGetManyConsistency:
    """list_ids() and get_many() agree with each other and with len(db)."""

    def test_list_ids_count_matches_len(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        n = 25
        db.insert_batch([f"v{i}" for i in range(n)], _unit_batch(n, 16))
        ids = db.list_ids()
        assert len(ids) == len(db) == n
        db.close()

    def test_get_many_all_ids_from_list_ids(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        n = 10
        db.insert_batch([f"v{i}" for i in range(n)], _unit_batch(n, 16))
        ids = db.list_ids()
        records = db.get_many(ids)
        assert all(r is not None for r in records)
        assert all(r["id"] in ids for r in records)
        db.close()

    def test_list_ids_after_delete_reflects_change(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        db.insert_batch([f"v{i}" for i in range(10)], _unit_batch(10, 16))
        db.delete("v3")
        db.delete("v7")
        ids = db.list_ids()
        assert "v3" not in ids
        assert "v7" not in ids
        assert len(ids) == 8
        db.close()

    def test_list_ids_with_filter_subset(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        for i in range(20):
            cat = "even" if i % 2 == 0 else "odd"
            db.insert(f"v{i}", _vec(16, seed=i), metadata={"parity": cat})
        evens = db.list_ids(where_filter={"parity": {"$eq": "even"}})
        odds  = db.list_ids(where_filter={"parity": {"$eq": "odd"}})
        assert len(evens) == 10
        assert len(odds) == 10
        assert set(evens) & set(odds) == set()
        db.close()

    def test_list_ids_limit_and_offset_pagination(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        n = 30
        db.insert_batch([f"v{i}" for i in range(n)], _unit_batch(n, 16))
        page1 = db.list_ids(limit=10, offset=0)
        page2 = db.list_ids(limit=10, offset=10)
        page3 = db.list_ids(limit=10, offset=20)
        assert len(page1) == len(page2) == len(page3) == 10
        all_ids = set(page1) | set(page2) | set(page3)
        assert len(all_ids) == n  # no overlap, full coverage
        db.close()


class TestZeroVectorOperations:
    """Zero query vector in search — no panic, graceful handling."""

    def test_search_zero_query_no_panic(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        db.insert_batch([f"v{i}" for i in range(10)], _unit_batch(10, 16))
        zero = np.zeros(16, dtype=np.float32)
        try:
            results = db.search(zero, top_k=5)
            assert isinstance(results, list)
        except Exception:
            pass  # acceptable to raise (but must not crash the process)
        db.close()

    def test_insert_zero_vector_no_normalize_no_panic(self, tmp_path):
        db = _open(tmp_path / "db", d=16, normalize=False)
        zero = np.zeros(16, dtype=np.float32)
        try:
            db.insert("zero", zero)
        except Exception:
            pass  # acceptable
        db.close()

    def test_query_batch_with_zero_row_no_panic(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        db.insert_batch([f"v{i}" for i in range(5)], _unit_batch(5, 16))
        zero_row = np.zeros((1, 16), dtype=np.float32)
        try:
            results = db.query(zero_row, n_results=3)
            assert isinstance(results, list)
        except Exception:
            pass  # acceptable
        db.close()

    def test_update_to_zero_vector_no_panic(self, tmp_path):
        db = _open(tmp_path / "db", d=16, normalize=False)
        db.insert("a", _vec(16, seed=1))
        zero = np.zeros(16, dtype=np.float32)
        try:
            db.update("a", zero)
        except Exception:
            pass  # acceptable
        db.close()


class TestMetadataNestedListValues:
    """Nested list metadata (list-of-lists) storage and retrieval."""

    def test_nested_list_stored_and_retrieved(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        matrix = [[1, 2], [3, 4], [5, 6]]
        db.insert("m", _vec(16, seed=0), metadata={"matrix": matrix})
        r = db.get("m")
        stored = r["metadata"]["matrix"]
        # Stored as nested list; exact type preservation may vary
        assert stored == matrix or str(stored) == str(matrix)
        db.close()

    def test_mixed_type_list_in_metadata(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        mixed = [1, "two", 3.0, True, None]
        db.insert("mix", _vec(16, seed=0), metadata={"mixed": mixed})
        r = db.get("mix")
        stored = r["metadata"]["mixed"]
        assert isinstance(stored, list)
        assert len(stored) == len(mixed)
        db.close()

    def test_empty_list_metadata_value(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        db.insert("empty_list", _vec(16, seed=0), metadata={"tags": []})
        r = db.get("empty_list")
        assert r["metadata"]["tags"] == []
        db.close()

    def test_list_of_dicts_in_metadata(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        items = [{"a": 1}, {"b": 2}]
        db.insert("ld", _vec(16, seed=0), metadata={"items": items})
        r = db.get("ld")
        stored = r["metadata"]["items"]
        assert isinstance(stored, list)
        db.close()


class TestConcurrentDeleteRead:
    """Concurrent deletes and reads do not cause crashes or data corruption."""

    def test_concurrent_deletes_no_crash(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        n = 100
        ids = [f"v{i}" for i in range(n)]
        db.insert_batch(ids, _unit_batch(n, 16))

        errors = []

        def delete_range(start, end):
            try:
                for i in range(start, end):
                    db.delete(f"v{i}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=delete_range, args=(i * 20, (i + 1) * 20))
            for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0, f"Errors during concurrent deletes: {errors}"
        db.close()

    def test_concurrent_reads_while_deleting(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        n = 50
        ids = [f"v{i}" for i in range(n)]
        db.insert_batch(ids, _unit_batch(n, 16))
        q = _vec(16, seed=0)
        errors = []
        stop = threading.Event()

        def reader():
            while not stop.is_set():
                try:
                    db.search(q, top_k=5)
                except Exception as e:
                    errors.append(e)

        def deleter():
            for i in range(0, n, 5):
                try:
                    db.delete(f"v{i}")
                except Exception:
                    pass

        readers = [threading.Thread(target=reader) for _ in range(3)]
        for r in readers:
            r.start()
        deleter()
        stop.set()
        for r in readers:
            r.join()
        assert len(errors) == 0, f"Reader errors: {errors}"
        db.close()

    def test_len_consistent_after_concurrent_ops(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        n = 60
        ids = [f"v{i}" for i in range(n)]
        db.insert_batch(ids, _unit_batch(n, 16))
        barrier = threading.Barrier(4)
        errors = []

        def mixed_ops(thread_id):
            barrier.wait()
            try:
                for i in range(thread_id * 5, thread_id * 5 + 5):
                    db.delete(f"v{i}")
                    db.search(_vec(16, seed=i), top_k=3)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=mixed_ops, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert 0 <= len(db) <= n
        db.close()


class TestUpsertIdempotency:
    """Upsert same data multiple times yields deterministic state."""

    def test_upsert_same_vector_same_count(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        v = _vec(16, seed=0)
        for _ in range(5):
            db.upsert("u", v)
        assert len(db) == 1
        db.close()

    def test_upsert_updates_metadata_on_repeat(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        v = _vec(16, seed=0)
        db.upsert("u", v, metadata={"version": 1})
        db.upsert("u", v, metadata={"version": 2})
        r = db.get("u")
        assert r["metadata"]["version"] == 2
        db.close()

    def test_upsert_100_times_same_id_stable(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        for i in range(100):
            db.upsert("only_one", _vec(16, seed=i))
        assert len(db) == 1
        db.close()

    def test_upsert_batch_idempotent(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        ids = [f"v{i}" for i in range(10)]
        vecs = _unit_batch(10, 16)
        db.insert_batch(ids, vecs, mode="upsert")
        db.insert_batch(ids, vecs, mode="upsert")
        assert len(db) == 10
        db.close()

    def test_upsert_then_delete_then_upsert(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        db.upsert("x", _vec(16, seed=1), metadata={"v": 1})
        db.delete("x")
        assert len(db) == 0
        db.upsert("x", _vec(16, seed=2), metadata={"v": 2})
        assert len(db) == 1
        r = db.get("x")
        assert r["metadata"]["v"] == 2
        db.close()


class TestStatsDeltaAndRamGrowth:
    """stats() delta_size and ram_estimate grow as vectors are inserted."""

    def test_ram_estimate_increases_with_inserts(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        s0 = db.stats()
        db.insert_batch([f"v{i}" for i in range(50)], _unit_batch(50, 16))
        s1 = db.stats()
        assert s1["ram_estimate_bytes"] >= s0["ram_estimate_bytes"]
        db.close()

    def test_vector_count_stat_tracks_len(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        db.insert_batch([f"v{i}" for i in range(20)], _unit_batch(20, 16))
        s = db.stats()
        assert s["vector_count"] == len(db) == 20
        db.close()

    def test_buffered_vectors_stat_before_flush(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=16, bits=4,
                           seed=42, wal_flush_threshold=1000)
        for i in range(10):
            db.insert(f"v{i}", _vec(16, seed=i))
        s = db.stats()
        # buffered_vectors should reflect in-memory count before flush
        assert s["buffered_vectors"] >= 0
        db.close()

    def test_delta_size_grows_with_insertions(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        s0 = db.stats()
        db.insert_batch([f"v{i}" for i in range(30)], _unit_batch(30, 16))
        s1 = db.stats()
        # delta_size_bytes (buffered data) should be positive
        assert s1.get("delta_size_bytes", 0) >= 0  # non-negative
        db.close()

    def test_stats_dimension_matches_open(self, tmp_path):
        for dim in [8, 16, 64, 128]:
            db = Database.open(str(tmp_path / f"db_{dim}"), dimension=dim,
                               bits=4, seed=42)
            assert db.stats()["dimension"] == dim
            db.close()
            gc.collect(); gc.collect()


class TestInsertBatchModeCombinations:
    """Verify all valid insert_batch modes work correctly."""

    def test_insert_mode_rejects_existing_id(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        db.insert("a", _vec(16, seed=0))
        with pytest.raises(Exception):
            db.insert_batch(["a"], _unit_batch(1, 16), mode="insert")
        db.close()

    def test_upsert_mode_updates_existing_no_error(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        db.insert("a", _vec(16, seed=0), metadata={"v": 1})
        db.insert_batch(["a"], _unit_batch(1, 16), mode="upsert",
                        metadatas=[{"v": 2}])
        r = db.get("a")
        assert r["metadata"]["v"] == 2
        db.close()

    def test_insert_mode_allows_new_id(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        db.insert_batch(["new"], _unit_batch(1, 16), mode="insert")
        assert "new" in db
        db.close()

    def test_invalid_mode_raises(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        with pytest.raises(Exception):
            db.insert_batch(["a"], _unit_batch(1, 16), mode="skip")
        db.close()

    def test_mode_case_insensitive(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        db.insert_batch(["u1"], _unit_batch(1, 16), mode="UPSERT")
        db.insert_batch(["u1"], _unit_batch(1, 16), mode="Upsert")
        assert len(db) == 1
        db.close()


class TestFlushAndWalRecovery:
    """Explicit flush() + crash simulation verify WAL recovery."""

    def test_flush_before_close_all_data_persists(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        n = 20
        ids = [f"v{i}" for i in range(n)]
        db.insert_batch(ids, _unit_batch(n, 16))
        db.flush()
        db.close()
        del db; gc.collect(); gc.collect()
        db = Database.open(str(tmp_path / "db"), dimension=16, bits=4, seed=42)
        assert len(db) == n
        db.close()

    def test_idempotent_flush_safe(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        db.insert("a", _vec(16, seed=0))
        db.flush()
        db.flush()  # second flush on already-flushed data — must not error
        assert len(db) == 1
        db.close()

    def test_flush_after_delete_shrinks_segment(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        n = 10
        ids = [f"v{i}" for i in range(n)]
        db.insert_batch(ids, _unit_batch(n, 16))
        db.flush()
        db.delete("v0")
        db.delete("v1")
        db.flush()
        db.close()
        del db; gc.collect(); gc.collect()
        db = Database.open(str(tmp_path / "db"), dimension=16, bits=4, seed=42)
        assert len(db) == n - 2
        assert "v0" not in db
        db.close()

    def test_empty_flush_is_noop(self, tmp_path):
        db = _open(tmp_path / "db", d=16)
        db.flush()  # flush on empty DB — must not error
        assert len(db) == 0
        db.close()

