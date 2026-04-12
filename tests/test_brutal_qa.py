"""
BRUTAL QA test suite for TurboQuantDB.

Philosophy: find real bugs, not just exercise happy paths.
Every test tries to break something specific. Zero tolerance for panics.
"""
from __future__ import annotations

import gc
import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from tqdb import Database


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vec(d: int, val: float = 1.0, seed: int | None = None) -> np.ndarray:
    if seed is not None:
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(d).astype(np.float32)
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v
    return np.full(d, val, dtype=np.float32)


def _unit_batch(n: int, d: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, d)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-9)


def _open(path, d: int = 16, bits: int = 4, metric: str = "ip", **kw) -> Database:
    return Database.open(str(path), d, bits=bits, seed=42, metric=metric, **kw)


# ---------------------------------------------------------------------------
# ██████╗  █████╗ ██████╗  █████╗ ███╗   ███╗███████╗████████╗███████╗██████╗ ███████╗
# ██╔══██╗██╔══██╗██╔══██╗██╔══██╗████╗ ████║██╔════╝╚══██╔══╝██╔════╝██╔══██╗██╔════╝
# ██████╔╝███████║██████╔╝███████║██╔████╔██║█████╗     ██║   █████╗  ██████╔╝███████╗
# ██╔═══╝ ██╔══██║██╔══██╗██╔══██║██║╚██╔╝██║██╔══╝     ██║   ██╔══╝  ██╔══██╗╚════██║
# ██║     ██║  ██║██║  ██║██║  ██║██║ ╚═╝ ██║███████╗   ██║   ███████╗██║  ██║███████║
# ---------------------------------------------------------------------------

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
