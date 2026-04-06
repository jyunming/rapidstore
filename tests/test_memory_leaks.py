"""
Memory leak tests for TurboQuantDB.

These tests verify that repeated insert/delete/reinsert cycles, open/close
cycles, and search loops do not cause unbounded RSS growth.

The metric used is process RSS (via psutil) rather than ram_estimate_bytes,
because these tests exist specifically to catch off-path leaks that
ram_estimate_bytes would not detect (e.g. temporary buffers not freed,
mmap handles not released, Python object cycles).

Allowance: 20 MB slack for Python allocator fragmentation and OS page
rounding — RSS may not drop back to exact baseline after deallocation.

Run with:
    pytest tests/test_memory_leaks.py -v
"""
from __future__ import annotations

import gc
import os
import tempfile

import numpy as np
import pytest

psutil = pytest.importorskip("psutil", reason="psutil not installed — skipping memory leak tests")

from tqdb import Database


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rss_mb() -> float:
    gc.collect()
    return psutil.Process(os.getpid()).memory_info().rss / (1 << 20)


def _unit_vecs(n: int, d: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, d)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


_LEAK_SLACK_MB = 20.0   # tolerated RSS growth per test


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestMemoryLeaks:
    def test_insert_delete_cycle_does_not_leak(self, tmp_path):
        """RSS after N insert+delete+flush cycles must not grow >20 MB."""
        d, n = 128, 2000
        ids = [f"v{i}" for i in range(n)]
        vecs = _unit_vecs(n, d)

        # Warm up Python/Rust allocators with one cycle before measuring
        with tempfile.TemporaryDirectory() as warm:
            db = Database.open(warm, d, bits=4, rerank=False, metric="ip")
            db.insert_batch(ids, vecs)
            db.flush()
            for vid in ids:
                db.delete(vid)
            db.flush()
            db.close()

        baseline_mb = _rss_mb()

        for cycle in range(5):
            cycle_dir = str(tmp_path / f"cycle_{cycle}")
            db = Database.open(cycle_dir, d, bits=4, rerank=False, metric="ip")
            db.insert_batch(ids, vecs)
            db.flush()
            for vid in ids:
                db.delete(vid)
            db.flush()
            db.close()

        after_mb = _rss_mb()
        growth_mb = after_mb - baseline_mb
        assert growth_mb < _LEAK_SLACK_MB, (
            f"RSS grew {growth_mb:.1f} MB across 5 insert+delete cycles "
            f"(baseline {baseline_mb:.1f} MB → {after_mb:.1f} MB)"
        )

    def test_open_close_cycle_does_not_leak(self, tmp_path):
        """Reopening the same database repeatedly must not leak RSS."""
        d, n = 128, 2000
        ids = [f"v{i}" for i in range(n)]
        vecs = _unit_vecs(n, d)
        db_path = str(tmp_path / "db")

        # Seed the database once
        db = Database.open(db_path, d, bits=4, rerank=True, metric="ip")
        db.insert_batch(ids, vecs)
        db.flush()
        db.close()

        # Warm up one open/close
        Database.open(db_path, d, bits=4, rerank=True, metric="ip").close()
        baseline_mb = _rss_mb()

        for _ in range(10):
            db = Database.open(db_path, d, bits=4, rerank=True, metric="ip")
            db.close()

        after_mb = _rss_mb()
        growth_mb = after_mb - baseline_mb
        assert growth_mb < _LEAK_SLACK_MB, (
            f"RSS grew {growth_mb:.1f} MB across 10 open/close cycles "
            f"(baseline {baseline_mb:.1f} MB → {after_mb:.1f} MB)"
        )

    def test_search_loop_does_not_leak(self, tmp_path):
        """Issuing many search queries must not grow RSS."""
        d, n, queries = 128, 2000, 200
        ids = [f"v{i}" for i in range(n)]
        vecs = _unit_vecs(n, d)
        qs = _unit_vecs(queries, d, seed=99)

        db = Database.open(str(tmp_path / "db"), d, bits=4, rerank=True, metric="ip")
        db.insert_batch(ids, vecs)
        db.flush()

        # Warm up
        for q in qs:
            db.search(q, top_k=10)
        baseline_mb = _rss_mb()

        for _ in range(5):
            for q in qs:
                db.search(q, top_k=10)

        db.close()
        after_mb = _rss_mb()
        growth_mb = after_mb - baseline_mb
        assert growth_mb < _LEAK_SLACK_MB, (
            f"RSS grew {growth_mb:.1f} MB across {5 * queries} searches "
            f"(baseline {baseline_mb:.1f} MB → {after_mb:.1f} MB)"
        )

    def test_ann_index_rebuild_does_not_leak(self, tmp_path):
        """Rebuilding the HNSW index multiple times must not leak RSS."""
        d, n = 128, 1000
        ids = [f"v{i}" for i in range(n)]
        vecs = _unit_vecs(n, d)

        def _build_index(cycle_path: str) -> None:
            db = Database.open(cycle_path, d, bits=4, rerank=False, metric="ip")
            db.insert_batch(ids, vecs)
            db.flush()
            db.create_index(max_degree=8, search_list_size=32)
            db.close()
            del db
            gc.collect()

        # Warm up one index build
        _build_index(str(tmp_path / "warm"))
        baseline_mb = _rss_mb()

        for i in range(5):
            _build_index(str(tmp_path / f"cycle_{i}"))

        after_mb = _rss_mb()
        growth_mb = after_mb - baseline_mb
        assert growth_mb < _LEAK_SLACK_MB, (
            f"RSS grew {growth_mb:.1f} MB across 5 HNSW index builds "
            f"(baseline {baseline_mb:.1f} MB → {after_mb:.1f} MB)"
        )
