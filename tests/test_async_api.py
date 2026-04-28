"""Tests for the asyncio facade at :mod:`tqdb.aio`.

The wrapper has no Rust-side counterpart — the value-add is purely the
``run_in_executor`` dispatch + the right method shapes. Tests focus on:

1. **API shape**: every wrapped method round-trips a value the sync API would
   return.
2. **Concurrency**: many ``await`` calls genuinely run in parallel without
   blocking the event loop. The Rust code already releases the GIL inside
   each PyO3 method (``py.allow_threads``), so the only risk is the wrapper
   accidentally serializing things.
3. **Lifecycle**: ``async with`` works; ``close()`` shuts the auto-created
   executor.
"""

from __future__ import annotations

import asyncio
import tempfile
import time

import numpy as np
import pytest

from tqdb.aio import AsyncDatabase


@pytest.fixture
def tmp_db_path():
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp


def _vec(d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(d).astype(np.float32)


# ── basic API shape ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_open_insert_search_close(tmp_db_path):
    d = 16
    db = await AsyncDatabase.open(tmp_db_path, dimension=d, bits=2)
    try:
        await db.insert("a", _vec(d, 1), document="alpha")
        await db.insert("b", _vec(d, 2), document="beta")
        results = await db.search(_vec(d, 1), top_k=2)
        ids = {r["id"] for r in results}
        assert "a" in ids
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_async_context_manager(tmp_db_path):
    async with await AsyncDatabase.open(tmp_db_path, dimension=8, bits=2) as db:
        await db.insert("only", _vec(8, 7))
        assert "only" in db
        assert len(db) == 1
    # Executor should be shut after exit; calling close again must be safe.
    # (Don't error on double-close because cleanup ordering in tests is
    # sometimes unpredictable.)


@pytest.mark.asyncio
async def test_batch_and_filter(tmp_db_path):
    d = 8
    async with await AsyncDatabase.open(tmp_db_path, dimension=d, bits=2) as db:
        ids = [f"d{i}" for i in range(10)]
        vecs = np.stack([_vec(d, i) for i in range(10)])
        metas = [{"bucket": i % 3} for i in range(10)]
        await db.insert_batch(ids, vecs, metas, None, "insert")

        bucket0 = await db.list_ids(where_filter={"bucket": 0})
        assert len(bucket0) == 4  # 0, 3, 6, 9

        deleted = await db.delete_batch(where_filter={"bucket": 0})
        assert deleted == 4
        assert await db.count() == 6


@pytest.mark.asyncio
async def test_query_batch(tmp_db_path):
    d = 8
    async with await AsyncDatabase.open(tmp_db_path, dimension=d, bits=2) as db:
        for i in range(5):
            await db.insert(f"d{i}", _vec(d, i))
        emb = np.stack([_vec(d, 0), _vec(d, 1)])
        batch = await db.query(emb, n_results=3)
        assert len(batch) == 2
        assert all(len(r) <= 3 for r in batch)


@pytest.mark.asyncio
async def test_get_and_metadata_update(tmp_db_path):
    d = 8
    async with await AsyncDatabase.open(tmp_db_path, dimension=d, bits=2) as db:
        await db.insert("x", _vec(d, 1), {"v": 1}, "doc-x")
        rec = await db.get("x")
        assert rec["metadata"]["v"] == 1
        await db.update_metadata("x", {"v": 2})
        rec = await db.get("x")
        assert rec["metadata"]["v"] == 2


# ── concurrency ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_concurrent_searches_dont_serialize(tmp_db_path):
    """Fire 50 concurrent searches and confirm they don't run sequentially.

    The expected wall-clock is much less than (n_tasks × per-task latency);
    if the wrapper accidentally serialized, total time would scale linearly
    with task count.
    """
    d = 64
    async with await AsyncDatabase.open(tmp_db_path, dimension=d, bits=4) as db:
        # Populate with enough data that each search isn't a sub-microsecond no-op.
        ids = [f"d{i}" for i in range(2_000)]
        vecs = np.stack([_vec(d, i) for i in range(2_000)])
        await db.insert_batch(ids, vecs, None, None, "insert")

        # Warm-up so the first call's overhead doesn't skew the median.
        await db.search(_vec(d, 0), top_k=10)

        # Time a single search to set the bar.
        t0 = time.perf_counter()
        await db.search(_vec(d, 0), top_k=10)
        single_s = max(time.perf_counter() - t0, 1e-4)

        # Fire many concurrent searches.
        n_tasks = 50
        t0 = time.perf_counter()
        await asyncio.gather(
            *(db.search(_vec(d, i), top_k=10) for i in range(n_tasks))
        )
        concurrent_s = time.perf_counter() - t0

        # Sequential would take ~ n_tasks × single_s. Concurrent should be
        # WAY less — we accept up to a generous 0.5 × that as proof of
        # parallelism (covers slow CI / Windows scheduler jitter).
        sequential_estimate = n_tasks * single_s
        assert concurrent_s < 0.5 * sequential_estimate, (
            f"50 concurrent searches took {concurrent_s:.3f}s; sequential "
            f"would be ~{sequential_estimate:.3f}s. The wrapper appears to "
            f"serialize."
        )


@pytest.mark.asyncio
async def test_event_loop_remains_responsive_during_search(tmp_db_path):
    """While a long-ish search runs in the executor, an unrelated coroutine
    must still get scheduling slices."""
    d = 32
    async with await AsyncDatabase.open(tmp_db_path, dimension=d, bits=4) as db:
        ids = [f"d{i}" for i in range(1_000)]
        vecs = np.stack([_vec(d, i) for i in range(1_000)])
        await db.insert_batch(ids, vecs, None, None, "insert")

        ticks: list[float] = []

        async def heartbeat():
            for _ in range(20):
                ticks.append(time.perf_counter())
                await asyncio.sleep(0.005)

        async def workload():
            for i in range(20):
                await db.search(_vec(d, i), top_k=5)

        await asyncio.gather(heartbeat(), workload())

        # If the loop blocked on workload(), heartbeat would skip ticks.
        # Verify monotonic spacing within reasonable bounds.
        spacings = [b - a for a, b in zip(ticks, ticks[1:])]
        assert max(spacings) < 0.5, (
            f"event loop was blocked: max heartbeat gap {max(spacings):.3f}s"
        )


# ── lifecycle ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_external_executor_is_not_shutdown(tmp_db_path):
    """When the caller supplies an executor, AsyncDatabase must not close it
    on `close()`. The caller manages its own lifecycle."""
    from concurrent.futures import ThreadPoolExecutor

    pool = ThreadPoolExecutor(max_workers=2)
    db = await AsyncDatabase.open(tmp_db_path, dimension=8, bits=2, executor=pool)
    await db.insert("x", _vec(8, 1))
    await db.close()
    # External pool must still be usable after db.close().
    fut = pool.submit(lambda: 42)
    assert fut.result(timeout=2) == 42
    pool.shutdown()


@pytest.mark.asyncio
async def test_sync_escape_hatch(tmp_db_path):
    async with await AsyncDatabase.open(tmp_db_path, dimension=8, bits=2) as db:
        await db.insert("x", _vec(8, 1))
        # Cheap sync ops can use the underlying Database directly.
        assert "x" in db.sync
        assert len(db.sync) == 1
