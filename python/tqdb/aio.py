"""asyncio-friendly wrapper around :class:`tqdb.Database`.

Every long-running method on the wrapped database dispatches to a thread-pool
executor. Because the Rust extension releases the GIL inside (`py.allow_threads`
is wired through every PyO3 method that touches the engine), this gives real
parallelism — many ``await db.search(...)`` calls genuinely run side by side
without blocking the event loop.

Quick start::

    import asyncio
    from tqdb.aio import AsyncDatabase

    async def main():
        db = await AsyncDatabase.open("./mydb", dimension=1536)
        await db.insert("doc1", vec)
        results = await db.search(query, top_k=5)
        await db.close()

    asyncio.run(main())

The wrapper owns the executor only when the caller didn't supply one. Pass
``executor=`` to share a pool across multiple databases or to control the
thread-pool size yourself.
"""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Optional

import numpy as np

from .tqdb import Database


def _default_max_workers() -> int:
    """Match Python 3.13's default ThreadPoolExecutor sizing.

    `min(32, cpu_count + 4)` is the cpython default; we replicate it explicitly
    so behaviour is identical across Python versions in case the default changes
    again upstream.
    """
    return min(32, (os.cpu_count() or 1) + 4)


class AsyncDatabase:
    """Async-friendly facade over the synchronous :class:`tqdb.Database`.

    Construction::

        await AsyncDatabase.open(path, dimension=…, …)

    The constructor uses ``run_in_executor`` so the database file I/O on first
    open doesn't block the event loop either.
    """

    __slots__ = ("_db", "_executor", "_owns_executor")

    def __init__(
        self,
        db: Database,
        executor: Optional[ThreadPoolExecutor] = None,
        owns_executor: bool = False,
    ) -> None:
        self._db = db
        self._executor = executor
        self._owns_executor = owns_executor

    # ── construction / teardown ──────────────────────────────────────────

    @classmethod
    async def open(
        cls,
        path: str,
        *,
        executor: Optional[ThreadPoolExecutor] = None,
        max_workers: Optional[int] = None,
        **kwargs: Any,
    ) -> "AsyncDatabase":
        """Open or create a database asynchronously.

        Args:
            path: Database directory (same semantics as ``Database.open``).
            executor: Optional shared executor. If omitted, a new
                ``ThreadPoolExecutor`` is created and owned by the returned
                instance — closed by ``close()``.
            max_workers: Worker count for the auto-created executor. Ignored
                when ``executor`` is supplied. Defaults to
                ``min(32, cpu_count + 4)``.
            **kwargs: Forwarded to ``Database.open`` (``dimension``, ``bits``, …).
        """
        owns = executor is None
        if owns:
            executor = ThreadPoolExecutor(
                max_workers=max_workers or _default_max_workers(),
                thread_name_prefix="tqdb-async",
            )
        loop = asyncio.get_running_loop()
        sync_db = await loop.run_in_executor(
            executor, partial(Database.open, path, **kwargs)
        )
        return cls(sync_db, executor=executor, owns_executor=owns)

    async def close(self) -> None:
        """Flush state and shut the underlying database. Closes the executor
        if this instance created it."""
        await self._run(self._db.close)
        if self._owns_executor and self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None

    async def __aenter__(self) -> "AsyncDatabase":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    # ── internal dispatch ─────────────────────────────────────────────────

    async def _run(self, func, *args, **kwargs):
        loop = asyncio.get_running_loop()
        if kwargs:
            func = partial(func, *args, **kwargs)
            return await loop.run_in_executor(self._executor, func)
        return await loop.run_in_executor(self._executor, func, *args)

    # ── write API ────────────────────────────────────────────────────────

    async def insert(
        self,
        id: str,
        vector,
        metadata: Optional[dict] = None,
        document: Optional[str] = None,
    ) -> None:
        await self._run(self._db.insert, id, vector, metadata, document)

    async def insert_batch(
        self,
        ids: list[str],
        vectors,
        metadatas: Optional[list] = None,
        documents: Optional[list] = None,
        mode: str = "insert",
    ) -> None:
        await self._run(
            self._db.insert_batch, ids, vectors, metadatas, documents, mode
        )

    async def upsert(
        self,
        id: str,
        vector,
        metadata: Optional[dict] = None,
        document: Optional[str] = None,
    ) -> None:
        await self._run(self._db.upsert, id, vector, metadata, document)

    async def update(
        self,
        id: str,
        vector,
        metadata: Optional[dict] = None,
        document: Optional[str] = None,
    ) -> None:
        await self._run(self._db.update, id, vector, metadata, document)

    async def update_metadata(
        self,
        id: str,
        metadata: Optional[dict] = None,
        document: Optional[str] = None,
    ) -> None:
        await self._run(self._db.update_metadata, id, metadata, document)

    async def delete(self, id: str) -> bool:
        return await self._run(self._db.delete, id)

    async def delete_batch(
        self,
        ids: Optional[list[str]] = None,
        where_filter: Optional[dict] = None,
    ) -> int:
        return await self._run(
            self._db.delete_batch, ids if ids is not None else [], where_filter
        )

    # ── read API ─────────────────────────────────────────────────────────

    async def get(self, id: str):
        return await self._run(self._db.get, id)

    async def get_many(self, ids: list[str]):
        return await self._run(self._db.get_many, ids)

    async def list_all(self) -> list[str]:
        return await self._run(self._db.list_all)

    async def list_ids(
        self,
        where_filter: Optional[dict] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> list[str]:
        return await self._run(self._db.list_ids, where_filter, limit, offset)

    async def count(self, filter: Optional[dict] = None) -> int:
        return await self._run(self._db.count, filter)

    async def search(
        self,
        query,
        top_k: int,
        **kwargs: Any,
    ) -> list[dict]:
        return await self._run(self._db.search, query, top_k, **kwargs)

    async def query(
        self,
        query_embeddings,
        n_results: int = 10,
        **kwargs: Any,
    ) -> list[list[dict]]:
        return await self._run(
            self._db.query, query_embeddings, n_results, **kwargs
        )

    # ── index / maintenance ──────────────────────────────────────────────

    async def create_index(self, **kwargs: Any) -> None:
        await self._run(self._db.create_index, **kwargs)

    async def create_coarse_index(self, n_clusters: int = 256) -> None:
        await self._run(self._db.create_coarse_index, n_clusters)

    async def checkpoint(self) -> None:
        await self._run(self._db.checkpoint)

    async def stats(self) -> dict:
        return await self._run(self._db.stats)

    # ── sync passthroughs (these are O(1), no executor needed) ──────────

    def __len__(self) -> int:
        return len(self._db)

    def __contains__(self, id: str) -> bool:
        return id in self._db

    @property
    def sync(self) -> Database:
        """Escape hatch: the underlying synchronous Database for any method
        that hasn't been wrapped here. Useful for read-only operations the
        caller is sure are fast (``len``, ``__contains__``, etc.)."""
        return self._db
