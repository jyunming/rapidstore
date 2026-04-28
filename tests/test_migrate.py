"""End-to-end tests for the Chroma / LanceDB migration toolkit.

Each test creates a real source DB with the source library, runs the migrator,
opens the resulting TQDB, and verifies that IDs / metadata / documents
round-trip exactly. Vector scores are verified within a tolerance that allows
for the lossy 4-bit quantization on the TQDB side.
"""

from __future__ import annotations

import gc
import shutil
import sys
import tempfile
from contextlib import contextmanager

import numpy as np
import pytest

from tqdb import Database
from tqdb.migrate import main as cli_main, migrate_chroma, migrate_lancedb


def _make_unit_vector(d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(d).astype(np.float32)
    return v / max(np.linalg.norm(v), 1e-9)


@contextmanager
def _tempdir():
    """Like tempfile.TemporaryDirectory but tolerates Chroma's SQLite locking
    on Windows. Chroma holds the file handle open longer than its client's
    refcount suggests; an extra `gc.collect()` plus `ignore_errors=True` on
    rmtree avoids spurious PermissionError on test cleanup."""
    path = tempfile.mkdtemp()
    try:
        yield path
    finally:
        gc.collect()
        shutil.rmtree(path, ignore_errors=True)


# ── Chroma round-trip ───────────────────────────────────────────────────


def _make_chroma(path: str, name: str, d: int, n: int):
    import chromadb
    client = chromadb.PersistentClient(path=path)
    col = client.get_or_create_collection(name=name)
    ids = [f"d{i}" for i in range(n)]
    embs = [_make_unit_vector(d, i).tolist() for i in range(n)]
    metas = [{"bucket": i % 3, "name": f"item-{i}"} for i in range(n)]
    docs = [f"document text {i}" for i in range(n)]
    col.add(ids=ids, embeddings=embs, metadatas=metas, documents=docs)
    return ids, embs, metas, docs


def test_migrate_chroma_round_trip():
    d, n = 16, 25
    with _tempdir() as src, _tempdir() as dst:
        ids, embs, metas, docs = _make_chroma(src, "test_col", d, n)

        migrated = migrate_chroma(src, dst, collection="test_col",
                                   bits=4, progress=False)
        assert migrated == n

        db = Database.open(dst)
        try:
            assert len(db) == n
            for i, expected_id in enumerate(ids):
                rec = db.get(expected_id)
                assert rec is not None, f"missing id {expected_id}"
                assert rec["metadata"]["bucket"] == metas[i]["bucket"]
                assert rec["metadata"]["name"] == metas[i]["name"]
                assert rec["document"] == docs[i]
            # Vector fidelity: query with d0's embedding, expect d0 in top-3
            # despite cosine + 4-bit quantization.
            q = np.asarray(embs[0], dtype=np.float32)
            results = db.search(q, top_k=3)
            assert "d0" in {r["id"] for r in results}
        finally:
            db.close()


def test_migrate_chroma_all_collections():
    d, n = 8, 10
    with _tempdir() as src, _tempdir() as dst:
        # Create two collections with disjoint IDs (so they can co-exist in TQDB).
        import chromadb
        client = chromadb.PersistentClient(path=src)
        for col_name, prefix in (("colA", "a"), ("colB", "b")):
            col = client.get_or_create_collection(name=col_name)
            ids = [f"{prefix}{i}" for i in range(n)]
            embs = [_make_unit_vector(d, i).tolist() for i in range(n)]
            col.add(ids=ids, embeddings=embs)

        migrated = migrate_chroma(src, dst, collection=None,
                                   bits=4, progress=False)
        assert migrated == 2 * n

        db = Database.open(dst)
        try:
            assert len(db) == 2 * n
        finally:
            db.close()


def test_migrate_chroma_missing_path_raises():
    with pytest.raises(FileNotFoundError):
        migrate_chroma("/nonexistent/chroma", "/nonexistent/dst", progress=False)


def test_migrate_chroma_empty_path_raises():
    with _tempdir() as src, _tempdir() as dst:
        # Empty src dir — no collections — should raise rather than silently do nothing.
        with pytest.raises(ValueError, match="No collections"):
            migrate_chroma(src, dst, progress=False)


# ── LanceDB round-trip ──────────────────────────────────────────────────


def _make_lancedb(path: str, name: str, d: int, n: int):
    import lancedb
    import pyarrow as pa

    conn = lancedb.connect(path)
    schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), d)),
            pa.field("text", pa.string()),
            pa.field("year", pa.int64()),
        ]
    )
    rows = [
        {
            "id": f"d{i}",
            "vector": _make_unit_vector(d, i).tolist(),
            "text": f"row {i}",
            "year": 2020 + (i % 5),
        }
        for i in range(n)
    ]
    table = conn.create_table(name, data=rows, schema=schema)
    return rows, table


def test_migrate_lancedb_round_trip():
    d, n = 16, 25
    with _tempdir() as src, _tempdir() as dst:
        rows, _ = _make_lancedb(src, "items", d, n)

        migrated = migrate_lancedb(src, dst, table_name="items",
                                    bits=4, progress=False)
        assert migrated == n

        db = Database.open(dst)
        try:
            assert len(db) == n
            for r in rows:
                rec = db.get(r["id"])
                assert rec is not None
                assert rec["metadata"]["year"] == r["year"]
                assert rec["document"] == r["text"]
            # Vector fidelity check.
            q = np.asarray(rows[0]["vector"], dtype=np.float32)
            results = db.search(q, top_k=3)
            assert "d0" in {r["id"] for r in results}
        finally:
            db.close()


def test_migrate_lancedb_table_not_found():
    with _tempdir() as src, _tempdir() as dst:
        _make_lancedb(src, "items", d=8, n=3)
        with pytest.raises(ValueError, match="not found"):
            migrate_lancedb(src, dst, table_name="missing", progress=False)


# ── CLI smoke ───────────────────────────────────────────────────────────


def test_cli_chroma_smoke(capsys):
    d, n = 8, 5
    with _tempdir() as src, _tempdir() as dst:
        _make_chroma(src, "cli_col", d, n)
        rc = cli_main(["chroma", src, dst, "--collection", "cli_col"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "OK: migrated 5 rows" in out


def test_cli_lancedb_smoke(capsys):
    d, n = 8, 5
    with _tempdir() as src, _tempdir() as dst:
        _make_lancedb(src, "items", d, n)
        rc = cli_main(["lancedb", src, dst, "--table", "items"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "OK: migrated 5 rows" in out
