"""
Persistence regression tests for delete -> reinsert flows.
"""

from __future__ import annotations

import numpy as np

from tqdb import Database


def _vec(d: int = 16) -> np.ndarray:
    return np.arange(d, dtype=np.float32)


def test_insert_persists_across_reopen_smoke(tmp_path):
    """
    Control case: a straightforward insert must survive close/reopen.
    """
    path = str(tmp_path / "db")
    db = Database.open(path, 16, bits=4, metric="ip")
    db.insert("x", _vec())
    assert db.count() == 1
    db.close()

    db2 = Database.open(path, 16)
    assert db2.count() == 1
    assert db2.get("x") is not None


def test_delete_then_reinsert_should_persist_across_reopen(tmp_path):
    """
    Minimal repro:
    1) insert/upsert ID
    2) delete ID
    3) reinsert same ID
    4) close + reopen

    Expected: ID is present.
    Current behavior: ID is missing after reopen.
    """
    path = str(tmp_path / "db")
    db = Database.open(path, 16, bits=4, metric="ip")
    v = _vec()

    db.upsert("x", v, metadata={"phase": 1}, document="first")
    assert db.count() == 1
    assert db.get("x") is not None

    deleted = db.delete("x")
    assert deleted is True
    assert db.count() == 0

    db.upsert("x", v, metadata={"phase": 2}, document="second")
    assert db.count() == 1
    assert db.get("x") is not None

    db.close()

    reopened = Database.open(path, 16)
    assert reopened.count() == 1
    got = reopened.get("x")
    assert got is not None
    assert got["metadata"]["phase"] == 2
    assert got["document"] == "second"
