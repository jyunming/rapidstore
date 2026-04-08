"""
Import-fallback tests for tqdb Python modules.

These tests exercise branches that only run when relative extension imports
fail, without changing production code.
"""

from __future__ import annotations

import builtins
import importlib
import importlib.util
import sys
from pathlib import Path

import importlib.metadata as ilmd
import pytest


ROOT = Path(__file__).resolve().parents[1]
PKG_DIR = ROOT / "python" / "tqdb"


def _exec_module_with_forced_relative_tqdb_import_failure(module_name: str, file_path: Path):
    """
    Execute a module while forcing `from .tqdb import Database` to raise ImportError.

    This is enough to drive fallback branches without uninstalling the extension.
    """
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)

    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        # `from .tqdb import Database` resolves to: name="tqdb", level=1.
        if level == 1 and name == "tqdb" and tuple(fromlist) == ("Database",):
            raise ImportError("forced relative tqdb import failure")
        return original_import(name, globals, locals, fromlist, level)

    builtins.__import__ = guarded_import
    sys.modules[module_name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        builtins.__import__ = original_import
        sys.modules.pop(module_name, None)
    return mod


def test_init_version_falls_back_to_0_when_package_metadata_missing(monkeypatch):
    """
    Covers python/tqdb/__init__.py fallback:
    except PackageNotFoundError: __version__ = "0.0.0"
    """
    import tqdb

    original_version_fn = ilmd.version

    def raise_not_found(_dist_name: str):
        raise ilmd.PackageNotFoundError("forced missing metadata")

    try:
        monkeypatch.setattr(ilmd, "version", raise_not_found)
        reloaded = importlib.reload(tqdb)
        assert reloaded.__version__ == "0.0.0"
    finally:
        # Avoid leaking module state to later tests.
        monkeypatch.setattr(ilmd, "version", original_version_fn)
        importlib.reload(tqdb)


def test_rag_stub_database_open_raises_runtime_error_when_relative_import_fails():
    """
    Covers python/tqdb/rag.py fallback stub class (lines in ImportError branch).
    """
    mod = _exec_module_with_forced_relative_tqdb_import_failure(
        "tqdb._rag_fallback_test",
        PKG_DIR / "rag.py",
    )
    with pytest.raises(RuntimeError, match="not available"):
        mod.Database.open("dummy_path", 8)


def test_chroma_compat_falls_back_to_absolute_tqdb_import_when_relative_import_fails():
    """
    Covers python/tqdb/chroma_compat.py ImportError fallback branch.
    """
    mod = _exec_module_with_forced_relative_tqdb_import_failure(
        "tqdb._chroma_fallback_test",
        PKG_DIR / "chroma_compat.py",
    )
    assert hasattr(mod, "Database")
    assert callable(mod.PersistentClient)


def test_lancedb_compat_falls_back_to_absolute_tqdb_import_when_relative_import_fails():
    """
    Covers python/tqdb/lancedb_compat.py ImportError fallback branch.
    """
    mod = _exec_module_with_forced_relative_tqdb_import_failure(
        "tqdb._lancedb_fallback_test",
        PKG_DIR / "lancedb_compat.py",
    )
    assert hasattr(mod, "Database")
    assert callable(mod.connect)
