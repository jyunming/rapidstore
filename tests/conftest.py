"""
Pytest configuration for TurboQuantDB tests.

On Windows, memory-mapped file handles are held by the OS until the Python
objects referencing them are garbage-collected.  pytest's ``tmp_path`` teardown
runs *before* the GC sweep, which causes ``PermissionError: [WinError 5]``
when pytest tries to delete the test directory.

The ``gc_collect`` autouse fixture forces a collection pass after every test so
that Rust-side ``Drop`` impls release the mmap handles before cleanup.
"""

import gc
import sys
from pathlib import Path

import pytest


def pytest_configure(config):
    """Redirect tmp_path basetemp to repo-local dir on Windows.

    C:\\Users\\...\\AppData\\Local\\Temp\\pytest-of-<user> can become inaccessible
    (WinError 5 – Access Denied) across sessions.  Using a repo-local dir avoids
    the system-temp permission issue without requiring --basetemp on every run.
    """
    if sys.platform == "win32":
        # Only override basetemp when the user hasn't explicitly set it via --basetemp.
        if not config.option.basetemp:
            repo_root = Path(__file__).parent.parent
            basetemp = repo_root / "tmp_pytest"
            basetemp.mkdir(exist_ok=True)
            config.option.basetemp = str(basetemp)


@pytest.fixture(autouse=True)
def gc_collect():
    """Force garbage collection after each test to release mmap handles (Windows)."""
    yield
    gc.collect()
    if sys.platform == "win32":
        # A second pass resolves any reference cycles that the first pass
        # only marks but does not collect immediately.
        gc.collect()
