"""Type stubs for the tqdb package."""

from tqdb.tqdb import Database as Database, TurboQuantDB as TurboQuantDB
from tqdb.chroma_compat import (
    CompatClient as ChromaCompatClient,
    PersistentClient as PersistentClient,
)
from tqdb.lancedb_compat import connect as lancedb_connect

__version__: str

__all__ = [
    "Database",
    "TurboQuantDB",
    "__version__",
    "ChromaCompatClient",
    "PersistentClient",
    "lancedb_connect",
]
