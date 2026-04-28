"""Type stubs for the tqdb package."""

from tqdb.tqdb import Database as Database, TurboQuantDB as TurboQuantDB
from tqdb.chroma_compat import (
    CompatClient as ChromaCompatClient,
    PersistentClient as PersistentClient,
)
from tqdb.lancedb_compat import connect as lancedb_connect
from tqdb.aio import AsyncDatabase as AsyncDatabase
from tqdb.multivector import MultiVectorStore as MultiVectorStore

__version__: str

__all__ = [
    "Database",
    "TurboQuantDB",
    "AsyncDatabase",
    "MultiVectorStore",
    "__version__",
    "ChromaCompatClient",
    "PersistentClient",
    "lancedb_connect",
]
