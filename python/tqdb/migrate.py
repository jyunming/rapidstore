"""One-command migration from existing Chroma / LanceDB collections to TQDB.

Reads each source's native on-disk format via the source library's own reader
(no schema-parsing tricks) and bulk-inserts into a fresh TQDB at the target
path. IDs, vectors, metadata, and document text are preserved.

Both source libraries are *optional* — they are imported lazily so users
without one installed get a clean ``ImportError`` only when they try to use
the corresponding migrator. Pass ``tqdb[migrate]`` to install both.

Programmatic API::

    from tqdb.migrate import migrate_chroma, migrate_lancedb

    n = migrate_chroma("/path/to/chroma_db", "/path/to/tqdb_dst",
                       collection="my_docs")
    n = migrate_lancedb("/path/to/lancedb", "/path/to/tqdb_dst",
                       table_name="my_docs")

CLI::

    python -m tqdb.migrate chroma /old /new                  # all collections
    python -m tqdb.migrate chroma /old /new --collection foo
    python -m tqdb.migrate lancedb /old /new --table my_docs
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from .tqdb import Database


# ── Chroma ──────────────────────────────────────────────────────────────


def _import_chromadb():
    try:
        import chromadb
        return chromadb
    except ImportError as e:
        raise ImportError(
            "Chroma migration requires the `chromadb` package. "
            "Install with: pip install 'tqdb[migrate]' "
            "or pip install chromadb"
        ) from e


def migrate_chroma(
    src_path: str,
    dst_path: str,
    *,
    collection: Optional[str] = None,
    bits: int = 4,
    batch_size: int = 1000,
    progress: bool = True,
) -> int:
    """Migrate a persistent Chroma database into a fresh TQDB.

    Args:
        src_path: Directory containing the Chroma SQLite + sidecar persistence.
        dst_path: Target directory for the new TQDB. Created if absent.
        collection: Name of the collection to migrate. ``None`` migrates every
            collection into the same TQDB instance — duplicate IDs across
            collections will fail; pass a single name when in doubt.
        bits: Quantization bits for the new TQDB (default 4 — good recall).
        batch_size: Number of rows per ``insert_batch`` call (default 1000).
        progress: Print a one-line summary per collection. Disable in scripts.

    Returns:
        Total number of rows migrated across all collections.
    """
    chromadb = _import_chromadb()
    src = Path(src_path)
    if not src.exists():
        raise FileNotFoundError(f"Chroma source path not found: {src}")

    client = chromadb.PersistentClient(path=str(src))
    if collection is not None:
        collection_names = [collection]
    else:
        collection_names = [c if isinstance(c, str) else c.name for c in client.list_collections()]
    if not collection_names:
        raise ValueError(f"No collections found at {src}")

    total_migrated = 0
    db: Optional[Database] = None
    try:
        for col_name in collection_names:
            col = client.get_collection(col_name)
            data = col.get(include=["embeddings", "metadatas", "documents"])
            ids = data.get("ids") or []
            if not ids:
                if progress:
                    print(f"  [{col_name}] empty — skipping")
                continue
            embeddings = data.get("embeddings")
            metadatas = data.get("metadatas") or [None] * len(ids)
            documents = data.get("documents") or [None] * len(ids)

            if embeddings is None:
                raise RuntimeError(
                    f"collection {col_name!r} has no embeddings — "
                    "cannot migrate without source vectors"
                )
            vecs = np.asarray(embeddings, dtype=np.float32)
            if vecs.ndim != 2:
                raise RuntimeError(
                    f"unexpected embeddings shape for {col_name!r}: {vecs.shape}"
                )
            d = vecs.shape[1]

            if db is None:
                db = Database.open(dst_path, dimension=d, bits=bits, metric="cosine")
            elif db.stats()["dimension"] != d:
                raise RuntimeError(
                    f"collection {col_name!r} has dimension {d}, but target "
                    f"TQDB was opened with dimension {db.stats()['dimension']}"
                )

            t0 = time.perf_counter()
            for start in range(0, len(ids), batch_size):
                end = min(start + batch_size, len(ids))
                db.insert_batch(
                    ids[start:end],
                    vecs[start:end],
                    metadatas[start:end],
                    documents[start:end],
                    "insert",
                )
            elapsed = time.perf_counter() - t0
            total_migrated += len(ids)
            if progress:
                print(
                    f"  [{col_name}] {len(ids):>7,} rows  d={d}  "
                    f"({elapsed:.2f}s, {len(ids)/max(elapsed,1e-6):,.0f} rows/s)"
                )
    finally:
        if db is not None:
            db.close()

    return total_migrated


# ── LanceDB ─────────────────────────────────────────────────────────────


def _import_lancedb():
    try:
        import lancedb
        return lancedb
    except ImportError as e:
        raise ImportError(
            "LanceDB migration requires the `lancedb` package. "
            "Install with: pip install 'tqdb[migrate]' "
            "or pip install lancedb"
        ) from e


def _detect_lance_columns(arrow_table) -> dict:
    """Heuristically map LanceDB schema columns to TQDB fields.

    LanceDB schemas vary by user — the only firm convention is a fixed-size
    list (or list) column for the vector. We pick the first one. ID/text
    columns are detected by name; metadata is "everything else".
    """
    import pyarrow as pa  # imported with lancedb

    cols = arrow_table.column_names
    schema = arrow_table.schema

    # Vector column: prefer name "vector"; else first FixedSizeList<float>.
    vector_col = None
    if "vector" in cols:
        vector_col = "vector"
    else:
        for name in cols:
            t = schema.field(name).type
            if pa.types.is_fixed_size_list(t) or (
                pa.types.is_list(t) and pa.types.is_floating(t.value_type)
            ):
                vector_col = name
                break
    if vector_col is None:
        raise RuntimeError(
            f"could not detect a vector column in {cols!r}; "
            "rename the source column to 'vector' before migrating"
        )

    # ID column: prefer "id"; else first string column that's not the vector.
    id_col = "id" if "id" in cols else None
    if id_col is None:
        for name in cols:
            if name == vector_col:
                continue
            if pa.types.is_string(schema.field(name).type):
                id_col = name
                break
    # ID may legitimately be absent — we'll synthesize one.

    # Document column: "text" or "document".
    doc_col = next((c for c in ("text", "document", "content") if c in cols), None)

    metadata_cols = [
        c for c in cols if c not in {vector_col, id_col, doc_col}
    ]
    return {
        "vector": vector_col,
        "id": id_col,
        "document": doc_col,
        "metadata": metadata_cols,
    }


def migrate_lancedb(
    src_path: str,
    dst_path: str,
    *,
    table_name: str,
    bits: int = 4,
    batch_size: int = 1000,
    progress: bool = True,
) -> int:
    """Migrate a LanceDB table into a fresh TQDB.

    Args:
        src_path: Path to the LanceDB dataset directory.
        dst_path: Target TQDB directory. Created if absent.
        table_name: Which table within the LanceDB dataset to migrate.
        bits: Quantization bits (default 4).
        batch_size: Rows per ``insert_batch`` (default 1000).
        progress: Print a single summary line. Disable in scripts.

    Returns:
        Number of rows migrated.
    """
    lancedb = _import_lancedb()
    src = Path(src_path)
    if not src.exists():
        raise FileNotFoundError(f"LanceDB source path not found: {src}")

    conn = lancedb.connect(str(src))
    if table_name not in conn.list_tables().tables:
        raise ValueError(
            f"table {table_name!r} not found in {src}; available: {conn.list_tables().tables}"
        )
    table = conn.open_table(table_name)
    arrow_table = table.to_arrow()
    n = arrow_table.num_rows
    if n == 0:
        if progress:
            print(f"  [{table_name}] empty — nothing to migrate")
        return 0

    schema_map = _detect_lance_columns(arrow_table)
    vector_col = schema_map["vector"]
    id_col = schema_map["id"]
    doc_col = schema_map["document"]
    meta_cols = schema_map["metadata"]

    # Materialize columns once; LanceDB's Arrow output handles the heavy lift.
    vec_array = arrow_table.column(vector_col).to_pylist()
    vecs = np.asarray(vec_array, dtype=np.float32)
    if vecs.ndim != 2:
        raise RuntimeError(
            f"vector column {vector_col!r} has unexpected shape {vecs.shape}"
        )
    d = vecs.shape[1]

    if id_col is not None:
        ids = [str(x) for x in arrow_table.column(id_col).to_pylist()]
    else:
        ids = [f"row_{i}" for i in range(n)]

    documents = (
        [None if x is None else str(x) for x in arrow_table.column(doc_col).to_pylist()]
        if doc_col
        else [None] * n
    )
    if meta_cols:
        meta_lists = {c: arrow_table.column(c).to_pylist() for c in meta_cols}
        metadatas: list[Any] = [
            {c: meta_lists[c][i] for c in meta_cols if meta_lists[c][i] is not None}
            for i in range(n)
        ]
    else:
        metadatas = [None] * n

    db = Database.open(dst_path, dimension=d, bits=bits, metric="cosine")
    try:
        t0 = time.perf_counter()
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            db.insert_batch(
                ids[start:end],
                vecs[start:end],
                metadatas[start:end],
                documents[start:end],
                "insert",
            )
        elapsed = time.perf_counter() - t0
    finally:
        db.close()

    if progress:
        print(
            f"  [{table_name}] {n:>7,} rows  d={d}  "
            f"({elapsed:.2f}s, {n/max(elapsed,1e-6):,.0f} rows/s)"
        )
    return n


# ── CLI ─────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tqdb.migrate",
        description="Migrate Chroma or LanceDB collections into TurboQuantDB.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_chroma = sub.add_parser("chroma", help="Migrate from Chroma persistent client")
    p_chroma.add_argument("src", help="Chroma source directory")
    p_chroma.add_argument("dst", help="Target TQDB directory")
    p_chroma.add_argument("--collection", default=None,
                           help="Migrate only this collection (default: all)")
    p_chroma.add_argument("--bits", type=int, default=4,
                           help="Target quantization bits (default 4)")
    p_chroma.add_argument("--batch-size", type=int, default=1000)

    p_lance = sub.add_parser("lancedb", help="Migrate from LanceDB")
    p_lance.add_argument("src", help="LanceDB source directory")
    p_lance.add_argument("dst", help="Target TQDB directory")
    p_lance.add_argument("--table", required=True,
                          help="Table name within the LanceDB dataset")
    p_lance.add_argument("--bits", type=int, default=4,
                          help="Target quantization bits (default 4)")
    p_lance.add_argument("--batch-size", type=int, default=1000)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    t0 = time.perf_counter()
    print(f"-> {args.cmd}: {args.src} -> {args.dst}")

    if args.cmd == "chroma":
        n = migrate_chroma(
            args.src,
            args.dst,
            collection=args.collection,
            bits=args.bits,
            batch_size=args.batch_size,
        )
    elif args.cmd == "lancedb":
        n = migrate_lancedb(
            args.src,
            args.dst,
            table_name=args.table,
            bits=args.bits,
            batch_size=args.batch_size,
        )
    else:
        # argparse 'required=True' guarantees we never reach this.
        return 2

    elapsed = time.perf_counter() - t0
    print(f"OK: migrated {n:,} rows in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
