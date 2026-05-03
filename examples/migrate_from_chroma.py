"""Migrate an existing ChromaDB collection into TurboQuantDB.

Two equivalent paths:
  1. Programmatic API — call `migrate_chroma()` from Python (this script).
  2. CLI — `python -m tqdb.migrate chroma <src_path> <dst_path>` (one-liner).

Use the programmatic path when you need to inspect / verify the destination
DB right after migration, or to chain with other code. Use the CLI when
you just want to move a collection and walk away.

Run:
    pip install 'tqdb[migrate-chroma]'
    python examples/migrate_from_chroma.py /path/to/chroma_db /path/to/tqdb_dst
"""
from __future__ import annotations

import sys
from pathlib import Path

from tqdb import Database
from tqdb.migrate import migrate_chroma


def main() -> None:
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <src_chroma_path> <dst_tqdb_path>")
        sys.exit(1)

    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    if not src.exists():
        print(f"Error: source Chroma DB {src} does not exist.")
        sys.exit(1)

    print(f"Migrating Chroma -> TQDB: {src} -> {dst}")
    n = migrate_chroma(str(src), str(dst), bits=4)
    print(f"  migrated {n} vector(s).")

    db = Database.open(str(dst))
    stats = db.stats()
    print(f"  destination DB stats: {stats}")

    sample_ids = db.list_ids(limit=3)
    if sample_ids:
        print(f"\nSample documents:")
        for record in db.get_many(sample_ids):
            if record is None:
                continue
            doc = (record.get("document") or "")[:80]
            print(f"  {record['id']}: {doc} ...")


if __name__ == "__main__":
    main()
