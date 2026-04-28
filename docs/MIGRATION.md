# Migrating to TurboQuantDB

One-command migration from existing **Chroma** or **LanceDB** databases.

## Install

The migrators read each source library's native on-disk format via the source
library itself, so you'll need it installed:

```bash
pip install 'tqdb[migrate]'    # installs both chromadb and lancedb
# or just one:
pip install 'tqdb' chromadb
pip install 'tqdb' lancedb
```

## CLI

```bash
# Migrate every collection in a Chroma persistent client into a fresh TQDB
python -m tqdb.migrate chroma /path/to/chroma_db /path/to/tqdb_dst

# Migrate a single Chroma collection
python -m tqdb.migrate chroma /old /new --collection my_docs

# Migrate one LanceDB table
python -m tqdb.migrate lancedb /old /new --table my_docs
```

Both commands print one summary line per collection/table:

```
-> chroma: ./old -> ./new
  [my_docs]   12,345 rows  d=1536  (2.41s, 5,123 rows/s)
OK: migrated 12345 rows in 2.4s
```

## Programmatic API

```python
from tqdb.migrate import migrate_chroma, migrate_lancedb

# Chroma
n = migrate_chroma(
    "/path/to/chroma_db",
    "/path/to/tqdb_dst",
    collection="my_docs",     # None = all collections
    bits=4,                    # TQDB quantization bits (default 4)
    batch_size=1000,
)

# LanceDB
n = migrate_lancedb(
    "/path/to/lancedb",
    "/path/to/tqdb_dst",
    table_name="my_docs",
    bits=4,
)
```

Both return the number of rows migrated.

## What gets preserved

| Field | Chroma | LanceDB |
|-------|--------|---------|
| ID | yes | yes (column `id`; if missing, auto-numbered as `row_N`) |
| Vector | yes (float32) | yes (auto-detected: column `vector` or first fixed-size-list column) |
| Metadata | yes | yes (every column other than id/vector/text) |
| Document text | yes | yes (column `text` / `document` / `content`, in that order) |

## What's *not* preserved

- The original quantization (Chroma uses float32; LanceDB varies). TQDB
  re-quantizes on insert at the requested `bits` setting.
- Any external indexes attached to the source DB (HNSW, IVF, etc.). Run
  `db.create_index()` after migration to rebuild on the TQDB side.

## Dimension and metric

Both migrators detect the dimension from the source's vector column and pass it
to `Database.open(dimension=…)`. The metric defaults to `cosine` — the most
common choice for text embeddings. Change with `Database.open(metric="ip")`
on the destination after migration if needed.

## Multi-collection Chroma sources

Calling `migrate_chroma` without `--collection` migrates every collection
into the **same** target TQDB. IDs are expected to be unique across
collections (otherwise `insert_batch` fails on the duplicate). When in
doubt, migrate each collection into its own target directory.

## Running large migrations

The default `batch_size=1000` is conservative. For corpora over a few
million rows, raise it (e.g. `batch_size=10000`) — `insert_batch` is
batched-WAL-friendly and benefits from larger chunks.
