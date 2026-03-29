# Batch CRUD and Filter Grammar v1

This document captures the current RapidStore batch APIs and filter behavior for M1.

## Python API (PyO3)

Use `Database` for direct operations:

```python
import numpy as np
import turboquantdb as tq

db = tq.Database.open(
    uri="C:/tmp/tqdb",
    dimension=8,
    bits=2,
    seed=42,
    metric="cosine",
)

ids = ["a", "b", "c"]
vectors = [np.ones(8) * 0.1, np.ones(8) * 0.2, np.ones(8) * 0.3]
metadatas = [
    {"tenant": "alpha", "profile": {"region": "eu", "score": 10}},
    {"tenant": "alpha", "profile": {"region": "us", "score": 20}},
    {"tenant": "beta"},
]
documents = ["doc-a", "doc-b", None]

db.insert_many(ids, vectors, metadatas=metadatas, documents=documents)
rows = db.get_many(ids, include_document=True)
print(rows)
```

Available batch methods:
- `insert_many(ids, vectors, metadatas=None, documents=None)`
- `upsert_many(ids, vectors, metadatas=None, documents=None)`
- `update_many(ids, vectors, metadatas=None, documents=None)`
- `get_many(ids, include_document=True)`
- `delete_many(ids)`

## Rust API

```rust
use ndarray::Array1;
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use turboquantdb::storage::engine::{BatchWriteItem, TurboQuantEngine};

let mut engine = TurboQuantEngine::open("C:/tmp/tqdb", "C:/tmp/tqdb", 8, 2, 42)?;
let items = vec![
    BatchWriteItem {
        id: "a".to_string(),
        vector: Array1::<f64>::from_elem(8, 0.1),
        metadata: HashMap::from([("tenant".to_string(), JsonValue::String("alpha".to_string()))]),
        document: Some("doc-a".to_string()),
    },
];
engine.insert_many(items)?;
```

## Filter Grammar v1

Supported operators:
- Logical: `$and`, `$or`
- Field operators: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`
- Nested field paths: dot notation, e.g. `profile.region`

Strict type policy:
- No implicit numeric/string coercion in comparisons.
- Unknown operators evaluate as non-match.

Missing-field policy:
- `$ne` matches missing fields.
- `$eq`, range operators, and `$in` do not match missing fields.

## Benchmark Harness

A reproducible benchmark test is available:
- `tests/bench_batch_crud.rs` (ignored by default)

Run it with:

```powershell
$env:CARGO_TARGET_DIR="$env:USERPROFILE\.cargo-target\turboquantDB"
cargo test -q --test bench_batch_crud -- --ignored --nocapture
```





