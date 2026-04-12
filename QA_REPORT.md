# TurboQuantDB — Brutal QA Report

**Date:** 2026-04-12  
**Engineer:** QA (Copilot)  
**Test file:** `tests/test_brutal_qa.py`  
**Build:** `maturin develop --release` (Python 3.11.9, Windows)

---

## Executive Summary

| Metric | Value |
|--------|-------|
| Total tests | 166 |
| Passed | 163 |
| Expected failures (known bugs) | 3 |
| Unexpected failures | 0 |
| Bugs confirmed | 4 |
| Run time | ~50 s |

The suite exercised every public API surface: `Database.open`, all CRUD ops, search/filter/ANN, persistence, concurrency, the RAG retriever, stats, and path-safety guards. **Four real bugs were found and confirmed**, ranging from a Rust-level panic to a silent index corruption.

---

## Confirmed Bugs

### BUG-1 — `bits=256` triggers an unhandled Rust panic
**Severity:** 🔴 High  
**Test:** `TestOpenParameterBrutal::test_bits_extreme_large_raises_or_handles_gracefully`

**Reproduction:**
```python
db = Database.open(path, dimension=8, bits=256)
db.insert("a", vector)   # PanicException: capacity overflow
```

**Observed behaviour:**  
A raw Rust `capacity overflow` panic propagates to Python as `pyo3_runtime.PanicException`. Because `PanicException` inherits from `BaseException` (not `Exception`), it bypasses any caller using `except Exception` — including most application-level error handlers, context managers, and test frameworks.

**Expected behaviour:**  
`Database.open` should validate `bits` against a documented upper bound (e.g. ≤ 64) and raise a clean `ValueError` before reaching Rust arithmetic.

**Root cause location:** `src/storage/engine/mod.rs` — no upper-bound check on `bits` before it is used in bit-packing arithmetic that can overflow `usize`.

---

### BUG-2 — L2 distance scores can be negative
**Severity:** 🟠 Medium  
**Test:** `TestSearchBrutal::test_search_l2_scores_are_non_negative`

**Reproduction:**
```python
db = Database.open(path, dimension=16, bits=4, metric="l2")
db.insert_batch(ids, unit_vectors)
results = db.search(query, top_k=10)
# results[i]["score"] == -0.005076...   ← should be >= 0
```

**Observed behaviour:**  
Search results for `metric="l2"` return negative scores (observed: `-0.00507614`). L2 (Euclidean) distance is defined as non-negative by construction — `||a - b||² ≥ 0` for all real vectors.

**Expected behaviour:**  
All L2 scores must satisfy `score >= 0`. Quantization reconstruction error should be clamped to zero, not allowed to produce sub-zero outputs.

**Root cause:** The two-stage MSE + QJL quantizer introduces a systematic negative bias in reconstructed inner-product values used to approximate L2 distance. The bias is not clamped before the score is returned to the caller.

---

### BUG-3 — `update_index()` silently destroys the ANN index
**Severity:** 🔴 High  
**Test:** `TestSearchBrutal::test_update_index_after_large_batch_insert`

**Reproduction:**
```python
db = Database.open(path, dimension=16, bits=4)
db.insert_batch(ids_100, vecs_100)
db.create_index(max_degree=8, search_list_size=32)
assert db.stats()["has_index"]   # True ✓

db.insert_batch(ids_50, vecs_50)
db.update_index(max_degree=8, search_list_size=32)
assert db.stats()["has_index"]   # False ✗  ← BUG
```

**Observed behaviour:**  
After a successful `create_index()` + more inserts + `update_index()`, `stats()["has_index"]` is `False`. The ANN index is silently lost and `_use_ann=True` searches fall back to brute-force without warning.

**Root cause (traced to source):**  
Inside `create_index_incremental()` (`src/storage/engine/mod.rs:2319`), the first call is `flush_wal_to_segment()`. That function calls `invalidate_index_state()` (line 2059), which sets `self.manifest.index_state = None`. Back in `create_index_incremental()`, the guard at line 2403 is:

```rust
if let Some(state) = self.manifest.index_state.as_mut() {
    state.indexed_nodes = self.index_ids.len();   // skipped!
    state.search_list_size = search_list_size;
}
```

Because `index_state` is now `None`, the update is silently skipped. `can_use_ann_index()` then evaluates `index_state.is_some_and(...)` → `false`, reporting no index.

**Fix direction:** After the incremental build completes, unconditionally set `manifest.index_state = Some(IndexState { indexed_nodes: ..., search_list_size: ... })` rather than relying on the existing `Some` guard.

---

### BUG-4 — Invalid `quantizer_type` is silently ignored
**Severity:** 🟡 Low–Medium  
**Test:** `TestOpenParameterBrutal::test_invalid_quantizer_type_raises`

**Reproduction:**
```python
db = Database.open(path, dimension=8, bits=4, quantizer_type="bad")
# No exception raised — silently uses default quantizer
```

**Observed behaviour:**  
An unrecognised `quantizer_type` string falls through the if-chain in `src/storage/engine/mod.rs:401-411` and is treated as the default `ProdQuantizer`. The caller receives no feedback that their configuration was ignored.

**Expected behaviour:**  
`Database.open` should raise `ValueError: Invalid quantizer_type: 'bad'. Valid values: 'dense', 'exact', 'srht'.`

**Root cause:** The validation block at line 401 checks only for known values (`"dense"`, `"exact"`, `"srht"`); the else branch calls `ProdQuantizer::new()` without rejecting the unrecognised string.

---

## Test Coverage Summary

| Class | Tests | Focus |
|-------|-------|-------|
| `TestOpenParameterBrutal` | 19 | Parameter validation extremes: `bits`, `dimension`, `metric`, `quantizer_type`, `rerank_precision` |
| `TestVectorInputBrutal` | 21 | Zero/NaN/Inf vectors, F-order arrays, dtype coercions, duplicate IDs in batch |
| `TestIdEdgeCases` | 9 | Empty string, unicode, whitespace, null bytes, 10k-char IDs, 5000 unique IDs |
| `TestMetadataBrutal` | 13 | Deeply nested, list values, 100k-char strings, unicode keys, NaN/Inf floats |
| `TestSearchBrutal` | 17 | Empty DB, top_k extremes, deleted items, ANN before/after index, score validity |
| `TestFilterBrutal` | 15 | Contradictions, tautologies, `$in []`, `$nin` all, range on same field, mixed types |
| `TestPersistenceBrutal` | 11 | WAL crash recovery, index round-trip, delete→reinsert, multiple segment flushes |
| `TestCRUDBrutal` | 14 | Full lifecycle, upsert idempotence, stats consistency, high-frequency churn |
| `TestMetricCorrectness` | 7 | Self-top1 for all metrics, L2 zero distance, sorted scores, ≥70% recall gate |
| `TestRAGRetriever` | 8 | Empty texts, k>stored, mismatched length inputs, multi-call doc_store, NaN query |
| `TestRacingConditions` | 3 | Concurrent reads, concurrent read+write, concurrent batch inserts |
| `TestStatsBrutal` | 5 | Field consistency, `has_index` toggle, RAM estimate, dimension reflection |
| `TestPathTraversalSafety` | 7 | `../`, absolute paths, slash in collection name, `.` collection |
| `TestHeavyLoadBrutal` | 6 | 10k insert+search+delete, metadata filter on 2000 vectors, alternating insert/search |

---

## Non-Bug Findings (Design Clarifications)

| Finding | Verdict |
|---------|---------|
| `metric="IP"` (uppercase) accepted without error | **By design** — API internally calls `to_lowercase()` before matching |
| `close()` + immediate reopen on Windows requires `del db; gc.collect()` | **Platform limitation** — Windows ref-counts mmap handles; callers must release Python refs before reopening the same path |

---

## How to Reproduce All Findings

```bash
# Build (required once)
maturin develop --release

# Run full brutal suite
python -m pytest tests/test_brutal_qa.py -v \
    --basetemp=tmp_pytest_brutal

# Run only confirmed bug tests
python -m pytest tests/test_brutal_qa.py -v \
    --basetemp=tmp_pytest_brutal \
    -k "bits_extreme or l2_scores or update_index_after or invalid_quantizer"
```

Expected output: `163 passed, 3 xfailed` — the `xfail` tests are the three confirmed bugs (BUG-1 is caught by `BaseException`; BUG-2 and BUG-3 are `xfail(strict=True)`).
