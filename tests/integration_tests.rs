use std::collections::HashMap;
use ndarray::Array1;
use serde_json::Value as JsonValue;
use tempfile::tempdir;

use turboquantdb::storage::engine::TurboQuantEngine;

/// Test full insert → flush → search roundtrip with metadata.
#[test]
fn test_insert_and_search() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().to_str().unwrap();

    let d = 64;
    let b = 4;
    let mut engine = TurboQuantEngine::open(db_path, d, b, 42).unwrap();

    // Insert 10 random vectors
    for i in 0..10usize {
        let vec = Array1::<f64>::from_iter((0..d).map(|j| (i * d + j) as f64 / 100.0));
        let mut meta = HashMap::new();
        meta.insert("index".to_string(), JsonValue::Number(i.into()));
        meta.insert("tag".to_string(), JsonValue::String(format!("doc_{}", i)));
        engine.insert(format!("vec_{}", i), &vec, meta).unwrap();
    }

    // Query with a vector close to vec_0 (all zeros except small values)
    let query = Array1::<f64>::from_iter((0..d).map(|j| j as f64 / 100.0));
    let results = engine.search(&query, 3).unwrap();

    assert_eq!(results.len(), 3, "Should return 3 results");
    // Scores should be descending
    assert!(results[0].score >= results[1].score, "Results should be sorted by score descending");
    assert!(results[0].score > 0.0, "Top score should be positive");
    // The top result metadata should have a valid 'index' key
    let meta = &results[0].metadata;
    assert!(meta.contains_key("index"), "Metadata should contain 'index' key");
}

/// Test WAL crash recovery: inserts survive without explicit flush.
#[test]
fn test_wal_recovery() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().to_str().unwrap();

    let d = 32;
    let b = 2;

    // Open, insert, drop without calling close() (simulates crash)
    {
        let mut engine = TurboQuantEngine::open(db_path, d, b, 42).unwrap();
        let vec = Array1::<f64>::from_iter((0..d).map(|i| i as f64));
        engine.insert("crash_vec".to_string(), &vec, HashMap::new()).unwrap();
        // Drop without flush — WAL persists the entry
    }

    // Reopen — WAL replay should recover the vector into the WAL buffer
    let mut engine = TurboQuantEngine::open(db_path, d, b, 42).unwrap();
    // WAL replay puts entries in wal_buffer; flush to count them in manifest
    engine.flush_wal_to_segment().unwrap();
    assert_eq!(engine.vector_count(), 1, "Vector should survive crash recovery via WAL");
}

/// Test persistence: save to disk, reload, verify vector count.
#[test]
fn test_persist_and_reload() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().to_str().unwrap();

    let d = 128;
    let b = 4;

    // Insert and explicitly flush
    {
        let mut engine = TurboQuantEngine::open(db_path, d, b, 99).unwrap();
        for i in 0..50usize {
            let vec = Array1::<f64>::from_iter((0..d).map(|_| i as f64 * 0.01));
            engine.insert(format!("v{}", i), &vec, HashMap::new()).unwrap();
        }
        engine.close().unwrap(); // Flushes WAL → segments + saves manifest
    }

    // Reload and verify
    let engine = TurboQuantEngine::open(db_path, d, b, 99).unwrap();
    assert_eq!(engine.vector_count(), 50, "All 50 vectors should survive persist/reload");
}

/// Test schema mismatch error on reload.
#[test]
fn test_schema_mismatch() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().to_str().unwrap();

    {
        let mut engine = TurboQuantEngine::open(db_path, 64, 4, 42).unwrap();
        let vec = Array1::<f64>::zeros(64);
        engine.insert("v0".to_string(), &vec, HashMap::new()).unwrap();
        engine.close().unwrap();
    }

    // Try to open with wrong dimension — should fail cleanly
    let result = TurboQuantEngine::open(db_path, 128, 4, 42);
    assert!(result.is_err(), "Should reject mismatched dimension");
}
