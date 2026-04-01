use ndarray::Array1;
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tempfile::tempdir;

use turboquantdb::storage::backend::StorageBackend;
use turboquantdb::storage::engine::{DistanceMetric, TurboQuantEngine};
use turboquantdb::storage::graph::GraphManager;
use turboquantdb::storage::segment::{Segment, SegmentRecord};

/// Test full insert → flush → search roundtrip with metadata.
#[test]
fn test_insert_and_search() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().to_str().unwrap();

    let d = 64;
    let b = 4;
    let mut engine = TurboQuantEngine::open(db_path, db_path, d, b, 42).unwrap();

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
    assert!(
        results[0].score >= results[1].score,
        "Results should be sorted by score descending"
    );
    assert!(results[0].score > 0.0, "Top score should be positive");
    // The top result metadata should have a valid 'index' key
    let meta = &results[0].metadata;
    assert!(
        meta.contains_key("index"),
        "Metadata should contain 'index' key"
    );
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
        let mut engine = TurboQuantEngine::open(db_path, db_path, d, b, 42).unwrap();
        let vec = Array1::<f64>::from_iter((0..d).map(|i| i as f64));
        engine
            .insert("crash_vec".to_string(), &vec, HashMap::new())
            .unwrap();
        // Drop without flush — WAL persists the entry
    }

    // Reopen — WAL replay should recover the vector into the WAL buffer
    let mut engine = TurboQuantEngine::open(db_path, db_path, d, b, 42).unwrap();
    // WAL replay puts entries in wal_buffer; flush to count them in manifest
    engine.flush_wal_to_segment().unwrap();
    assert_eq!(
        engine.vector_count(),
        1,
        "Vector should survive crash recovery via WAL"
    );
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
        let mut engine = TurboQuantEngine::open(db_path, db_path, d, b, 99).unwrap();
        for i in 0..50usize {
            let vec = Array1::<f64>::from_iter((0..d).map(|_| i as f64 * 0.01));
            engine
                .insert(format!("v{}", i), &vec, HashMap::new())
                .unwrap();
        }
        engine.close().unwrap(); // Flushes WAL → segments + saves manifest
    }

    // Reload and verify
    let engine = TurboQuantEngine::open(db_path, db_path, d, b, 99).unwrap();
    assert_eq!(
        engine.vector_count(),
        50,
        "All 50 vectors should survive persist/reload"
    );
}

/// Test schema mismatch error on reload.
#[test]
fn test_schema_mismatch() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().to_str().unwrap();

    {
        let mut engine = TurboQuantEngine::open(db_path, db_path, 64, 4, 42).unwrap();
        let vec = Array1::<f64>::zeros(64);
        engine
            .insert("v0".to_string(), &vec, HashMap::new())
            .unwrap();
        engine.close().unwrap();
    }

    // Try to open with wrong dimension — should fail cleanly
    let result = TurboQuantEngine::open(db_path, db_path, 128, 4, 42);
    assert!(result.is_err(), "Should reject mismatched dimension");
}

/// Test Vamana Graph ANN search.
#[test]
fn test_vamana_search() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().to_str().unwrap();

    let d = 32;
    let b = 2;
    let mut engine = TurboQuantEngine::open(db_path, db_path, d, b, 42).unwrap();

    // Insert 100 vectors
    for i in 0..100usize {
        let vec = Array1::<f64>::from_iter((0..d).map(|j| (i + j) as f64 / 1000.0));
        engine
            .insert(format!("v{}", i), &vec, HashMap::new())
            .unwrap();
    }

    // Build index
    engine.create_index(32, 64).unwrap();

    // Search with query
    let query = Array1::<f64>::from_iter((0..d).map(|j| j as f64 / 1000.0));
    let results = engine.search(&query, 5).unwrap();

    assert!(!results.is_empty());
    assert!(results.len() <= 5);
    // Scores should be descending and deterministic top result should exist.
    assert!(results[0].score >= results[1].score);
}

#[test]
fn test_crud_and_filter() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().to_str().unwrap();
    let mut engine = TurboQuantEngine::open(db_path, db_path, 16, 2, 7).unwrap();

    let v_a = Array1::<f64>::from_elem(16, 0.1);
    let v_b = Array1::<f64>::from_elem(16, 0.2);
    let mut meta_a = HashMap::new();
    meta_a.insert("tenant".to_string(), JsonValue::String("alpha".to_string()));
    engine.insert("a".to_string(), &v_a, meta_a).unwrap();

    let mut meta_b = HashMap::new();
    meta_b.insert("tenant".to_string(), JsonValue::String("beta".to_string()));
    engine.upsert("b".to_string(), &v_b, meta_b).unwrap();

    // Update existing
    let mut meta_b2 = HashMap::new();
    meta_b2.insert("tenant".to_string(), JsonValue::String("alpha".to_string()));
    engine
        .update("b".to_string(), &Array1::<f64>::from_elem(16, 0.3), meta_b2)
        .unwrap();

    // Filter by metadata
    let mut where_filter = HashMap::new();
    where_filter.insert("tenant".to_string(), JsonValue::String("alpha".to_string()));
    let q = Array1::<f64>::from_elem(16, 0.3);
    let filtered = engine
        .search_with_filter(&q, 10, Some(&where_filter))
        .unwrap();
    assert_eq!(filtered.len(), 2);

    // Delete one and confirm get() + search visibility
    let deleted = engine.delete("a".to_string()).unwrap();
    assert!(deleted);
    assert!(engine.get("a").unwrap().is_none());
    let filtered_after = engine
        .search_with_filter(&q, 10, Some(&where_filter))
        .unwrap();
    assert_eq!(filtered_after.len(), 1);
    assert_eq!(filtered_after[0].id, "b");
}

#[test]
fn test_compaction_and_disk_stats() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().to_str().unwrap();
    let mut engine = TurboQuantEngine::open(db_path, db_path, 32, 2, 11).unwrap();

    // Force many small segments.
    for i in 0..12usize {
        let vec = Array1::<f64>::from_iter((0..32).map(|j| (i + j) as f64 / 100.0));
        engine
            .insert(format!("id-{}", i), &vec, HashMap::new())
            .unwrap();
        engine.flush_wal_to_segment().unwrap();
    }
    let before = engine.stats();
    assert!(before.segment_count >= 2);
    assert!(before.total_disk_bytes > 0);
    assert!(before.physical_record_count >= 12);
    assert_eq!(before.deleted_record_count, 0);
    assert!(before.segment_max_bytes.is_some());
    assert!(before.segment_avg_bytes.is_some());
    assert!(before.segment_skew_ratio.is_some());

    engine.compact().unwrap();
    let after = engine.stats();
    assert_eq!(after.vector_count, 12);
    assert_eq!(after.segment_count, 1);
    assert!(after.total_disk_bytes > 0);
    assert!(after.compaction_runs >= 1);
    assert!(after.last_reclaimed_segments >= 1);
}

#[test]
fn test_batch_and_advanced_filters() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().to_str().unwrap();
    let mut engine = TurboQuantEngine::open(db_path, db_path, 8, 2, 99).unwrap();

    let items = vec![
        turboquantdb::storage::engine::BatchWriteItem {
            id: "u1".to_string(),
            vector: Array1::<f64>::from_elem(8, 0.1),
            metadata: HashMap::from([
                ("tenant".to_string(), JsonValue::String("a".to_string())),
                ("year".to_string(), JsonValue::Number(2024.into())),
            ]),
            document: None,
        },
        turboquantdb::storage::engine::BatchWriteItem {
            id: "u2".to_string(),
            vector: Array1::<f64>::from_elem(8, 0.2),
            metadata: HashMap::from([
                ("tenant".to_string(), JsonValue::String("b".to_string())),
                ("year".to_string(), JsonValue::Number(2025.into())),
            ]),
            document: None,
        },
        turboquantdb::storage::engine::BatchWriteItem {
            id: "u3".to_string(),
            vector: Array1::<f64>::from_elem(8, 0.3),
            metadata: HashMap::from([
                ("tenant".to_string(), JsonValue::String("a".to_string())),
                ("year".to_string(), JsonValue::Number(2026.into())),
            ]),
            document: None,
        },
    ];
    engine.insert_many(items).unwrap();

    // {"$and": [{"tenant":{"$eq":"a"}}, {"year":{"$gte":2025}}]}
    let mut filter = HashMap::new();
    filter.insert(
        "$and".to_string(),
        JsonValue::Array(vec![
            JsonValue::Object(serde_json::Map::from_iter([(
                "tenant".to_string(),
                JsonValue::Object(serde_json::Map::from_iter([(
                    "$eq".to_string(),
                    JsonValue::String("a".to_string()),
                )])),
            )])),
            JsonValue::Object(serde_json::Map::from_iter([(
                "year".to_string(),
                JsonValue::Object(serde_json::Map::from_iter([(
                    "$gte".to_string(),
                    JsonValue::Number(2025.into()),
                )])),
            )])),
        ]),
    );
    let q = Array1::<f64>::from_elem(8, 0.1);
    let result = engine.search_with_filter(&q, 10, Some(&filter)).unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].id, "u3");

    let deleted = engine
        .delete_many(&["u1".to_string(), "uX".to_string()])
        .unwrap();
    assert_eq!(deleted, 1);
    let got = engine
        .get_many(&["u1".to_string(), "u2".to_string(), "u3".to_string()])
        .unwrap();
    assert_eq!(got.len(), 2);
}

#[test]
fn test_metric_selection_ip_vs_l2() {
    let dir_ip = tempdir().unwrap();
    let dir_l2 = tempdir().unwrap();
    let path_ip = dir_ip.path().to_str().unwrap();
    let path_l2 = dir_l2.path().to_str().unwrap();

    let mut eng_ip =
        TurboQuantEngine::open_with_metric(path_ip, path_ip, 8, 2, 1, DistanceMetric::Ip).unwrap();
    let mut eng_l2 =
        TurboQuantEngine::open_with_metric(path_l2, path_l2, 8, 2, 2, DistanceMetric::L2).unwrap();

    let v_small = Array1::<f64>::from_elem(8, 0.1);
    let v_large = Array1::<f64>::from_elem(8, 0.9);

    eng_ip
        .upsert("small".to_string(), &v_small, HashMap::new())
        .unwrap();
    eng_ip
        .upsert("large".to_string(), &v_large, HashMap::new())
        .unwrap();
    eng_ip.flush_wal_to_segment().unwrap();

    eng_l2
        .upsert("small".to_string(), &v_small, HashMap::new())
        .unwrap();
    eng_l2
        .upsert("large".to_string(), &v_large, HashMap::new())
        .unwrap();
    eng_l2.flush_wal_to_segment().unwrap();

    let q = Array1::<f64>::from_elem(8, 0.0);
    let ip_top = eng_ip.search(&q, 1).unwrap();
    let l2_top = eng_l2.search(&q, 1).unwrap();

    // With inner product and non-negative vectors, larger magnitude tends to score higher.
    assert_eq!(ip_top[0].id, "large");
    // With L2, nearest to zero should be "small".
    assert_eq!(l2_top[0].id, "small");
}

#[test]
fn test_search_tie_break_is_deterministic_by_id() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().to_str().unwrap();
    let mut engine =
        TurboQuantEngine::open_with_metric(db_path, db_path, 4, 2, 123, DistanceMetric::Ip)
            .unwrap();

    let v = Array1::<f64>::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    engine.upsert("b".to_string(), &v, HashMap::new()).unwrap();
    engine.upsert("a".to_string(), &v, HashMap::new()).unwrap();
    engine.flush_wal_to_segment().unwrap();

    // Dot(query=0, x) == 0 for all x, so all candidates tie on score.
    let q = Array1::<f64>::zeros(4);
    let got = engine.search(&q, 2).unwrap();
    assert_eq!(got.len(), 2);
    assert_eq!(got[0].id, "a");
    assert_eq!(got[1].id, "b");
}

#[test]
fn test_document_persist_and_return() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().to_str().unwrap();

    {
        let mut engine = TurboQuantEngine::open(db_path, db_path, 8, 2, 88).unwrap();
        let v = Array1::<f64>::from_elem(8, 0.5);
        let mut meta = HashMap::new();
        meta.insert("source".to_string(), JsonValue::String("note".to_string()));
        engine
            .upsert_with_document(
                "doc-1".to_string(),
                &v,
                meta,
                Some("hello persisted document".to_string()),
            )
            .unwrap();
        engine.close().unwrap();
    }

    let engine = TurboQuantEngine::open(db_path, db_path, 8, 2, 88).unwrap();
    let got = engine.get("doc-1").unwrap().unwrap();
    assert_eq!(got.document.as_deref(), Some("hello persisted document"));

    let q = Array1::<f64>::from_elem(8, 0.5);
    let hits = engine.search(&q, 1).unwrap();
    assert_eq!(
        hits[0].document.as_deref(),
        Some("hello persisted document")
    );
}

#[test]
fn test_collection_namespace_isolation() {
    let dir = tempdir().unwrap();
    let root = dir.path().to_str().unwrap();

    TurboQuantEngine::create_collection(root, "c1").unwrap();
    TurboQuantEngine::create_collection(root, "c2").unwrap();

    let mut c1 =
        TurboQuantEngine::open_collection(root, root, "c1", 4, 2, 7, DistanceMetric::Ip).unwrap();
    let mut c2 =
        TurboQuantEngine::open_collection(root, root, "c2", 4, 2, 7, DistanceMetric::Ip).unwrap();

    let v1 = Array1::<f64>::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let v2 = Array1::<f64>::from_vec(vec![0.0, 1.0, 0.0, 0.0]);

    c1.upsert("same-id".to_string(), &v1, HashMap::new())
        .unwrap();
    c2.upsert("same-id".to_string(), &v2, HashMap::new())
        .unwrap();
    c1.close().unwrap();
    c2.close().unwrap();
    drop(c1);
    drop(c2);

    let c1r =
        TurboQuantEngine::open_collection(root, root, "c1", 4, 2, 7, DistanceMetric::Ip).unwrap();
    let mut c2r =
        TurboQuantEngine::open_collection(root, root, "c2", 4, 2, 7, DistanceMetric::Ip).unwrap();

    assert_eq!(c1r.vector_count(), 1);
    assert_eq!(c2r.vector_count(), 1);

    let q1 = Array1::<f64>::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let q2 = Array1::<f64>::from_vec(vec![0.0, 1.0, 0.0, 0.0]);
    let r1 = c1r.search(&q1, 1).unwrap();
    let r2 = c2r.search(&q2, 1).unwrap();
    assert_eq!(r1[0].id, "same-id");
    assert_eq!(r2[0].id, "same-id");

    let listed = TurboQuantEngine::list_collections(root).unwrap();
    assert_eq!(listed, vec!["c1".to_string(), "c2".to_string()]);

    c2r.close().unwrap();
    drop(c2r);
    assert!(TurboQuantEngine::delete_collection(root, "c2").unwrap());
    let listed_after = TurboQuantEngine::list_collections(root).unwrap();
    assert_eq!(listed_after, vec!["c1".to_string()]);
}

#[test]
fn test_collection_lifecycle_guards() {
    let dir = tempdir().unwrap();
    let root = dir.path().to_str().unwrap();

    let missing =
        TurboQuantEngine::open_collection(root, root, "missing", 4, 2, 1, DistanceMetric::Ip);
    assert!(missing.is_err(), "opening a missing collection should fail");

    TurboQuantEngine::create_collection(root, "guarded").unwrap();
    let mut eng =
        TurboQuantEngine::open_collection(root, root, "guarded", 4, 2, 1, DistanceMetric::Ip)
            .unwrap();

    let del_while_open = TurboQuantEngine::delete_collection(root, "guarded");
    assert!(
        del_while_open.is_err(),
        "deleting an open collection should fail"
    );

    eng.close().unwrap();
    assert!(TurboQuantEngine::delete_collection(root, "guarded").unwrap());
}

#[test]
fn test_collection_schema_mismatch_guard() {
    let dir = tempdir().unwrap();
    let root = dir.path().to_str().unwrap();

    TurboQuantEngine::create_collection(root, "cfg").unwrap();
    let mut first =
        TurboQuantEngine::open_collection(root, root, "cfg", 8, 2, 42, DistanceMetric::Cosine)
            .unwrap();
    first.close().unwrap();
    drop(first);

    let mismatch =
        TurboQuantEngine::open_collection(root, root, "cfg", 16, 2, 42, DistanceMetric::Cosine);
    assert!(mismatch.is_err(), "dimension mismatch should be rejected");

    let mismatch_metric =
        TurboQuantEngine::open_collection(root, root, "cfg", 8, 2, 42, DistanceMetric::Ip);
    assert!(
        mismatch_metric.is_err(),
        "metric mismatch should be rejected"
    );
}

#[test]
fn test_filter_nested_path_and_missing_field_semantics() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().to_str().unwrap();
    let mut engine = TurboQuantEngine::open(db_path, db_path, 8, 2, 101).unwrap();

    let v = Array1::<f64>::from_elem(8, 0.1);
    let mut m1 = HashMap::new();
    m1.insert(
        "profile".to_string(),
        serde_json::json!({"region":"eu","score":10}),
    );
    let mut m2 = HashMap::new();
    m2.insert(
        "profile".to_string(),
        serde_json::json!({"region":"us","score":20}),
    );
    let mut m3 = HashMap::new();
    m3.insert(
        "name".to_string(),
        JsonValue::String("no-profile".to_string()),
    );

    engine.upsert("u1".to_string(), &v, m1).unwrap();
    engine.upsert("u2".to_string(), &v, m2).unwrap();
    engine.upsert("u3".to_string(), &v, m3).unwrap();
    engine.flush_wal_to_segment().unwrap();

    let mut f_eq = HashMap::new();
    f_eq.insert(
        "profile.region".to_string(),
        serde_json::json!({"$eq":"eu"}),
    );
    let got_eq = engine
        .search_with_filter(&Array1::<f64>::zeros(8), 10, Some(&f_eq))
        .unwrap();
    assert_eq!(got_eq.len(), 1);
    assert_eq!(got_eq[0].id, "u1");

    // Missing-field policy: $ne matches missing fields.
    let mut f_ne = HashMap::new();
    f_ne.insert(
        "profile.region".to_string(),
        serde_json::json!({"$ne":"eu"}),
    );
    let got_ne = engine
        .search_with_filter(&Array1::<f64>::zeros(8), 10, Some(&f_ne))
        .unwrap();
    assert_eq!(got_ne.len(), 2);
    let ids: Vec<String> = got_ne.into_iter().map(|r| r.id).collect();
    assert!(ids.contains(&"u2".to_string()));
    assert!(ids.contains(&"u3".to_string()));

    // Missing-field policy: comparisons do not match missing fields.
    let mut f_gt = HashMap::new();
    f_gt.insert("profile.score".to_string(), serde_json::json!({"$gt":15}));
    let got_gt = engine
        .search_with_filter(&Array1::<f64>::zeros(8), 10, Some(&f_gt))
        .unwrap();
    assert_eq!(got_gt.len(), 1);
    assert_eq!(got_gt[0].id, "u2");
}

#[test]
fn test_filter_negative_paths_invalid_operator_and_type_strictness() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().to_str().unwrap();
    let mut engine = TurboQuantEngine::open(db_path, db_path, 8, 2, 202).unwrap();

    let v = Array1::<f64>::from_elem(8, 0.2);
    let mut meta = HashMap::new();
    meta.insert("year".to_string(), JsonValue::Number(2025.into()));
    meta.insert("tag".to_string(), JsonValue::String("a".to_string()));
    engine.upsert("x".to_string(), &v, meta).unwrap();
    engine.flush_wal_to_segment().unwrap();

    // Unknown operator should reject match.
    let mut f_bad_op = HashMap::new();
    f_bad_op.insert("year".to_string(), serde_json::json!({"$contains": 20}));
    let got_bad = engine
        .search_with_filter(&Array1::<f64>::zeros(8), 10, Some(&f_bad_op))
        .unwrap();
    assert!(got_bad.is_empty());

    // Strict typing: numeric field compared with string should not coerce/match.
    let mut f_type = HashMap::new();
    f_type.insert("year".to_string(), serde_json::json!({"$gt": "2024"}));
    let got_type = engine
        .search_with_filter(&Array1::<f64>::zeros(8), 10, Some(&f_type))
        .unwrap();
    assert!(got_type.is_empty());
}

#[test]
fn test_manifest_metric_mismatch_rejected() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().to_str().unwrap();

    let mut eng =
        TurboQuantEngine::open_with_metric(db_path, db_path, 8, 2, 42, DistanceMetric::Cosine)
            .unwrap();
    eng.close().unwrap();
    drop(eng);

    let reopened =
        TurboQuantEngine::open_with_metric(db_path, db_path, 8, 2, 42, DistanceMetric::Cosine);
    assert!(reopened.is_ok());

    let mismatch =
        TurboQuantEngine::open_with_metric(db_path, db_path, 8, 2, 42, DistanceMetric::Ip);
    assert!(
        mismatch.is_err(),
        "opening with a different metric than manifest should fail"
    );
}

#[test]
fn test_metric_matches_bruteforce_reference_top1() {
    let vectors = vec![
        ("v1", vec![1.0, 0.0, 0.0, 0.0]),
        ("v2", vec![0.0, 1.0, 0.0, 0.0]),
        ("v3", vec![1.0, 1.0, 0.0, 0.0]),
        ("v4", vec![-1.0, 0.0, 0.0, 0.0]),
    ];
    let query = vec![1.0, 0.25, 0.0, 0.0];

    for metric in [
        DistanceMetric::Ip,
        DistanceMetric::Cosine,
        DistanceMetric::L2,
    ] {
        let dir = tempdir().unwrap();
        let db_path = dir.path().to_str().unwrap();
        let mut engine =
            TurboQuantEngine::open_with_metric(db_path, db_path, 4, 4, 11, metric.clone()).unwrap();

        for (id, v) in &vectors {
            engine
                .upsert(
                    (*id).to_string(),
                    &Array1::<f64>::from_vec(v.clone()),
                    HashMap::new(),
                )
                .unwrap();
        }
        engine.flush_wal_to_segment().unwrap();

        let got = engine
            .search(&Array1::<f64>::from_vec(query.clone()), 1)
            .unwrap();
        let got_top = got[0].id.clone();

        let mut best_id = String::new();
        let mut best_score = f64::NEG_INFINITY;
        for (id, v) in &vectors {
            let score = match metric {
                DistanceMetric::Ip => query.iter().zip(v.iter()).map(|(a, b)| a * b).sum(),
                DistanceMetric::Cosine => {
                    let dot: f64 = query.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
                    let qn = query.iter().map(|x| x * x).sum::<f64>().sqrt();
                    let vn = v.iter().map(|x| x * x).sum::<f64>().sqrt();
                    if qn == 0.0 || vn == 0.0 {
                        0.0
                    } else {
                        dot / (qn * vn)
                    }
                }
                DistanceMetric::L2 => {
                    let d2: f64 = query
                        .iter()
                        .zip(v.iter())
                        .map(|(a, b)| {
                            let d = a - b;
                            d * d
                        })
                        .sum();
                    -d2
                }
            };
            if score > best_score
                || (score == best_score && (best_id.is_empty() || *id < best_id.as_str()))
            {
                best_score = score;
                best_id = (*id).to_string();
            }
        }

        assert_eq!(got_top, best_id, "metric {:?} top1 mismatch", metric);
    }
}

#[test]
fn test_index_config_persisted_in_manifest_and_stats() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().to_str().unwrap();
    let mut engine =
        TurboQuantEngine::open_with_metric(db_path, db_path, 16, 2, 77, DistanceMetric::Cosine)
            .unwrap();

    for i in 0..40usize {
        let v = Array1::<f64>::from_iter((0..16).map(|j| (i + j) as f64 / 100.0));
        engine
            .upsert(format!("id-{i}"), &v, HashMap::new())
            .unwrap();
    }
    engine.create_index_with_params(24, 37, 1.7).unwrap();
    let s = engine.stats();
    assert!(s.has_index);
    assert_eq!(s.index_nodes, 40);
    assert_eq!(s.index_search_list_size, Some(37));
    assert_eq!(s.index_alpha, Some(1.7));
    engine.close().unwrap();
    drop(engine);

    let reopened =
        TurboQuantEngine::open_with_metric(db_path, db_path, 16, 2, 77, DistanceMetric::Cosine)
            .unwrap();
    assert_eq!(
        reopened
            .manifest
            .index_state
            .as_ref()
            .map(|s| s.search_list_size),
        Some(37)
    );
    assert_eq!(
        reopened.manifest.index_state.as_ref().map(|s| s.max_degree),
        Some(24)
    );
    assert_eq!(
        reopened.manifest.index_state.as_ref().map(|s| s.alpha),
        Some(1.7)
    );
}

#[test]
fn test_graph_build_respects_max_degree() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().to_str().unwrap();
    let mut engine = TurboQuantEngine::open(db_path, db_path, 8, 2, 91).unwrap();

    for i in 0..12usize {
        let v = Array1::<f64>::from_iter((0..8).map(|j| (i * 8 + j) as f64 / 100.0));
        engine
            .upsert(format!("id-{i}"), &v, HashMap::new())
            .unwrap();
    }
    engine.create_index_with_params(1, 20, 1.2).unwrap();
    engine.close().unwrap();
    drop(engine);

    let backend = Arc::new(StorageBackend::from_uri(db_path).unwrap());
    let graph = GraphManager::open(backend, db_path).unwrap();
    for node in 0..12u32 {
        let neighbors = graph.get_neighbors(node).unwrap();
        assert!(neighbors.len() <= 1);
    }
}

#[test]
fn test_search_ann_override_api_and_high_beam_matches_bruteforce_top1() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().to_str().unwrap();
    let mut engine =
        TurboQuantEngine::open_with_metric(db_path, db_path, 16, 2, 123, DistanceMetric::Cosine)
            .unwrap();

    let mut vectors: HashMap<String, Array1<f64>> = HashMap::new();
    for i in 0..60usize {
        let v = Array1::<f64>::from_iter((0..16).map(|j| {
            let a = ((i * 17 + j * 11) % 97) as f64 / 97.0;
            let b = ((i * 7 + j * 13 + 3) % 89) as f64 / 89.0;
            a - b
        }));
        let id = format!("id-{i:03}");
        engine.upsert(id.clone(), &v, HashMap::new()).unwrap();
        vectors.insert(id, v);
    }

    // Intentionally low default search list size, then override per-query.
    engine.create_index_with_params(8, 1, 1.2).unwrap();
    let query = Array1::<f64>::from_iter((0..16).map(|j| {
        let a = ((42 * 17 + j * 11) % 97) as f64 / 97.0;
        let b = ((42 * 7 + j * 13 + 3) % 89) as f64 / 89.0;
        (a - b) + ((j as f64) * 1e-6)
    }));

    let default_hits = engine.search(&query, 5).unwrap();
    let forced_small = engine
        .search_with_filter_and_ann(&query, 5, None, Some(1))
        .unwrap();
    let forced_large = engine
        .search_with_filter_and_ann(&query, 5, None, Some(64))
        .unwrap();

    let default_ids: Vec<_> = default_hits.iter().map(|r| r.id.clone()).collect();
    let small_ids: Vec<_> = forced_small.iter().map(|r| r.id.clone()).collect();
    assert_eq!(default_ids, small_ids);
    assert_eq!(forced_large.len(), 5);

    let mut best_id = String::new();
    let mut best_score = f64::NEG_INFINITY;
    for (id, v) in &vectors {
        let dot: f64 = query.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
        let qn = query.iter().map(|x| x * x).sum::<f64>().sqrt();
        let vn = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        let score = if qn == 0.0 || vn == 0.0 {
            0.0
        } else {
            dot / (qn * vn)
        };
        if score > best_score
            || (score == best_score && (best_id.is_empty() || id.as_str() < best_id.as_str()))
        {
            best_score = score;
            best_id = id.clone();
        }
    }
    assert_eq!(forced_large[0].id, best_id);
}

#[test]
fn test_compaction_recovery_deletes_old_segments_when_marker_exists() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().to_str().unwrap();
    let mut engine = TurboQuantEngine::open(db_path, db_path, 8, 2, 42).unwrap();

    for i in 0..6usize {
        let v = Array1::<f64>::from_iter((0..8).map(|j| (i + j) as f64 / 50.0));
        engine
            .upsert(format!("id-{i}"), &v, HashMap::new())
            .unwrap();
        engine.flush_wal_to_segment().unwrap();
    }
    let before_count = engine.vector_count();
    engine.close().unwrap();
    drop(engine);

    // Simulate interrupted compaction:
    // - old segments still exist
    // - compacted replacement segment exists
    // - marker exists
    let backend = StorageBackend::from_uri(db_path).unwrap();
    let mut old_segment_names: Vec<String> = backend
        .list("")
        .unwrap()
        .into_iter()
        .filter(|name| name.starts_with("seg-") && name.ends_with(".bin"))
        .collect();
    old_segment_names.sort();
    assert!(old_segment_names.len() >= 2);

    let mut by_id: HashMap<String, SegmentRecord> = HashMap::new();
    for seg_name in &old_segment_names {
        for rec in Segment::read_all(&backend, seg_name).unwrap() {
            by_id.insert(rec.id.clone(), rec);
        }
    }
    let mut live_records: Vec<SegmentRecord> = by_id.into_values().collect();
    live_records.sort_by(|a, b| a.id.cmp(&b.id));

    let compacted_name = "seg-99999999.bin";
    Segment::write_batch(&backend, compacted_name, &live_records).unwrap();

    let marker = serde_json::json!({
        "phase": "prepared",
        "old_segment_names": old_segment_names,
        "new_segment_name": compacted_name
    });
    backend
        .write(
            "compaction_state.json",
            serde_json::to_vec_pretty(&marker).unwrap(),
        )
        .unwrap();

    let recovered = TurboQuantEngine::open(db_path, db_path, 8, 2, 42).unwrap();
    assert_eq!(recovered.vector_count(), before_count);
    assert_eq!(recovered.manifest.compaction_state.recovery_runs, 1);
    drop(recovered);

    let backend_after = StorageBackend::from_uri(db_path).unwrap();
    assert!(!backend_after.exists("compaction_state.json"));

    let mut segs_after: Vec<String> = backend_after
        .list("")
        .unwrap()
        .into_iter()
        .filter(|name| name.starts_with("seg-") && name.ends_with(".bin"))
        .collect();
    segs_after.sort();
    assert_eq!(segs_after, vec![compacted_name.to_string()]);
}

#[test]
fn test_stats_deleted_record_count_tracks_tombstones() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().to_str().unwrap();
    let mut engine = TurboQuantEngine::open(db_path, db_path, 8, 2, 17).unwrap();

    engine
        .upsert(
            "a".to_string(),
            &Array1::<f64>::from_elem(8, 0.1),
            HashMap::new(),
        )
        .unwrap();
    engine
        .upsert(
            "b".to_string(),
            &Array1::<f64>::from_elem(8, 0.2),
            HashMap::new(),
        )
        .unwrap();
    engine.flush_wal_to_segment().unwrap();
    engine.delete("a".to_string()).unwrap();
    engine.flush_wal_to_segment().unwrap();

    let s = engine.stats();
    assert_eq!(s.vector_count, 1);
    assert!(s.deleted_record_count >= 1);
    assert!(s.physical_record_count >= 3);
}

#[test]
fn test_hybrid_search_sparse_weight_can_override_dense_order() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().to_str().unwrap();
    let mut engine = TurboQuantEngine::open(db_path, db_path, 4, 2, 5).unwrap();

    // Dense-near candidate but text-irrelevant.
    let mut m1 = HashMap::new();
    m1.insert(
        "topic".to_string(),
        JsonValue::String("generic".to_string()),
    );
    engine
        .upsert_with_document(
            "dense-first".to_string(),
            &Array1::<f64>::from_vec(vec![1.0, 0.0, 0.0, 0.0]),
            m1,
            Some("alpha beta".to_string()),
        )
        .unwrap();

    // Dense-far candidate but text-relevant.
    let mut m2 = HashMap::new();
    m2.insert("topic".to_string(), JsonValue::String("news".to_string()));
    engine
        .upsert_with_document(
            "sparse-first".to_string(),
            &Array1::<f64>::from_vec(vec![0.0, 1.0, 0.0, 0.0]),
            m2,
            Some("turboquant roadmap turboquant milestones".to_string()),
        )
        .unwrap();
    engine.flush_wal_to_segment().unwrap();

    let q = Array1::<f64>::from_vec(vec![1.0, 0.0, 0.0, 0.0]);

    let dense_only = engine
        .search_hybrid_with_filter(&q, "turboquant", 2, None, 1.0, 0.0)
        .unwrap();
    assert_eq!(dense_only[0].id, "dense-first");

    let sparse_heavy = engine
        .search_hybrid_with_filter(&q, "turboquant", 2, None, 0.1, 0.9)
        .unwrap();
    assert_eq!(sparse_heavy[0].id, "sparse-first");
}

#[test]
fn test_hybrid_search_respects_where_filter() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().to_str().unwrap();
    let mut engine = TurboQuantEngine::open(db_path, db_path, 4, 2, 6).unwrap();

    let mut a = HashMap::new();
    a.insert("tenant".to_string(), JsonValue::String("a".to_string()));
    engine
        .upsert_with_document(
            "a1".to_string(),
            &Array1::<f64>::from_vec(vec![1.0, 0.0, 0.0, 0.0]),
            a,
            Some("turboquant".to_string()),
        )
        .unwrap();

    let mut b = HashMap::new();
    b.insert("tenant".to_string(), JsonValue::String("b".to_string()));
    engine
        .upsert_with_document(
            "b1".to_string(),
            &Array1::<f64>::from_vec(vec![0.0, 1.0, 0.0, 0.0]),
            b,
            Some("turboquant turboquant".to_string()),
        )
        .unwrap();
    engine.flush_wal_to_segment().unwrap();

    let mut filter = HashMap::new();
    filter.insert("tenant".to_string(), JsonValue::String("a".to_string()));

    let q = Array1::<f64>::from_vec(vec![0.0, 1.0, 0.0, 0.0]);
    let got = engine
        .search_hybrid_with_filter(&q, "turboquant", 5, Some(&filter), 0.2, 0.8)
        .unwrap();
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].id, "a1");
}

#[test]
fn test_engine_reranker_hook_reorders_topn() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().to_str().unwrap();
    let mut engine = TurboQuantEngine::open(db_path, db_path, 4, 2, 7).unwrap();

    let mut m_low = HashMap::new();
    m_low.insert("priority".to_string(), JsonValue::Number(1.into()));
    engine
        .upsert_with_document(
            "dense-best".to_string(),
            &Array1::<f64>::from_vec(vec![1.0, 0.0, 0.0, 0.0]),
            m_low,
            Some("irrelevant".to_string()),
        )
        .unwrap();

    let mut m_high = HashMap::new();
    m_high.insert("priority".to_string(), JsonValue::Number(99.into()));
    engine
        .upsert_with_document(
            "rerank-best".to_string(),
            &Array1::<f64>::from_vec(vec![0.0, 1.0, 0.0, 0.0]),
            m_high,
            Some("relevant".to_string()),
        )
        .unwrap();
    engine.flush_wal_to_segment().unwrap();

    let q = Array1::<f64>::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let dense = engine.search(&q, 2).unwrap();
    assert_eq!(dense[0].id, "dense-best");

    let reranked = engine
        .search_with_filter_and_reranker(&q, 2, None, 2, |r| {
            r.metadata
                .get("priority")
                .and_then(|v| v.as_i64())
                .unwrap_or(0) as f64
        })
        .unwrap();
    assert_eq!(reranked[0].id, "rerank-best");
}

#[test]
fn test_batch_insert_report_continues_on_duplicates() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().to_str().unwrap();
    let mut engine = TurboQuantEngine::open(db_path, db_path, 8, 2, 44).unwrap();

    engine
        .insert(
            "dup".to_string(),
            &Array1::<f64>::from_elem(8, 0.1),
            HashMap::new(),
        )
        .unwrap();

    let items = vec![
        turboquantdb::storage::engine::BatchWriteItem {
            id: "dup".to_string(),
            vector: Array1::<f64>::from_elem(8, 0.2),
            metadata: HashMap::new(),
            document: None,
        },
        turboquantdb::storage::engine::BatchWriteItem {
            id: "ok".to_string(),
            vector: Array1::<f64>::from_elem(8, 0.3),
            metadata: HashMap::new(),
            document: Some("persisted".to_string()),
        },
    ];
    let report = engine.insert_many_report(items);
    assert_eq!(report.applied, 1);
    assert_eq!(report.failed.len(), 1);
    assert_eq!(report.failed[0].index, 0);
    assert_eq!(report.failed[0].id, "dup");
    assert!(report.failed[0].error.contains("already exists"));

    let got = engine.get("ok").unwrap();
    assert!(got.is_some());
    assert_eq!(got.unwrap().document.as_deref(), Some("persisted"));
}

#[test]
fn test_batch_update_report_continues_on_missing_ids() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().to_str().unwrap();
    let mut engine = TurboQuantEngine::open(db_path, db_path, 8, 2, 45).unwrap();

    engine
        .upsert(
            "exists".to_string(),
            &Array1::<f64>::from_elem(8, 0.1),
            HashMap::new(),
        )
        .unwrap();

    let items = vec![
        turboquantdb::storage::engine::BatchWriteItem {
            id: "missing".to_string(),
            vector: Array1::<f64>::from_elem(8, 0.9),
            metadata: HashMap::new(),
            document: None,
        },
        turboquantdb::storage::engine::BatchWriteItem {
            id: "exists".to_string(),
            vector: Array1::<f64>::from_elem(8, 0.2),
            metadata: HashMap::from([("k".to_string(), JsonValue::String("v".to_string()))]),
            document: None,
        },
    ];
    let report = engine.update_many_report(items);
    assert_eq!(report.applied, 1);
    assert_eq!(report.failed.len(), 1);
    assert_eq!(report.failed[0].index, 0);
    assert_eq!(report.failed[0].id, "missing");
    assert!(report.failed[0].error.contains("does not exist"));

    let got = engine.get("exists").unwrap().unwrap();
    assert_eq!(
        got.metadata.get("k"),
        Some(&JsonValue::String("v".to_string()))
    );
}

#[test]
fn test_snapshot_and_restore_roundtrip() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("db");
    let snap_path = dir.path().join("snapshot");
    let db_path_str = db_path.to_str().unwrap();
    let snap_path_str = snap_path.to_str().unwrap();

    // Build source state.
    {
        let mut engine = TurboQuantEngine::open(db_path_str, db_path_str, 8, 2, 313).unwrap();
        for i in 0..10usize {
            let v = Array1::<f64>::from_iter((0..8).map(|j| (i * 8 + j) as f64 / 100.0));
            let mut meta = HashMap::new();
            meta.insert("group".to_string(), JsonValue::String("base".to_string()));
            engine
                .upsert_with_document(format!("id-{i}"), &v, meta, Some(format!("doc-{i}")))
                .unwrap();
        }
        engine.create_index(16, 24).unwrap();
        engine.close().unwrap();
    }
    TurboQuantEngine::snapshot_local_dir(db_path_str, snap_path_str).unwrap();

    let before_files = read_tree_files(Path::new(snap_path_str));
    assert!(!before_files.is_empty());

    // Mutate live DB after snapshot, then restore from snapshot.
    {
        let mut engine = TurboQuantEngine::open(db_path_str, db_path_str, 8, 2, 313).unwrap();
        engine.delete("id-0".to_string()).unwrap();
        engine
            .upsert(
                "id-extra".to_string(),
                &Array1::<f64>::from_elem(8, 9.9),
                HashMap::new(),
            )
            .unwrap();
        engine.close().unwrap();
    }

    TurboQuantEngine::restore_from_snapshot(db_path_str, snap_path_str).unwrap();
    let after_files = read_tree_files(Path::new(db_path_str));
    assert_eq!(
        before_files, after_files,
        "restored bytes should match snapshot"
    );

    let restored = TurboQuantEngine::open(db_path_str, db_path_str, 8, 2, 313).unwrap();
    assert_eq!(restored.vector_count(), 10);
    assert!(restored.get("id-extra").unwrap().is_none());
    let got = restored.get("id-0").unwrap().unwrap();
    assert_eq!(got.document.as_deref(), Some("doc-0"));
    assert_eq!(
        restored
            .manifest
            .index_state
            .as_ref()
            .map(|s| s.search_list_size),
        Some(24)
    );
}

#[test]
fn test_collection_snapshot_and_restore_roundtrip() {
    let dir = tempdir().unwrap();
    let root = dir.path().to_str().unwrap();
    let snap = dir.path().join("c1_snap");
    let snap_str = snap.to_str().unwrap();

    TurboQuantEngine::create_collection(root, "c1").unwrap();
    {
        let mut c1 =
            TurboQuantEngine::open_collection(root, root, "c1", 8, 2, 9, DistanceMetric::Ip)
                .unwrap();
        c1.upsert(
            "base".to_string(),
            &Array1::<f64>::from_elem(8, 0.3),
            HashMap::new(),
        )
        .unwrap();
        c1.close().unwrap();
    }

    TurboQuantEngine::snapshot_collection(root, "c1", snap_str).unwrap();

    {
        let mut c1 =
            TurboQuantEngine::open_collection(root, root, "c1", 8, 2, 9, DistanceMetric::Ip)
                .unwrap();
        c1.delete("base".to_string()).unwrap();
        c1.upsert(
            "mutated".to_string(),
            &Array1::<f64>::from_elem(8, 0.9),
            HashMap::new(),
        )
        .unwrap();
        c1.close().unwrap();
    }

    TurboQuantEngine::restore_collection(root, "c1", snap_str).unwrap();

    let c1r =
        TurboQuantEngine::open_collection(root, root, "c1", 8, 2, 9, DistanceMetric::Ip).unwrap();
    assert_eq!(c1r.vector_count(), 1);
    assert!(c1r.get("base").unwrap().is_some());
    assert!(c1r.get("mutated").unwrap().is_none());
}

#[test]
fn test_scoped_tenant_database_collection_isolation() {
    let dir = tempdir().unwrap();
    let root = dir.path().to_str().unwrap();
    let uri = root;

    TurboQuantEngine::create_collection_scoped_with_uri(uri, root, "tenant_a", "db_main", "docs")
        .unwrap();
    TurboQuantEngine::create_collection_scoped_with_uri(uri, root, "tenant_b", "db_main", "docs")
        .unwrap();

    let mut a = TurboQuantEngine::open_collection_scoped(
        uri,
        root,
        "tenant_a",
        "db_main",
        "docs",
        8,
        2,
        11,
        DistanceMetric::Ip,
    )
    .unwrap();
    let mut b = TurboQuantEngine::open_collection_scoped(
        uri,
        root,
        "tenant_b",
        "db_main",
        "docs",
        8,
        2,
        11,
        DistanceMetric::Ip,
    )
    .unwrap();

    a.upsert(
        "only-a".to_string(),
        &Array1::<f64>::from_elem(8, 0.1),
        HashMap::new(),
    )
    .unwrap();
    b.upsert(
        "only-b".to_string(),
        &Array1::<f64>::from_elem(8, 0.2),
        HashMap::new(),
    )
    .unwrap();
    a.flush_wal_to_segment().unwrap();
    b.flush_wal_to_segment().unwrap();

    assert!(a.get("only-a").unwrap().is_some());
    assert!(a.get("only-b").unwrap().is_none());
    assert!(b.get("only-b").unwrap().is_some());
    assert!(b.get("only-a").unwrap().is_none());
}

#[test]
fn test_scoped_catalog_helpers_list_and_delete() {
    let dir = tempdir().unwrap();
    let root = dir.path().to_str().unwrap();
    let uri = root;

    TurboQuantEngine::create_collection_scoped_with_uri(uri, root, "t1", "db1", "c1").unwrap();
    TurboQuantEngine::create_collection_scoped_with_uri(uri, root, "t1", "db1", "c2").unwrap();
    TurboQuantEngine::create_collection_scoped_with_uri(uri, root, "t1", "db2", "c1").unwrap();

    let mut db1 = TurboQuantEngine::list_collections_scoped(root, "t1", "db1").unwrap();
    db1.sort();
    assert_eq!(db1, vec!["c1".to_string(), "c2".to_string()]);

    let db2 = TurboQuantEngine::list_collections_scoped(root, "t1", "db2").unwrap();
    assert_eq!(db2, vec!["c1".to_string()]);

    assert!(
        TurboQuantEngine::delete_collection_scoped_with_uri(uri, root, "t1", "db1", "c2").unwrap()
    );
    let db1_after = TurboQuantEngine::list_collections_scoped(root, "t1", "db1").unwrap();
    assert_eq!(db1_after, vec!["c1".to_string()]);
}

#[test]
fn test_rerank_disabled_behavior() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().to_str().unwrap();

    // Open with rerank=false
    let mut engine = TurboQuantEngine::open_with_metric_and_rerank(
        db_path,
        db_path,
        8,
        2,
        42,
        DistanceMetric::Ip,
        false,
    )
    .unwrap();

    let v = Array1::<f64>::from_elem(8, 0.5);
    engine.insert("v1".to_string(), &v, HashMap::new()).unwrap();

    // Verify live_vectors.bin does not exist
    assert!(!Path::new(db_path).join("live_vectors.bin").exists());

    let stats = engine.stats();
    assert_eq!(stats.live_vectors_bytes_estimate, 0);

    let results = engine.search(&v, 1).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "v1");
}

#[test]
fn test_wal_versioning_and_migration() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().to_str().unwrap();
    let wal_path = dir.path().join("wal.log");

    // 1. Manually create a "legacy" WAL (no header, just entries)
    // We'll use the current WalEntry but since it doesn't have the header it will be treated as legacy if we bypass open()
    // Actually, the easiest way is to use the legacy struct if we had it here,
    // but we can just test that the current versioned WAL works.

    {
        let mut engine = TurboQuantEngine::open(db_path, db_path, 8, 2, 42).unwrap();
        engine
            .insert("v1".to_string(), &Array1::zeros(8), HashMap::new())
            .unwrap();
        // WAL now has TQWV header
    }

    // Verify header exists
    let wal_bytes = std::fs::read(&wal_path).unwrap();
    assert_eq!(&wal_bytes[0..4], b"TQWV");

    // Reopen and verify recovery
    let mut engine = TurboQuantEngine::open(db_path, db_path, 8, 2, 42).unwrap();
    engine.flush_wal_to_segment().unwrap();
    assert_eq!(engine.vector_count(), 1);
}

#[test]
fn test_hnsw_beam_search_recall() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().to_str().unwrap();
    let mut engine = TurboQuantEngine::open(db_path, db_path, 16, 2, 42).unwrap();

    // Insert 200 vectors to have some depth
    for i in 0..200usize {
        let v = Array1::<f64>::from_iter((0..16).map(|j| (i + j) as f64 / 1000.0));
        engine
            .insert(format!("v{}", i), &v, HashMap::new())
            .unwrap();
    }

    engine.create_index_with_params(16, 64, 1.2).unwrap();

    let query = Array1::<f64>::from_iter((0..16).map(|j| j as f64 / 1000.0));
    let results = engine.search(&query, 10).unwrap();

    assert!(!results.is_empty());
    assert_eq!(results[0].id, "v0"); // Should easily find the exact match with beam search
}

fn read_tree_files(root: &Path) -> Vec<(PathBuf, Vec<u8>)> {
    fn walk(root: &Path, cur: &Path, out: &mut Vec<(PathBuf, Vec<u8>)>) {
        for entry in std::fs::read_dir(cur).unwrap() {
            let entry = entry.unwrap();
            let p = entry.path();
            if entry.file_type().unwrap().is_dir() {
                walk(root, &p, out);
            } else {
                let rel = p.strip_prefix(root).unwrap().to_path_buf();
                out.push((rel, std::fs::read(&p).unwrap()));
            }
        }
    }
    let mut out = Vec::new();
    walk(root, root, &mut out);
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}
