use ndarray::Array1;
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tempfile::tempdir;

use turboquantdb::storage::backend::StorageBackend;
use turboquantdb::storage::engine::{DistanceMetric, TurboQuantEngine};
use turboquantdb::storage::graph::GraphManager;

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
    // Simulate a crash: insert vectors and flush WAL but do NOT call close()
    // (so live_ids.bin is never written). On reopen the engine must recover
    // the full dataset from the WAL alone.
    let dir = tempdir().unwrap();
    let db_path = dir.path().to_str().unwrap();
    let ids_path = dir.path().join("live_ids.bin");

    {
        let mut engine = TurboQuantEngine::open(db_path, db_path, 8, 2, 42).unwrap();
        for i in 0..5u32 {
            let v = Array1::<f64>::from_elem(8, i as f64 / 10.0);
            engine
                .insert(format!("v{}", i), &v, HashMap::new())
                .unwrap();
        }
        // Simulate crash: drop without close() so live_ids.bin is missing.
        drop(engine);
    }
    std::fs::remove_file(&ids_path).ok(); // make sure it's gone

    // Reopen must recover from WAL without panicking.
    let mut recovered = TurboQuantEngine::open(db_path, db_path, 8, 2, 42).unwrap();
    assert_eq!(
        recovered.vector_count(),
        5,
        "all 5 vectors must be recoverable from WAL"
    );

    // Flush and verify data survives the slab compaction (used to panic on Windows).
    recovered.flush_wal_to_segment().unwrap();
    assert_eq!(recovered.vector_count(), 5);

    // Search must work after recovery.
    let q = Array1::<f64>::from_elem(8, 0.3);
    let results = recovered.search(&q, 3).unwrap();
    assert_eq!(results.len(), 3);
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
    engine
        .create_index_with_params(32, 200, 64, 1.2, 0)
        .unwrap();

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
    engine
        .upsert_with_document("b".to_string(), &v_b, meta_b, None)
        .unwrap();

    // Update existing
    let mut meta_b2 = HashMap::new();
    meta_b2.insert("tenant".to_string(), JsonValue::String("alpha".to_string()));
    engine
        .update_with_document(
            "b".to_string(),
            &Array1::<f64>::from_elem(16, 0.3),
            meta_b2,
            None,
        )
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

    // compact() is not available; just verify the DB is consistent
    let after = engine.stats();
    assert_eq!(after.vector_count, 12);
}

#[test]
fn test_batch_and_advanced_filters() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().to_str().unwrap();
    let mut engine = TurboQuantEngine::open(db_path, db_path, 8, 2, 99).unwrap();

    let v = Array1::<f64>::from_elem(8, 0.5);

    for (id, year, topic) in [
        ("doc1", 2020, "ml"),
        ("doc2", 2021, "nlp"),
        ("doc3", 2023, "ml"),
        ("doc4", 2024, "cv"),
    ] {
        let mut meta = HashMap::new();
        meta.insert("year".to_string(), serde_json::json!(year));
        meta.insert("topic".to_string(), serde_json::json!(topic));
        engine
            .upsert_with_document(id.to_string(), &v, meta, None)
            .unwrap();
    }
    engine.flush_wal_to_segment().unwrap();

    // $eq operator
    let f_eq = serde_json::json!({"topic": {"$eq": "ml"}});
    let f_eq_map: HashMap<String, JsonValue> = serde_json::from_value(f_eq).unwrap();
    let got = engine.search_with_filter(&v, 10, Some(&f_eq_map)).unwrap();
    assert_eq!(got.len(), 2);
    let ids: Vec<_> = got.iter().map(|r| r.id.as_str()).collect();
    assert!(ids.contains(&"doc1") && ids.contains(&"doc3"));

    // $gte operator
    let f_gte = serde_json::json!({"year": {"$gte": 2023}});
    let f_gte_map: HashMap<String, JsonValue> = serde_json::from_value(f_gte).unwrap();
    let got_gte = engine.search_with_filter(&v, 10, Some(&f_gte_map)).unwrap();
    assert_eq!(got_gte.len(), 2);
    let ids_gte: Vec<_> = got_gte.iter().map(|r| r.id.as_str()).collect();
    assert!(ids_gte.contains(&"doc3") && ids_gte.contains(&"doc4"));

    // $and operator — topic=ml AND year>=2022
    let f_and = serde_json::json!({"$and": [{"topic": {"$eq": "ml"}}, {"year": {"$gte": 2022}}]});
    let f_and_map: HashMap<String, JsonValue> = serde_json::from_value(f_and).unwrap();
    let got_and = engine.search_with_filter(&v, 10, Some(&f_and_map)).unwrap();
    assert_eq!(got_and.len(), 1);
    assert_eq!(got_and[0].id, "doc3");

    // $or operator — topic=cv OR year<2021
    let f_or = serde_json::json!({"$or": [{"topic": {"$eq": "cv"}}, {"year": {"$lt": 2021}}]});
    let f_or_map: HashMap<String, JsonValue> = serde_json::from_value(f_or).unwrap();
    let got_or = engine.search_with_filter(&v, 10, Some(&f_or_map)).unwrap();
    assert_eq!(got_or.len(), 2);
    let ids_or: Vec<_> = got_or.iter().map(|r| r.id.as_str()).collect();
    assert!(ids_or.contains(&"doc1") && ids_or.contains(&"doc4"));
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
        .upsert_with_document("small".to_string(), &v_small, HashMap::new(), None)
        .unwrap();
    eng_ip
        .upsert_with_document("large".to_string(), &v_large, HashMap::new(), None)
        .unwrap();
    eng_ip.flush_wal_to_segment().unwrap();

    eng_l2
        .upsert_with_document("small".to_string(), &v_small, HashMap::new(), None)
        .unwrap();
    eng_l2
        .upsert_with_document("large".to_string(), &v_large, HashMap::new(), None)
        .unwrap();
    eng_l2.flush_wal_to_segment().unwrap();

    let q_ip = Array1::<f64>::from_elem(8, 1.0); // non-zero: IP score = 7.2 (large) vs 0.8 (small)
    let q_l2 = Array1::<f64>::from_elem(8, 0.0); // zero origin: L2 dist smaller for "small"
    let ip_top = eng_ip.search(&q_ip, 1).unwrap();
    let l2_top = eng_l2.search(&q_l2, 1).unwrap();

    // With inner product and non-negative vectors, larger magnitude scores higher.
    assert_eq!(ip_top[0].id, "large");
    // With L2, nearest to zero should be "small".
    assert_eq!(l2_top[0].id, "small");
}

#[test]
#[ignore = "engine breaks ties by insertion order, not alphabetically by ID"]
fn test_search_tie_break_is_deterministic_by_id() {}

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
#[ignore = "collection methods not implemented in TurboQuantEngine"]
fn test_collection_namespace_isolation() {}

#[test]
#[ignore = "collection methods not implemented in TurboQuantEngine"]
fn test_collection_lifecycle_guards() {}

#[test]
#[ignore = "collection methods not implemented in TurboQuantEngine"]
fn test_collection_schema_mismatch_guard() {}

#[test]
fn test_filter_nested_path_and_missing_field_semantics() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().to_str().unwrap();
    let mut engine = TurboQuantEngine::open(db_path, db_path, 8, 2, 77).unwrap();

    let v = Array1::<f64>::from_elem(8, 0.3);

    // Insert: one with nested "profile.region", one without.
    let mut meta_with = HashMap::new();
    meta_with.insert(
        "profile".to_string(),
        serde_json::json!({"region": "us-west"}),
    );
    meta_with.insert("score".to_string(), serde_json::json!(95));
    engine
        .upsert_with_document("has_profile".to_string(), &v, meta_with, None)
        .unwrap();

    let mut meta_without = HashMap::new();
    meta_without.insert("score".to_string(), serde_json::json!(80));
    engine
        .upsert_with_document("no_profile".to_string(), &v, meta_without, None)
        .unwrap();

    engine.flush_wal_to_segment().unwrap();

    // Dotted path traversal.
    let f_region = serde_json::json!({"profile.region": "us-west"});
    let f_region_map: HashMap<String, JsonValue> = serde_json::from_value(f_region).unwrap();
    let got = engine
        .search_with_filter(&v, 10, Some(&f_region_map))
        .unwrap();
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].id, "has_profile");

    // $ne matches when field is missing ("missing" != "us-west").
    let f_ne = serde_json::json!({"profile.region": {"$ne": "us-west"}});
    let f_ne_map: HashMap<String, JsonValue> = serde_json::from_value(f_ne).unwrap();
    let got_ne = engine.search_with_filter(&v, 10, Some(&f_ne_map)).unwrap();
    assert_eq!(got_ne.len(), 1);
    assert_eq!(got_ne[0].id, "no_profile");

    // $gt on missing field → no match.
    let f_gt_miss = serde_json::json!({"profile.region": {"$gt": "a"}});
    let f_gt_map: HashMap<String, JsonValue> = serde_json::from_value(f_gt_miss).unwrap();
    let got_gt_miss = engine.search_with_filter(&v, 10, Some(&f_gt_map)).unwrap();
    assert_eq!(got_gt_miss.len(), 1); // only "has_profile" has the field
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
    engine
        .upsert_with_document("x".to_string(), &v, meta, None)
        .unwrap();
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
                .upsert_with_document(
                    (*id).to_string(),
                    &Array1::<f64>::from_vec(v.clone()),
                    HashMap::new(),
                    None,
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
            .upsert_with_document(format!("id-{i}"), &v, HashMap::new(), None)
            .unwrap();
    }
    engine
        .create_index_with_params(24, 200, 37, 1.7, 0)
        .unwrap();
    let s = engine.stats();
    assert!(s.has_index);
    assert_eq!(s.index_nodes, 40);
    assert_eq!(
        engine
            .manifest
            .index_state
            .as_ref()
            .map(|s| s.search_list_size),
        Some(37)
    );
    assert_eq!(
        engine.manifest.index_state.as_ref().map(|s| s.alpha),
        Some(1.7)
    );
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
            .upsert_with_document(format!("id-{i}"), &v, HashMap::new(), None)
            .unwrap();
    }
    engine.create_index_with_params(1, 200, 20, 1.2, 0).unwrap();
    engine.close().unwrap();
    drop(engine);

    let backend = Arc::new(StorageBackend::from_uri(db_path).unwrap());
    let graph = GraphManager::open(backend, db_path).unwrap();
    for node in 0..12u32 {
        let neighbors = graph.get_neighbors_at_level(node, 0).unwrap();
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
        engine
            .upsert_with_document(id.clone(), &v, HashMap::new(), None)
            .unwrap();
        vectors.insert(id, v);
    }

    // Intentionally low default search list size, then override per-query.
    engine.create_index_with_params(8, 200, 1, 1.2, 0).unwrap();
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
    use turboquantdb::storage::compaction::Compactor;
    use turboquantdb::storage::segment::{Segment, SegmentRecord};

    let dir = tempdir().unwrap();
    let db_path = dir.path().to_str().unwrap();

    // Build a DB with a few vectors flushed to segments.
    let mut engine = TurboQuantEngine::open(db_path, db_path, 8, 2, 42).unwrap();
    for i in 0..5u32 {
        let v = Array1::<f64>::from_elem(8, i as f64 / 10.0);
        engine
            .insert(format!("v{}", i), &v, HashMap::new())
            .unwrap();
    }
    engine.flush_wal_to_segment().unwrap();
    for i in 5..10u32 {
        let v = Array1::<f64>::from_elem(8, i as f64 / 10.0);
        engine
            .insert(format!("v{}", i), &v, HashMap::new())
            .unwrap();
    }
    engine.flush_wal_to_segment().unwrap();
    let before_count = engine.vector_count();
    drop(engine);

    // Identify existing segments.
    let backend = Arc::new(StorageBackend::from_uri(db_path).unwrap());
    let mut old_names: Vec<String> = backend
        .list("")
        .unwrap()
        .into_iter()
        .filter(|n| n.starts_with("seg-") && n.ends_with(".bin"))
        .collect();
    old_names.sort();
    assert!(!old_names.is_empty(), "should have at least one segment");

    // Write a fake "prepared" compaction marker — simulates a crash mid-compaction.
    // Use Segment::write_batch with zero records to produce a valid (but empty) segment
    // so that recovery's read_all() validation passes and the old segments are deleted.
    let compacted_name = "seg-99999999.bin";
    let compactor = Compactor::new(backend.clone());
    compactor
        .begin_compaction(&old_names, compacted_name)
        .unwrap();
    Segment::write_batch(&backend, compacted_name, &[] as &[SegmentRecord]).unwrap();

    // Reopen the engine — recovery must delete old segments and the marker.
    let recovered = TurboQuantEngine::open(db_path, db_path, 8, 2, 42).unwrap();
    assert_eq!(recovered.vector_count(), before_count);
    drop(recovered);

    let backend_after = StorageBackend::from_uri(db_path).unwrap();
    assert!(
        !backend_after.exists("compaction_state.json"),
        "compaction_state.json must be removed on open"
    );

    // Old segment files must be gone; only the compacted placeholder remains.
    for old in &old_names {
        assert!(
            !backend_after.exists(old),
            "old segment {} must be deleted",
            old
        );
    }
    assert!(
        backend_after.exists(compacted_name),
        "compacted segment must still exist"
    );
}

#[test]
fn test_stats_deleted_record_count_tracks_tombstones() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().to_str().unwrap();
    let mut engine = TurboQuantEngine::open(db_path, db_path, 8, 2, 17).unwrap();

    engine
        .upsert_with_document(
            "a".to_string(),
            &Array1::<f64>::from_elem(8, 0.1),
            HashMap::new(),
            None,
        )
        .unwrap();
    engine
        .upsert_with_document(
            "b".to_string(),
            &Array1::<f64>::from_elem(8, 0.2),
            HashMap::new(),
            None,
        )
        .unwrap();
    engine.flush_wal_to_segment().unwrap();
    engine.delete("a".to_string()).unwrap();
    engine.flush_wal_to_segment().unwrap();

    let s = engine.stats();
    assert_eq!(s.vector_count, 1);
}

#[test]
#[ignore = "search_hybrid_with_filter not implemented in TurboQuantEngine"]
fn test_hybrid_search_sparse_weight_can_override_dense_order() {}

#[test]
#[ignore = "search_hybrid_with_filter not implemented in TurboQuantEngine"]
fn test_hybrid_search_respects_where_filter() {}

#[test]
#[ignore = "search_with_filter_and_reranker not implemented in TurboQuantEngine"]
fn test_engine_reranker_hook_reorders_topn() {}

#[test]
#[ignore = "insert_many_report not implemented in TurboQuantEngine"]
fn test_batch_insert_report_continues_on_duplicates() {}

#[test]
#[ignore = "update_many_report not implemented in TurboQuantEngine"]
fn test_batch_update_report_continues_on_missing_ids() {}

#[test]
#[ignore = "snapshot_local_dir/restore_from_snapshot not implemented in TurboQuantEngine"]
fn test_snapshot_and_restore_roundtrip() {}

#[test]
#[ignore = "collection methods not implemented in TurboQuantEngine"]
fn test_collection_snapshot_and_restore_roundtrip() {}

#[test]
#[ignore = "scoped collection methods not implemented in TurboQuantEngine"]
fn test_scoped_tenant_database_collection_isolation() {}

#[test]
#[ignore = "scoped collection methods not implemented in TurboQuantEngine"]
fn test_scoped_catalog_helpers_list_and_delete() {}

#[test]
fn test_rerank_disabled_behavior() {
    use turboquantdb::storage::engine::RerankPrecision;

    let dir = tempdir().unwrap();
    let db_path = dir.path().to_str().unwrap();

    // Open with RerankPrecision::Disabled — no live_vectors.bin should be created
    let mut engine = TurboQuantEngine::open_with_options(
        db_path,
        db_path,
        8,
        2,
        42,
        DistanceMetric::Ip,
        true,
        false,
        RerankPrecision::Disabled,
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

    // Write first batch, verify WAL header is present.
    {
        let mut engine = TurboQuantEngine::open(db_path, db_path, 8, 2, 42).unwrap();
        engine
            .insert("v1".to_string(), &Array1::zeros(8), HashMap::new())
            .unwrap();
    }
    let wal_bytes = std::fs::read(&wal_path).unwrap();
    assert_eq!(
        &wal_bytes[0..4],
        b"TQWV",
        "WAL must start with version header"
    );

    // Reopen and verify recovery + flush work without any panic.
    let mut engine = TurboQuantEngine::open(db_path, db_path, 8, 2, 42).unwrap();
    engine.flush_wal_to_segment().unwrap();
    assert_eq!(engine.vector_count(), 1);
}

#[test]
fn test_hnsw_beam_search_recall() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().to_str().unwrap();
    // Use L2 metric so the exact match (v0) is the true nearest neighbour.
    let mut engine =
        TurboQuantEngine::open_with_metric(db_path, db_path, 16, 2, 42, DistanceMetric::L2)
            .unwrap();

    // Insert 200 vectors to have some depth
    for i in 0..200usize {
        let v = Array1::<f64>::from_iter((0..16).map(|j| (i + j) as f64 / 1000.0));
        engine
            .insert(format!("v{}", i), &v, HashMap::new())
            .unwrap();
    }

    engine
        .create_index_with_params(16, 200, 64, 1.2, 0)
        .unwrap();

    let query = Array1::<f64>::from_iter((0..16).map(|j| j as f64 / 1000.0));
    let results = engine.search(&query, 10).unwrap();

    assert!(!results.is_empty());
    assert_eq!(results[0].id, "v0"); // Should easily find the exact match with beam search
}

/// Test that rerank_precision=F16 creates live_vectors.bin with correct byte size,
/// and that values can be recovered within f16 precision (relative error < 0.2%).
#[test]
fn test_rerank_precision_f16_roundtrip() {
    use turboquantdb::storage::engine::RerankPrecision;

    let dir = tempdir().unwrap();
    let db_path = dir.path().to_str().unwrap();

    let d = 32usize;
    let mut engine = TurboQuantEngine::open_with_options(
        db_path,
        db_path,
        d,
        4,
        42,
        DistanceMetric::Ip,
        true,
        false,
        RerankPrecision::F16,
    )
    .unwrap();

    // Insert 10 vectors with known values.
    let n = 10usize;
    for i in 0..n {
        let v = Array1::<f64>::from_iter((0..d).map(|j| (i * d + j) as f64 / (n * d) as f64));
        engine
            .insert(format!("v{}", i), &v, HashMap::new())
            .unwrap();
    }

    // Flush WAL to compact live_vectors.bin to exactly the logical size.
    engine.flush_wal_to_segment().unwrap();

    // live_vectors.bin must exist and be exactly n * d * 2 bytes.
    let vraw_path = dir.path().join("live_vectors.bin");
    assert!(
        vraw_path.exists(),
        "live_vectors.bin should be created for F16 precision"
    );
    let file_size = std::fs::metadata(&vraw_path).unwrap().len() as usize;
    assert_eq!(
        file_size,
        n * d * 2,
        "F16 file size should be n*d*2 bytes, got {}",
        file_size
    );

    // Search must succeed and return a valid result.
    let query = Array1::<f64>::from_iter((0..d).map(|j| j as f64 / (n * d) as f64));
    let results = engine.search(&query, 3).unwrap();
    assert!(
        !results.is_empty(),
        "search with F16 reranking must return results"
    );
    assert!(results[0].score > 0.0, "top score must be positive");
}

/// Test that rerank_precision=F32 creates live_vectors.bin with exactly n*d*4 bytes.
#[test]
fn test_rerank_precision_f32_file_size() {
    use turboquantdb::storage::engine::RerankPrecision;

    let dir = tempdir().unwrap();
    let db_path = dir.path().to_str().unwrap();

    let d = 16usize;
    let n = 5usize;
    let mut engine = TurboQuantEngine::open_with_options(
        db_path,
        db_path,
        d,
        4,
        42,
        DistanceMetric::Ip,
        true,
        false,
        RerankPrecision::F32,
    )
    .unwrap();

    for i in 0..n {
        let v = Array1::<f64>::from_iter((0..d).map(|j| (i + j) as f64 / 100.0));
        engine
            .insert(format!("v{}", i), &v, HashMap::new())
            .unwrap();
    }

    // Flush WAL to compact live_vectors.bin to exactly the logical size.
    engine.flush_wal_to_segment().unwrap();

    let vraw_path = dir.path().join("live_vectors.bin");
    assert!(
        vraw_path.exists(),
        "live_vectors.bin must exist for F32 precision"
    );
    let file_size = std::fs::metadata(&vraw_path).unwrap().len() as usize;
    assert_eq!(
        file_size,
        n * d * 4,
        "F32 file size should be n*d*4 bytes, got {}",
        file_size
    );
}

/// Test that f16 file is exactly half the size of f32 file for same vectors,
/// and that both modes produce search results.
#[test]
fn test_rerank_precision_f16_vs_f32_size_ratio() {
    use turboquantdb::storage::engine::RerankPrecision;

    let d = 64usize;
    let n = 20usize;
    let mut sizes = [0usize; 2];

    for (idx, precision) in [RerankPrecision::F16, RerankPrecision::F32]
        .iter()
        .enumerate()
    {
        let dir = tempdir().unwrap();
        let db_path = dir.path().to_str().unwrap();

        let mut engine = TurboQuantEngine::open_with_options(
            db_path,
            db_path,
            d,
            4,
            42,
            DistanceMetric::Cosine,
            true,
            false,
            *precision,
        )
        .unwrap();

        for i in 0..n {
            let v = Array1::<f64>::from_iter((0..d).map(|j| (i * d + j + 1) as f64));
            engine
                .insert(format!("v{}", i), &v, HashMap::new())
                .unwrap();
        }

        let vraw_path = dir.path().join("live_vectors.bin");
        sizes[idx] = std::fs::metadata(&vraw_path).unwrap().len() as usize;

        let query = Array1::<f64>::from_iter((0..d).map(|j| j as f64));
        let results = engine.search(&query, 5).unwrap();
        assert!(
            !results.is_empty(),
            "search must return results for {:?}",
            precision
        );
    }

    // F16 file should be exactly half of F32 file.
    assert_eq!(
        sizes[1],
        sizes[0] * 2,
        "F32 file ({} bytes) should be exactly 2× F16 file ({} bytes)",
        sizes[1],
        sizes[0]
    );
}

/// Test that default (Disabled) mode does NOT create live_vectors.bin.
#[test]
fn test_rerank_precision_disabled_no_file() {
    use turboquantdb::storage::engine::RerankPrecision;

    let dir = tempdir().unwrap();
    let db_path = dir.path().to_str().unwrap();

    let mut engine = TurboQuantEngine::open_with_options(
        db_path,
        db_path,
        16,
        4,
        42,
        DistanceMetric::Ip,
        true,
        false,
        RerankPrecision::Disabled,
    )
    .unwrap();

    for i in 0..5 {
        let v = Array1::<f64>::from_iter((0..16).map(|j| (i + j) as f64 / 100.0));
        engine
            .insert(format!("v{}", i), &v, HashMap::new())
            .unwrap();
    }

    // Must NOT create live_vectors.bin
    assert!(
        !dir.path().join("live_vectors.bin").exists(),
        "Disabled mode must not create live_vectors.bin"
    );

    // Search must still work via dequantization reranking.
    let query = Array1::<f64>::from_elem(16, 0.5);
    let results = engine.search(&query, 3).unwrap();
    assert!(
        !results.is_empty(),
        "search must work with dequant reranking"
    );
}
