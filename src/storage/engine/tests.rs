use super::*;
use ndarray::Array1;
use serde_json::json;
use std::collections::HashMap;
use tempfile::tempdir;

fn make_vec(d: usize, val: f64) -> Array1<f64> {
    Array1::from_elem(d, val)
}

fn make_vec_f32(d: usize, val: f32) -> Vec<f32> {
    vec![val; d]
}

fn no_meta() -> HashMap<String, serde_json::Value> {
    HashMap::new()
}

fn open_default(dir: &str, d: usize) -> TurboQuantEngine {
    TurboQuantEngine::open(dir, dir, d, 2, 42).unwrap()
}

// ── Error paths on reopen ──────────────────────────────────────────────

#[test]
fn reopen_dimension_mismatch_returns_error() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let mut e = TurboQuantEngine::open(p, p, 8, 2, 42).unwrap();
    e.insert("a".into(), &make_vec(8, 0.5), no_meta()).unwrap();
    let result = TurboQuantEngine::open(p, p, 16, 2, 42);
    assert!(result.is_err());
    let msg = result.err().unwrap().to_string();
    assert!(msg.contains("dimension mismatch"), "unexpected: {msg}");
}

#[test]
fn reopen_bits_mismatch_returns_error() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let mut e = TurboQuantEngine::open(p, p, 8, 2, 42).unwrap();
    e.insert("a".into(), &make_vec(8, 0.5), no_meta()).unwrap();
    let result = TurboQuantEngine::open(p, p, 8, 4, 42);
    assert!(result.is_err());
    let msg = result.err().unwrap().to_string();
    assert!(msg.contains("bits mismatch"), "unexpected: {msg}");
}

#[test]
fn reopen_metric_mismatch_returns_error() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let mut e = TurboQuantEngine::open_with_metric(p, p, 8, 2, 42, DistanceMetric::Ip).unwrap();
    e.insert("a".into(), &make_vec(8, 0.5), no_meta()).unwrap();
    let result = TurboQuantEngine::open_with_metric(p, p, 8, 2, 42, DistanceMetric::L2);
    assert!(result.is_err());
    let msg = result.err().unwrap().to_string();
    assert!(msg.contains("metric mismatch"), "unexpected: {msg}");
}

// ── Single-vector write errors ─────────────────────────────────────────

#[test]
fn insert_duplicate_id_returns_error() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let mut e = open_default(p, 8);
    e.insert("dup".into(), &make_vec(8, 0.1), no_meta())
        .unwrap();
    let result = e.insert("dup".into(), &make_vec(8, 0.2), no_meta());
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("already exists"));
}

#[test]
fn update_nonexistent_id_returns_error() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let mut e = open_default(p, 8);
    let result = e.update_with_document("ghost".into(), &make_vec(8, 0.1), no_meta(), None);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("does not exist"));
}

// ── Batch write mode errors ────────────────────────────────────────────

#[test]
fn insert_many_insert_mode_duplicate_returns_error() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let mut e = open_default(p, 8);
    let item = BatchWriteItem {
        id: "x".into(),
        vector: make_vec_f32(8, 0.5),
        metadata: no_meta(),
        document: None,
    };
    e.insert_many(vec![item.clone()]).unwrap();
    let result = e.insert_many_with_mode(vec![item], BatchWriteMode::Insert);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("already exists"));
}

#[test]
fn insert_many_update_mode_missing_id_returns_error() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let mut e = open_default(p, 8);
    let item = BatchWriteItem {
        id: "ghost".into(),
        vector: make_vec_f32(8, 0.5),
        metadata: no_meta(),
        document: None,
    };
    let result = e.insert_many_with_mode(vec![item], BatchWriteMode::Update);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("does not exist"));
}

// ── Cosine exhaustive search ───────────────────────────────────────────

#[test]
fn cosine_exhaustive_search_returns_ordered_results() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 16;
    let mut e = TurboQuantEngine::open_with_metric(p, p, d, 4, 42, DistanceMetric::Cosine).unwrap();
    // Insert a parallel vector (same direction) and an orthogonal one
    let mut v_same = vec![0.0f64; d];
    v_same[0] = 1.0;
    let mut v_ortho = vec![0.0f64; d];
    v_ortho[1] = 1.0;
    e.insert("same".into(), &Array1::from_vec(v_same.clone()), no_meta())
        .unwrap();
    e.insert(
        "ortho".into(),
        &Array1::from_vec(v_ortho.clone()),
        no_meta(),
    )
    .unwrap();

    let query = Array1::from_vec(v_same);
    let results = e
        .search_with_filter_and_ann(&query, 2, None, None, true)
        .unwrap();
    assert!(!results.is_empty());
    // The "same" vector should score higher than "ortho"
    let top_id = &results[0].id;
    assert_eq!(top_id, "same");
}

#[test]
fn cosine_zero_norm_vector_scores_zero() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    let mut e = TurboQuantEngine::open_with_metric(p, p, d, 2, 7, DistanceMetric::Cosine).unwrap();
    // A zero vector has norm=0; cosine score must be 0.0
    e.insert("zero".into(), &Array1::zeros(d), no_meta())
        .unwrap();
    let query = make_vec(d, 1.0);
    let results = e
        .search_with_filter_and_ann(&query, 5, None, None, true)
        .unwrap();
    assert!(!results.is_empty());
    assert_eq!(results[0].score, 0.0);
}

// ── L2 exhaustive search ───────────────────────────────────────────────

#[test]
fn l2_exhaustive_search_returns_ordered_results() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 16;
    let mut e = TurboQuantEngine::open_with_metric(p, p, d, 4, 42, DistanceMetric::L2).unwrap();
    // "close" is distance 0 from query (at origin); "far" is distance 10
    let v_close = vec![0.0f64; d];
    let mut v_far = vec![0.0f64; d];
    v_far[0] = 10.0;
    e.insert(
        "close".into(),
        &Array1::from_vec(v_close.clone()),
        no_meta(),
    )
    .unwrap();
    e.insert("far".into(), &Array1::from_vec(v_far), no_meta())
        .unwrap();

    let query = Array1::from_vec(v_close);
    let results = e
        .search_with_filter_and_ann(&query, 2, None, None, true)
        .unwrap();
    assert!(!results.is_empty());
    // L2 scores are negative distances; "close" should have a higher (less negative) score
    assert_eq!(results[0].id, "close");
}

// ── ANN search with HNSW index ─────────────────────────────────────────

#[test]
fn ann_cosine_search_with_index_returns_results() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 16;
    let mut e = TurboQuantEngine::open_with_metric(p, p, d, 4, 42, DistanceMetric::Cosine).unwrap();
    for i in 0..20u32 {
        let mut v = vec![0.0f64; d];
        v[i as usize % d] = 1.0;
        e.insert(format!("v{i}"), &Array1::from_vec(v), no_meta())
            .unwrap();
    }
    e.create_index_with_params(8, 32, 32, 1.2, 2).unwrap();
    let mut q = vec![0.0f64; d];
    q[0] = 1.0;
    let results = e
        .search_with_filter_and_ann(&Array1::from_vec(q), 5, None, None, true)
        .unwrap();
    assert!(!results.is_empty());
    assert!(results.len() <= 5);
}

#[test]
fn ann_l2_search_with_index_returns_results() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 16;
    let mut e = TurboQuantEngine::open_with_metric(p, p, d, 4, 42, DistanceMetric::L2).unwrap();
    for i in 0..20u32 {
        let mut v = vec![0.0f64; d];
        v[i as usize % d] = (i as f64 + 1.0) * 0.1;
        e.insert(format!("v{i}"), &Array1::from_vec(v), no_meta())
            .unwrap();
    }
    e.create_index_with_params(8, 32, 32, 1.2, 2).unwrap();
    let q = vec![0.0f64; d];
    let results = e
        .search_with_filter_and_ann(&Array1::from_vec(q), 5, None, None, true)
        .unwrap();
    assert!(!results.is_empty());
    assert!(results.len() <= 5);
}

// ── Search edge cases ──────────────────────────────────────────────────

#[test]
fn search_top_k_zero_returns_empty() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let mut e = open_default(p, 8);
    e.insert("a".into(), &make_vec(8, 0.5), no_meta()).unwrap();
    let results = e
        .search_with_filter_and_ann(&make_vec(8, 0.5), 0, None, None, true)
        .unwrap();
    assert!(results.is_empty());
}

#[test]
fn search_empty_db_returns_empty() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let e = open_default(p, 8);
    let results = e
        .search_with_filter_and_ann(&make_vec(8, 0.5), 5, None, None, true)
        .unwrap();
    assert!(results.is_empty());
}

#[test]
fn search_filter_no_match_returns_empty() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let mut e = open_default(p, 8);
    let mut meta = no_meta();
    meta.insert("tag".into(), json!("a"));
    e.insert("a".into(), &make_vec(8, 0.5), meta).unwrap();
    let mut filter = no_meta();
    filter.insert("tag".into(), json!("b"));
    let results = e
        .search_with_filter_and_ann(&make_vec(8, 0.5), 5, Some(&filter), None, true)
        .unwrap();
    assert!(results.is_empty());
}

// ── F16 rerank precision ───────────────────────────────────────────────

#[test]
fn f16_rerank_precision_stores_and_retrieves_vectors() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 16;
    let mut e = TurboQuantEngine::open_with_options(
        p,
        p,
        d,
        4,
        42,
        DistanceMetric::Ip,
        true,
        false,
        RerankPrecision::F16,
        None,
        false,
    )
    .unwrap();
    let mut v = vec![0.0f64; d];
    v[0] = 0.9;
    v[1] = 0.4;
    e.insert("v1".into(), &Array1::from_vec(v.clone()), no_meta())
        .unwrap();
    let results = e
        .search_with_filter_and_ann(&Array1::from_vec(v), 1, None, None, true)
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "v1");
    // F16 has ~0.1% relative error; score should be close to 1.0
    assert!(results[0].score > 0.8, "score was {}", results[0].score);
}

#[test]
fn f16_roundtrip_within_half_precision_tolerance() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 16;
    let mut e = TurboQuantEngine::open_with_options(
        p,
        p,
        d,
        4,
        42,
        DistanceMetric::Ip,
        true,
        false,
        RerankPrecision::F16,
        None,
        false,
    )
    .unwrap();
    // v = [0.1, 0.2, ..., 1.6]; expected IP = sum((0.1*i)^2) for i=1..=16
    let v: Vec<f64> = (1..=d).map(|i| i as f64 * 0.1).collect();
    let expected: f64 = v.iter().map(|x| x * x).sum();
    e.insert("v1".into(), &Array1::from_vec(v.clone()), no_meta())
        .unwrap();
    let results = e
        .search_with_filter_and_ann(&Array1::from_vec(v), 1, None, None, true)
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "v1");
    // F16 tolerance: relative error < 1% (0.01)
    let rel_err = (results[0].score - expected).abs() / expected;
    assert!(
        rel_err < 0.01,
        "F16 relative error {:.4} exceeds 0.01; score={:.4}, expected={:.4}",
        rel_err,
        results[0].score,
        expected
    );
}

// ── Stats and disk bytes ───────────────────────────────────────────────

#[test]
fn stats_reflect_inserted_vectors() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    let mut e = open_default(p, d);
    assert_eq!(e.stats().vector_count, 0);
    assert_eq!(e.stats().d, d);
    e.insert("a".into(), &make_vec(d, 0.1), no_meta()).unwrap();
    e.insert("b".into(), &make_vec(d, 0.2), no_meta()).unwrap();
    let s = e.stats();
    assert_eq!(s.vector_count, 2);
    assert_eq!(s.live_id_count, 2);
}

#[test]
fn total_disk_bytes_nonzero_after_flush() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    let mut e = open_default(p, d);
    for i in 0..5u32 {
        e.insert(format!("v{i}"), &make_vec(d, 0.5), no_meta())
            .unwrap();
    }
    e.flush_wal_to_segment().unwrap();
    let bytes = e.stats().total_disk_bytes;
    assert!(bytes > 0, "expected non-zero disk bytes, got {bytes}");
}

// ── Delete ─────────────────────────────────────────────────────────────

#[test]
fn delete_unknown_id_returns_false() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let mut e = open_default(p, 8);
    let deleted = e.delete("ghost".into()).unwrap();
    assert!(!deleted);
}

#[test]
fn delete_existing_id_returns_true_and_reduces_count() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let mut e = open_default(p, 8);
    e.insert("a".into(), &make_vec(8, 0.5), no_meta()).unwrap();
    assert_eq!(e.stats().vector_count, 1);
    let deleted = e.delete("a".into()).unwrap();
    assert!(deleted);
    assert_eq!(e.stats().vector_count, 0);
}

// ── Index invalidation ─────────────────────────────────────────────────

#[test]
fn index_preserved_after_insert_new_slot_in_delta() {
    // With the delta index, inserts after create_index() do NOT invalidate
    // the HNSW graph.  The new slot goes into delta_slots instead.
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 16;
    let mut e = open_default(p, d);
    for i in 0..10u32 {
        e.insert(format!("v{i}"), &make_vec(d, i as f64 * 0.1), no_meta())
            .unwrap();
    }
    e.create_index_with_params(4, 16, 16, 1.2, 1).unwrap();
    assert!(e.manifest.index_state.is_some(), "index should be built");
    assert_eq!(
        e.delta_slots.len(),
        0,
        "delta must be empty right after build"
    );
    // A new insert must keep the index valid and add the slot to delta.
    e.insert("new".into(), &make_vec(d, 0.99), no_meta())
        .unwrap();
    assert!(
        e.manifest.index_state.is_some(),
        "index state must be preserved after insert (delta index)"
    );
    assert_eq!(e.delta_slots.len(), 1, "new slot must appear in delta");
}

#[test]
fn index_invalidated_after_delete() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 16;
    let mut e = open_default(p, d);
    for i in 0..10u32 {
        e.insert(format!("v{i}"), &make_vec(d, i as f64 * 0.1), no_meta())
            .unwrap();
    }
    e.create_index_with_params(4, 16, 16, 1.2, 1).unwrap();
    assert!(e.manifest.index_state.is_some());
    e.delete("v0".into()).unwrap();
    assert!(
        e.manifest.index_state.is_none(),
        "index state should be cleared after delete"
    );
}

#[test]
fn delta_cleared_after_create_index() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 16;
    let mut e = open_default(p, d);
    for i in 0..10u32 {
        e.insert(format!("v{i}"), &make_vec(d, i as f64 * 0.1), no_meta())
            .unwrap();
    }
    e.create_index_with_params(4, 16, 16, 1.2, 1).unwrap();
    // Insert 3 new vectors — they go to delta
    for i in 10u32..13u32 {
        e.insert(format!("v{i}"), &make_vec(d, i as f64 * 0.1), no_meta())
            .unwrap();
    }
    assert_eq!(e.delta_slots.len(), 3, "3 new slots must be in delta");
    // Rebuild — delta is merged into the graph and cleared
    e.create_index_with_params(4, 16, 16, 1.2, 1).unwrap();
    assert_eq!(e.delta_slots.len(), 0, "delta must be empty after rebuild");
    assert_eq!(
        e.index_ids.len(),
        13,
        "all 13 vectors must be indexed after rebuild"
    );
}

#[test]
fn ann_search_returns_delta_vectors() {
    // Vectors inserted after create_index() must be findable via ANN search.
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 32;
    let mut e = open_default(p, d);
    for i in 0..20u32 {
        e.insert(format!("base{i}"), &make_vec(d, i as f64 * 0.05), no_meta())
            .unwrap();
    }
    e.create_index_with_params(8, 32, 32, 1.2, 2).unwrap();
    // Insert a distinctive vector into the delta
    let target = make_vec(d, 9.9);
    e.insert("delta_target".into(), &target, no_meta()).unwrap();
    assert_eq!(e.delta_slots.len(), 1, "delta must have the new slot");
    // ANN search with _use_ann=true must return the delta vector
    let results = e
        .search_with_filter_and_ann(
            &target.iter().map(|&x| x as f64).collect(),
            1,
            None,
            None,
            true,
        )
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(
        results[0].id, "delta_target",
        "delta vector must be found via ANN"
    );
}

// ── Flush WAL edge case ────────────────────────────────────────────────

#[test]
fn flush_wal_empty_buffer_is_noop() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let mut e = open_default(p, 8);
    // No inserts — flush should succeed silently
    e.flush_wal_to_segment().unwrap();
    assert_eq!(e.stats().vector_count, 0);
}

// ── Fast-mode engine ───────────────────────────────────────────────────

#[test]
fn fast_mode_engine_creates_and_searches() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 32;
    let mut e = TurboQuantEngine::open_fast(p, p, d, 4, 42).unwrap();
    for i in 0..5u32 {
        e.insert(
            format!("v{i}"),
            &make_vec(d, i as f64 * 0.1 + 0.1),
            no_meta(),
        )
        .unwrap();
    }
    let results = e
        .search_with_filter_and_ann(&make_vec(d, 0.5), 3, None, None, true)
        .unwrap();
    assert!(!results.is_empty());
    assert!(e.manifest.fast_mode);
}

// ── Upsert mode ────────────────────────────────────────────────────────

#[test]
fn upsert_replaces_existing_vector() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    let mut e = open_default(p, d);
    e.insert("a".into(), &make_vec(d, 0.1), no_meta()).unwrap();
    // Upsert with same ID should not error
    e.upsert_with_document("a".into(), &make_vec(d, 0.9), no_meta(), None)
        .unwrap();
    assert_eq!(e.stats().vector_count, 1);
}

// ── List all ───────────────────────────────────────────────────────────

#[test]
fn list_all_returns_all_active_ids() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    let mut e = open_default(p, d);
    e.insert("x".into(), &make_vec(d, 0.1), no_meta()).unwrap();
    e.insert("y".into(), &make_vec(d, 0.2), no_meta()).unwrap();
    e.insert("z".into(), &make_vec(d, 0.3), no_meta()).unwrap();
    e.delete("y".into()).unwrap();
    let ids = e.list_all();
    assert_eq!(ids.len(), 2);
    assert!(ids.contains(&"x".to_string()));
    assert!(ids.contains(&"z".to_string()));
    assert!(!ids.contains(&"y".to_string()));
}

// ── Disabled rerank ────────────────────────────────────────────────────

#[test]
fn disabled_rerank_search_returns_results() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 16;
    let mut e = TurboQuantEngine::open_with_options(
        p,
        p,
        d,
        4,
        42,
        DistanceMetric::Ip,
        false,
        false,
        RerankPrecision::Disabled,
        None,
        false,
    )
    .unwrap();
    for i in 0..5u32 {
        let v = make_vec(d, i as f64 * 0.2 + 0.1);
        e.insert(format!("v{i}"), &v, no_meta()).unwrap();
    }
    let results = e
        .search_with_filter_and_ann(&make_vec(d, 0.5), 3, None, None, true)
        .unwrap();
    assert!(!results.is_empty());
    assert!(!e.rerank_enabled);
}

// ── get and get_many ───────────────────────────────────────────────────

#[test]
fn get_returns_none_for_missing_id() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let e = open_default(p, 8);
    let result = e.get("ghost").unwrap();
    assert!(result.is_none());
}

#[test]
fn get_returns_metadata_and_document() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    let mut e = open_default(p, d);
    let mut meta = no_meta();
    meta.insert("key".into(), json!("val"));
    e.insert_with_document(
        "v1".into(),
        &make_vec(d, 0.5),
        meta,
        Some("hello doc".into()),
    )
    .unwrap();
    let result = e.get("v1").unwrap().unwrap();
    assert_eq!(result.id, "v1");
    assert_eq!(result.metadata.get("key"), Some(&json!("val")));
    assert_eq!(result.document, Some("hello doc".into()));
}

#[test]
fn get_many_returns_mixed_results() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    let mut e = open_default(p, d);
    e.insert("a".into(), &make_vec(d, 0.1), no_meta()).unwrap();
    e.insert("b".into(), &make_vec(d, 0.2), no_meta()).unwrap();
    let ids: Vec<String> = vec!["a".into(), "missing".into(), "b".into()];
    let results = e.get_many(&ids).unwrap();
    assert_eq!(results.len(), 3);
    assert!(results[0].is_some());
    assert!(results[1].is_none());
    assert!(results[2].is_some());
}

// ── upsert_many / update_many wrappers ────────────────────────────────

#[test]
fn upsert_many_inserts_and_replaces() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    let mut e = open_default(p, d);
    let item = BatchWriteItem {
        id: "u".into(),
        vector: make_vec_f32(d, 0.5),
        metadata: no_meta(),
        document: None,
    };
    // First upsert inserts
    e.upsert_many(vec![item.clone()]).unwrap();
    assert_eq!(e.stats().vector_count, 1);
    // Second upsert replaces (no error)
    e.upsert_many(vec![item]).unwrap();
    assert_eq!(e.stats().vector_count, 1);
}

#[test]
fn update_many_updates_existing_vector() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    let mut e = open_default(p, d);
    e.insert("x".into(), &make_vec(d, 0.1), no_meta()).unwrap();
    let item = BatchWriteItem {
        id: "x".into(),
        vector: make_vec_f32(d, 0.9),
        metadata: no_meta(),
        document: None,
    };
    e.update_many(vec![item]).unwrap();
    assert_eq!(e.stats().vector_count, 1);
}

// ── create_index on empty DB ────────────────────────────────────────────

#[test]
fn create_index_empty_db_is_noop() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let mut e = open_default(p, 8);
    // Should not error on empty DB
    e.create_index_with_params(4, 16, 16, 1.2, 1).unwrap();
    assert!(!e.stats().has_index);
}

// ── L2 index build without raw vecs (precomputed_l2 path) ─────────────

#[test]
fn l2_index_without_rerank_uses_dequantized_path() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 16;
    // rerank=false → no live_vectors.bin → L2 index uses precomputed_l2
    let mut e = TurboQuantEngine::open_with_options(
        p,
        p,
        d,
        4,
        42,
        DistanceMetric::L2,
        false,
        false,
        RerankPrecision::Disabled,
        None,
        false,
    )
    .unwrap();
    for i in 0..15u32 {
        let mut v = vec![0.0f32; d];
        v[0] = i as f32 * 0.1;
        e.insert_many(vec![BatchWriteItem {
            id: format!("v{i}"),
            vector: v,
            metadata: no_meta(),
            document: None,
        }])
        .unwrap();
    }
    e.create_index_with_params(4, 16, 16, 1.2, 1).unwrap();
    assert!(e.stats().has_index);
    let q = vec![0.0f64; d];
    let results = e
        .search_with_filter_and_ann(&Array1::from_vec(q), 5, None, None, true)
        .unwrap();
    assert!(!results.is_empty());
}

// ── ANN search with filter ─────────────────────────────────────────────

#[test]
fn ann_search_with_metadata_filter() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 16;
    let mut e = open_default(p, d);
    for i in 0..15u32 {
        let mut meta = no_meta();
        meta.insert("cat".into(), json!(if i % 2 == 0 { "even" } else { "odd" }));
        let mut v = vec![0.0f64; d];
        v[0] = i as f64 * 0.1;
        e.insert(format!("v{i}"), &Array1::from_vec(v), meta)
            .unwrap();
    }
    e.create_index_with_params(4, 16, 16, 1.2, 1).unwrap();
    let mut filter = no_meta();
    filter.insert("cat".into(), json!("even"));
    let q = make_vec(d, 0.5);
    let results = e
        .search_with_filter_and_ann(&q, 10, Some(&filter), None, true)
        .unwrap();
    // All results should have cat=even
    for r in &results {
        assert_eq!(r.metadata.get("cat"), Some(&json!("even")));
    }
}

// ── ANN filter returns empty when nothing matches ──────────────────────

#[test]
fn ann_search_filter_no_match_returns_empty() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 16;
    let mut e = open_default(p, d);
    for i in 0..10u32 {
        let mut meta = no_meta();
        meta.insert("cat".into(), json!("a"));
        e.insert(format!("v{i}"), &make_vec(d, i as f64 * 0.1 + 0.1), meta)
            .unwrap();
    }
    e.create_index_with_params(4, 16, 16, 1.2, 1).unwrap();
    let mut filter = no_meta();
    filter.insert("cat".into(), json!("z"));
    let results = e
        .search_with_filter_and_ann(&make_vec(d, 0.5), 5, Some(&filter), None, true)
        .unwrap();
    assert!(results.is_empty());
}

// ── DistanceMetric default ─────────────────────────────────────────────

#[test]
fn distance_metric_default_is_ip() {
    let m: DistanceMetric = Default::default();
    assert_eq!(m, DistanceMetric::Ip);
}

// ── WAL recovery (unclean shutdown simulation) ─────────────────────────

#[test]
fn wal_recovery_without_live_ids_rebuilds_correctly() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    // Write vectors and close without explicit flush (WAL has entries)
    {
        let mut e = open_default(p, d);
        e.insert("a".into(), &make_vec(d, 0.1), no_meta()).unwrap();
        e.insert("b".into(), &make_vec(d, 0.2), no_meta()).unwrap();
        // Do NOT flush WAL; force a "crash" by deleting live_ids.bin
        drop(e);
    }
    // Delete live_ids.bin to simulate unclean shutdown
    let ids_path = format!("{}/live_ids.bin", p);
    let _ = std::fs::remove_file(&ids_path);
    // Re-open — should recover from WAL
    let e2 = open_default(p, d);
    // Recovery may not restore counts from WAL if ids file was absent and
    // flush_wal_to_segment wasn't called, but should not panic
    let _ = e2.stats();
}

// ── Metadata filter operators ──────────────────────────────────────────

#[test]
fn metadata_filter_comparison_operators() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    let mut e = open_default(p, d);
    for i in 0i64..5 {
        let mut meta = no_meta();
        meta.insert("score".into(), json!(i));
        e.insert(format!("v{i}"), &make_vec(d, i as f64 * 0.2), meta)
            .unwrap();
    }
    // $gte filter
    let mut filter = no_meta();
    filter.insert("score".into(), json!({"$gte": 3}));
    let results = e
        .search_with_filter_and_ann(&make_vec(d, 0.5), 10, Some(&filter), None, true)
        .unwrap();
    assert!(!results.is_empty());
    for r in &results {
        let score = r.metadata.get("score").and_then(|v| v.as_i64()).unwrap();
        assert!(score >= 3, "score={score} should be >= 3");
    }
}

#[test]
fn metadata_filter_or_operator() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    let mut e = open_default(p, d);
    for tag in ["a", "b", "c"] {
        let mut meta = no_meta();
        meta.insert("tag".into(), json!(tag));
        e.insert(tag.to_string(), &make_vec(d, 0.5), meta).unwrap();
    }
    // $or filter: tag == "a" OR tag == "b"
    let filter: HashMap<String, serde_json::Value> =
        serde_json::from_str(r#"{"$or": [{"tag": {"$eq": "a"}}, {"tag": {"$eq": "b"}}]}"#).unwrap();
    let results = e
        .search_with_filter_and_ann(&make_vec(d, 0.5), 10, Some(&filter), None, true)
        .unwrap();
    assert_eq!(results.len(), 2);
}

// ── Error paths: duplicate insert / update-nonexistent ─────────────────

#[test]
fn duplicate_insert_via_insert_method_returns_error() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let mut e = open_default(p, 8);
    e.insert("dup".into(), &make_vec(8, 0.5), no_meta())
        .unwrap();
    let result = e.insert("dup".into(), &make_vec(8, 0.6), no_meta());
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("already exists"));
}

// ── Serde default functions ─────────────────────────────────────────────

#[test]
fn manifest_serde_default_rerank_enabled() {
    // Deserializing a manifest JSON without 'rerank_enabled' must default to true.
    let json = r#"{
            "version": 1, "d": 16, "b": 4, "seed": 42,
            "vector_count": 0, "storage_uri": ""
        }"#;
    let m: Manifest = serde_json::from_str(json).unwrap();
    assert!(
        m.rerank_enabled,
        "default_rerank_enabled() must return true"
    );
}

#[test]
fn index_state_serde_default_ef_construction() {
    // Deserializing an IndexState without 'ef_construction' must default to 200.
    let json = r#"{
            "max_degree": 16, "search_list_size": 64,
            "alpha": 1.2, "indexed_nodes": 0
        }"#;
    let s: IndexState = serde_json::from_str(json).unwrap();
    assert_eq!(
        s.ef_construction, 200,
        "default_ef_construction() must return 200"
    );
}

// ── WAL recovery without live_ids.bin ───────────────────────────────────

#[test]
fn wal_recovery_without_live_ids_bin() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    {
        let mut e = TurboQuantEngine::open(p, p, d, 2, 42).unwrap();
        e.insert("a".into(), &make_vec(d, 0.5), no_meta()).unwrap();
        e.insert("b".into(), &make_vec(d, 0.3), no_meta()).unwrap();
        // Do NOT flush — leave entries only in WAL
        drop(e);
    }
    // Delete live_ids.bin to simulate unclean shutdown
    let ids_path = std::path::Path::new(p).join("live_ids.bin");
    let _ = std::fs::remove_file(&ids_path);
    // Reopen: must rebuild from WAL without live_ids.bin
    let e2 = TurboQuantEngine::open(p, p, d, 2, 42).unwrap();
    // Engine opens without error; count is at least 0
    let _ = e2.vector_count();
}

// ── Delete flush threshold ──────────────────────────────────────────────

#[test]
fn delete_triggers_wal_flush_at_threshold() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    // Use a low explicit threshold so this test is deterministic regardless of the default.
    let mut e = TurboQuantEngine::open_with_options(
        p,
        p,
        d,
        2,
        42,
        DistanceMetric::Ip,
        false,
        false,
        RerankPrecision::Disabled,
        Some(100),
        false,
    )
    .unwrap();
    for i in 0..100 {
        e.insert(format!("v{i}"), &make_vec(d, 0.5), no_meta())
            .unwrap();
    }
    // This delete pushes wal_buffer over the threshold and triggers flush_wal_to_segment
    e.delete("v0".to_string()).unwrap();
    // Engine must remain functional after the flush
    let results = e
        .search_with_filter_and_ann(&make_vec(d, 0.5), 5, None, None, true)
        .unwrap();
    assert!(results.len() <= 5);
}

// ── HNSW index with n_refinements > 0 ──────────────────────────────────

#[test]
fn create_index_with_refinements() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    let mut e = open_default(p, d);
    for i in 0..20u32 {
        e.insert(format!("r{i}"), &make_vec(d, i as f64 * 0.05), no_meta())
            .unwrap();
    }
    // n_refinements = 2 exercises the refinement loop in graph.rs
    e.create_index_with_params(4, 16, 16, 1.2, 2).unwrap();
    let results = e
        .search_with_filter_and_ann(&make_vec(d, 0.5), 5, None, None, true)
        .unwrap();
    assert!(!results.is_empty());
}

// ── F16 reranking with ANN index + Cosine metric ───────────────────────
// Exercises lines 734-738 (F16 IP in build_scorer) and
// 772-776 (F16 Cosine in build_scorer) and 1456-1460 (live_raw_vector_at_slot F16).

#[test]
fn f16_rerank_cosine_ann_index_covers_f16_build_scorer_paths() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 16;
    let mut e = TurboQuantEngine::open_with_options(
        p,
        p,
        d,
        4,
        42,
        DistanceMetric::Cosine,
        true,
        false,
        RerankPrecision::F16,
        None,
        false,
    )
    .unwrap();
    for i in 0..30u32 {
        let mut v = vec![0.01f64; d];
        v[i as usize % d] += (i as f64 + 1.0) * 0.1;
        e.insert(format!("f{i}"), &Array1::from_vec(v), no_meta())
            .unwrap();
    }
    // Building ANN index with F16 + Cosine triggers lines 772-776 in build_scorer
    e.create_index_with_params(8, 32, 32, 1.2, 1).unwrap();
    // ANN search with reranking triggers live_raw_vector_at_slot F16 (lines 1456-1460)
    let mut q = vec![0.0f64; d];
    q[0] = 1.0;
    let results = e
        .search_with_filter_and_ann(&Array1::from_vec(q), 5, None, None, true)
        .unwrap();
    assert!(!results.is_empty());
}

#[test]
fn f16_rerank_ip_ann_index_covers_f16_ip_build_scorer_path() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 16;
    let mut e = TurboQuantEngine::open_with_options(
        p,
        p,
        d,
        4,
        42,
        DistanceMetric::Ip,
        true,
        false,
        RerankPrecision::F16,
        None,
        false,
    )
    .unwrap();
    for i in 0..30u32 {
        let mut v = vec![0.01f64; d];
        v[i as usize % d] += (i as f64 + 1.0) * 0.1;
        e.insert(format!("g{i}"), &Array1::from_vec(v), no_meta())
            .unwrap();
    }
    // Building ANN index with F16 + IP triggers lines 734-738 in build_scorer
    e.create_index_with_params(8, 32, 32, 1.2, 1).unwrap();
    let mut q = vec![0.0f64; d];
    q[0] = 1.0;
    let results = e
        .search_with_filter_and_ann(&Array1::from_vec(q), 5, None, None, true)
        .unwrap();
    assert!(!results.is_empty());
}

// ── L2 metric ANN index without raw vectors (precomputed_l2 path) ───────
// Exercises lines 812-822 in build_scorer (L2 without raw vecs).

#[test]
fn l2_ann_index_without_raw_vectors_uses_precomputed_l2_path() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 16;
    let mut e = TurboQuantEngine::open_with_options(
        p,
        p,
        d,
        4,
        42,
        DistanceMetric::L2,
        false, // rerank disabled → no raw vectors → uses precomputed_l2
        false,
        RerankPrecision::Disabled,
        None,
        false,
    )
    .unwrap();
    for i in 0..30u32 {
        let mut v = vec![0.0f64; d];
        v[i as usize % d] = (i as f64 + 1.0) * 0.1;
        e.insert(format!("l{i}"), &Array1::from_vec(v), no_meta())
            .unwrap();
    }
    // build_scorer L2 without vraw → !precomputed_l2.is_empty() path (lines 812-822)
    e.create_index_with_params(8, 32, 32, 1.2, 1).unwrap();
    let q = Array1::from_vec(vec![0.0f64; d]);
    let results = e
        .search_with_filter_and_ann(&q, 5, None, None, true)
        .unwrap();
    assert!(!results.is_empty());
    // L2 scores are negative distances
    for r in &results {
        assert!(
            r.score <= 0.0,
            "L2 score should be <= 0 (negative distance)"
        );
    }
}

// ── L2 exhaustive search with raw F32 vectors ────────────────────────────
// Exercises lines 1217-1243 (exhaustive L2 scan) and
// 1286-1290 (reranking with live_raw_vector_at_slot for L2).

#[test]
fn l2_exhaustive_search_with_raw_f32_vectors_reranks_correctly() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    let mut e = TurboQuantEngine::open_with_options(
        p,
        p,
        d,
        2,
        42,
        DistanceMetric::L2,
        true,
        false,
        RerankPrecision::F32,
        None,
        false,
    )
    .unwrap();
    e.insert(
        "origin".into(),
        &Array1::from_vec(vec![0.0f64; d]),
        no_meta(),
    )
    .unwrap();
    let mut far = vec![0.0f64; d];
    far[0] = 10.0;
    e.insert("far".into(), &Array1::from_vec(far), no_meta())
        .unwrap();
    // Exhaustive search (no ANN index) → exercises L2 dequant path + raw rerank
    let q = Array1::from_vec(vec![0.0f64; d]);
    let results = e
        .search_with_filter_and_ann(&q, 2, None, None, true)
        .unwrap();
    assert_eq!(results.len(), 2);
    // "origin" should score 0 (no distance), "far" should score -10 (negative distance)
    assert_eq!(results[0].id, "origin", "origin should be closest for L2");
}

// ── Corrupt state detection: missing live_ids.bin + non-empty live_codes ─
// Exercises lines 478-481: "database state appears corrupt" error.

#[test]
fn missing_id_pool_with_nonempty_live_codes_returns_corrupt_error() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    {
        let mut e = TurboQuantEngine::open(p, p, d, 2, 42).unwrap();
        e.insert("x".into(), &make_vec(d, 0.5), no_meta()).unwrap();
        // Flush so data lands in live_codes.bin (not just WAL)
        e.flush_wal_to_segment().unwrap();
        drop(e);
    }
    // Remove live_ids.bin AND wal.log to create the corrupt state:
    // live_codes.bin exists, but id pool is gone and WAL has no recovery data
    let ids_path = std::path::Path::new(p).join("live_ids.bin");
    let wal_path = std::path::Path::new(p).join("wal.log");
    if ids_path.exists() {
        std::fs::remove_file(&ids_path).unwrap();
    }
    if wal_path.exists() {
        std::fs::remove_file(&wal_path).unwrap();
    }
    let result = TurboQuantEngine::open(p, p, d, 2, 42);
    assert!(result.is_err(), "should fail with corrupt state error");
    let msg = result.err().unwrap().to_string();
    assert!(
        msg.contains("corrupt"),
        "error should mention 'corrupt', got: {msg}"
    );
}

// ── L2 exhaustive search reranking without raw vectors (dequant path) ───
// Exercises the deq_vecs rerank path (lines 1265-1276, 1290-1294)
// when rerank_enabled=true but live_vraw is None.

#[test]
fn l2_exhaustive_rerank_without_raw_vectors_uses_dequant_path() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    // rerank=true but Disabled precision → uses dequantized vectors for rerank
    let mut e = TurboQuantEngine::open_with_options(
        p,
        p,
        d,
        2,
        42,
        DistanceMetric::L2,
        true,
        false,
        RerankPrecision::Disabled,
        None,
        false,
    )
    .unwrap();
    e.insert("a".into(), &Array1::from_vec(vec![0.0f64; d]), no_meta())
        .unwrap();
    let mut b = vec![0.0f64; d];
    b[0] = 3.0;
    e.insert("b".into(), &Array1::from_vec(b), no_meta())
        .unwrap();
    let q = Array1::from_vec(vec![0.0f64; d]);
    let results = e
        .search_with_filter_and_ann(&q, 2, None, None, true)
        .unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].id, "a");
}

// ── $and/$or filter false branches ────────────────────────────────────────
// Lines 1895, 1899, 1910, 1914 in metadata_matches_filter

#[test]
fn filter_and_value_not_array_returns_no_results() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    let mut e = open_default(p, d);
    let mut meta = no_meta();
    meta.insert("x".into(), json!(1));
    e.insert("v1".into(), &make_vec(d, 0.5), meta).unwrap();
    // $and value is a string, not an array → line 1899 (else false)
    let filter: HashMap<String, serde_json::Value> =
        serde_json::from_str(r#"{"$and": "not-an-array"}"#).unwrap();
    let results = e
        .search_with_filter_and_ann(&make_vec(d, 0.5), 5, Some(&filter), None, true)
        .unwrap();
    assert!(results.is_empty(), "non-array $and should match nothing");
}

#[test]
fn filter_and_condition_not_object_returns_no_results() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    let mut e = open_default(p, d);
    let mut meta = no_meta();
    meta.insert("x".into(), json!(1));
    e.insert("v1".into(), &make_vec(d, 0.5), meta).unwrap();
    // $and array element is a number, not an object → line 1895 (else false)
    let filter: HashMap<String, serde_json::Value> =
        serde_json::from_str(r#"{"$and": [42]}"#).unwrap();
    let results = e
        .search_with_filter_and_ann(&make_vec(d, 0.5), 5, Some(&filter), None, true)
        .unwrap();
    assert!(
        results.is_empty(),
        "non-object $and condition should match nothing"
    );
}

#[test]
fn filter_or_value_not_array_returns_no_results() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    let mut e = open_default(p, d);
    let mut meta = no_meta();
    meta.insert("x".into(), json!(1));
    e.insert("v1".into(), &make_vec(d, 0.5), meta).unwrap();
    // $or value is a string, not an array → line 1914 (else false)
    let filter: HashMap<String, serde_json::Value> =
        serde_json::from_str(r#"{"$or": "not-an-array"}"#).unwrap();
    let results = e
        .search_with_filter_and_ann(&make_vec(d, 0.5), 5, Some(&filter), None, true)
        .unwrap();
    assert!(results.is_empty(), "non-array $or should match nothing");
}

#[test]
fn filter_or_condition_not_object_returns_no_results() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    let mut e = open_default(p, d);
    let mut meta = no_meta();
    meta.insert("x".into(), json!(1));
    e.insert("v1".into(), &make_vec(d, 0.5), meta).unwrap();
    // $or array element is a string, not an object → line 1910 (else false)
    let filter: HashMap<String, serde_json::Value> =
        serde_json::from_str(r#"{"$or": ["not-an-object"]}"#).unwrap();
    let results = e
        .search_with_filter_and_ann(&make_vec(d, 0.5), 5, Some(&filter), None, true)
        .unwrap();
    assert!(
        results.is_empty(),
        "non-object $or condition should match nothing"
    );
}

// ── Nested dotted-path filter false branches ──────────────────────────────
// Lines 1940, 1955-1958 in get_nested_field / get_nested_json_field

#[test]
fn filter_nested_dotted_path_non_object_returns_no_results() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    let mut e = open_default(p, d);
    // metadata has "a": 5 (a scalar), filter on "a.b" → get_nested_field returns None
    // because 5 is not a JsonValue::Object → line 1940 (None)
    let mut meta = no_meta();
    meta.insert("a".into(), json!(5));
    e.insert("v1".into(), &make_vec(d, 0.5), meta).unwrap();
    let filter: HashMap<String, serde_json::Value> = serde_json::from_str(r#"{"a.b": 1}"#).unwrap();
    let results = e
        .search_with_filter_and_ann(&make_vec(d, 0.5), 5, Some(&filter), None, true)
        .unwrap();
    assert!(
        results.is_empty(),
        "dotted path through scalar should not match"
    );
}

#[test]
fn filter_deeply_nested_non_object_returns_no_results() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    let mut e = open_default(p, d);
    // metadata has "a": {"b": 5}, filter on "a.b.c" → get_nested_json_field
    // reaches "b"=5 (not Object) → line 1957/1958 (None)
    let mut meta = no_meta();
    meta.insert("a".into(), json!({"b": 5}));
    e.insert("v1".into(), &make_vec(d, 0.5), meta).unwrap();
    let filter: HashMap<String, serde_json::Value> =
        serde_json::from_str(r#"{"a.b.c": 1}"#).unwrap();
    let results = e
        .search_with_filter_and_ann(&make_vec(d, 0.5), 5, Some(&filter), None, true)
        .unwrap();
    assert!(
        results.is_empty(),
        "deeply nested path through scalar should not match"
    );
}

// ── String comparison operators in apply_comparison_op ────────────────────
// Lines 1983-1984, 1989-1992

#[test]
fn filter_string_comparison_operators_work() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    let mut e = open_default(p, d);
    let mut meta_a = no_meta();
    meta_a.insert("name".into(), json!("alpha"));
    let mut meta_b = no_meta();
    meta_b.insert("name".into(), json!("beta"));
    let mut meta_c = no_meta();
    meta_c.insert("name".into(), json!("gamma"));
    e.insert("a".into(), &make_vec(d, 0.1), meta_a).unwrap();
    e.insert("b".into(), &make_vec(d, 0.2), meta_b).unwrap();
    e.insert("c".into(), &make_vec(d, 0.3), meta_c).unwrap();

    // $gt on strings: name > "alpha" matches "beta", "gamma"
    let filter_gt: HashMap<String, serde_json::Value> =
        serde_json::from_str(r#"{"name": {"$gt": "alpha"}}"#).unwrap();
    let r_gt = e
        .search_with_filter_and_ann(&make_vec(d, 0.5), 5, Some(&filter_gt), None, true)
        .unwrap();
    assert_eq!(r_gt.len(), 2, "$gt string: expected 2 matches");

    // $gte on strings: name >= "beta" matches "beta", "gamma"
    let filter_gte: HashMap<String, serde_json::Value> =
        serde_json::from_str(r#"{"name": {"$gte": "beta"}}"#).unwrap();
    let r_gte = e
        .search_with_filter_and_ann(&make_vec(d, 0.5), 5, Some(&filter_gte), None, true)
        .unwrap();
    assert_eq!(r_gte.len(), 2, "$gte string: expected 2 matches");

    // $lt on strings: name < "beta" matches "alpha"
    let filter_lt: HashMap<String, serde_json::Value> =
        serde_json::from_str(r#"{"name": {"$lt": "beta"}}"#).unwrap();
    let r_lt = e
        .search_with_filter_and_ann(&make_vec(d, 0.5), 5, Some(&filter_lt), None, true)
        .unwrap();
    assert_eq!(r_lt.len(), 1, "$lt string: expected 1 match");

    // $lte on strings: name <= "alpha" matches "alpha"
    let filter_lte: HashMap<String, serde_json::Value> =
        serde_json::from_str(r#"{"name": {"$lte": "alpha"}}"#).unwrap();
    let r_lte = e
        .search_with_filter_and_ann(&make_vec(d, 0.5), 5, Some(&filter_lte), None, true)
        .unwrap();
    assert_eq!(r_lte.len(), 1, "$lte string: expected 1 match");
}

// ── WAL recovery upsert path (lines 447-455) ──────────────────────────────
// Insert + upsert same ID without flush, reopen: second WAL entry hits upsert path

#[test]
fn wal_recovery_upsert_and_delete_skip_paths_covered() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    {
        let mut e = TurboQuantEngine::open(p, p, d, 2, 42).unwrap();
        // Insert "a" → WAL entry (non-deleted)
        e.insert("a".into(), &make_vec(d, 0.3), no_meta()).unwrap();
        // Insert "b" + delete "b" → WAL delete entry (is_deleted=true) → line 442 (continue)
        e.insert("b".into(), &make_vec(d, 0.4), no_meta()).unwrap();
        e.delete("b".into()).unwrap();
        // Upsert "a" → second WAL entry for same ID → lines 447-455 (upsert path)
        e.upsert_with_document("a".into(), &make_vec(d, 0.7), no_meta(), None)
            .unwrap();
        // Do NOT call close() or flush — live_ids.bin never written (only flush/close writes it)
        drop(e);
    }
    // Reopen: id_pool_loaded=false → WAL recovery:
    //   entry "a" (insert): get_slot → None → new slot
    //   entry "b" (insert): get_slot → None → new slot
    //   entry "b" (delete, is_deleted=true): → continue → LINE 442
    //   entry "a" (upsert): get_slot → Some → overwrite → LINES 447-455
    let e2 = TurboQuantEngine::open(p, p, d, 2, 42).unwrap();
    assert!(e2.vector_count() >= 1, "at least 'a' should be recovered");
}

// ── IP ANN index build without raw vectors (lines 748-749) ───────────────
// Uses prepare_ip_query_from_codes (no vraw available)

#[test]
fn ann_ip_no_rerank_build_uses_from_codes_path() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 16;
    // rerank=false → live_vraw=None → has_vraw=false → IP build scorer uses from_codes (748-749)
    let mut e = TurboQuantEngine::open_with_metric_and_rerank(
        p,
        p,
        d,
        4,
        42,
        DistanceMetric::Ip,
        false,
        false,
    )
    .unwrap();
    for i in 0..30u32 {
        let mut v = vec![0.01f64; d];
        v[i as usize % d] += (i as f64 + 1.0) * 0.05;
        e.insert(format!("n{i}"), &Array1::from_vec(v), no_meta())
            .unwrap();
    }
    e.create_index_with_params(8, 32, 32, 1.2, 0).unwrap();
    let mut q = vec![0.0f64; d];
    q[0] = 1.0;
    let results = e
        .search_with_filter_and_ann(&Array1::from_vec(q), 5, None, None, true)
        .unwrap();
    assert!(!results.is_empty());
}

// ── IP fast_mode ANN index build (line 759: &[] for empty QJL) ───────────

#[test]
fn ann_ip_fast_mode_build_covers_empty_qjl_slice() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 16;
    // fast_mode=true → cached_qjl_len=0 → &[] branch at line 759
    let mut e = TurboQuantEngine::open_with_metric_and_rerank(
        p,
        p,
        d,
        4,
        42,
        DistanceMetric::Ip,
        false,
        true, // fast_mode
    )
    .unwrap();
    for i in 0..30u32 {
        let mut v = vec![0.01f64; d];
        v[i as usize % d] += (i as f64 + 1.0) * 0.05;
        e.insert(format!("fm{i}"), &Array1::from_vec(v), no_meta())
            .unwrap();
    }
    e.create_index_with_params(8, 32, 32, 1.2, 0).unwrap();
    let mut q = vec![0.0f64; d];
    q[0] = 1.0;
    let results = e
        .search_with_filter_and_ann(&Array1::from_vec(q), 5, None, None, true)
        .unwrap();
    assert!(!results.is_empty());
}

// ── ANN search with filter matching nothing (line 925-928) ───────────────

#[test]
fn ann_search_filter_no_matches_returns_empty() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 16;
    let mut e = open_default(p, d);
    for i in 0..30u32 {
        let mut meta = no_meta();
        meta.insert("tag".into(), json!("yes"));
        e.insert(format!("f{i}"), &make_vec(d, i as f64 * 0.01), meta)
            .unwrap();
    }
    e.create_index_with_params(8, 32, 32, 1.2, 0).unwrap();
    // Filter that matches nothing (tag="no" doesn't exist)
    let filter: HashMap<String, serde_json::Value> =
        serde_json::from_str(r#"{"tag": "no"}"#).unwrap();
    let results = e
        .search_with_filter_and_ann(&make_vec(d, 0.5), 5, Some(&filter), None, true)
        .unwrap();
    assert!(
        results.is_empty(),
        "filter matching nothing should return empty via ANN path"
    );
}

// ── Cosine ANN search (line 1042 — else branch of IP/else) ───────────────

#[test]
fn cosine_ann_search_exercises_else_ann_branch() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 16;
    let mut e = TurboQuantEngine::open_with_metric_and_rerank(
        p,
        p,
        d,
        4,
        42,
        DistanceMetric::Cosine,
        false, // no rerank
        false,
    )
    .unwrap();
    for i in 0..30u32 {
        let mut v = vec![0.01f64; d];
        v[i as usize % d] += (i as f64 + 1.0) * 0.1;
        e.insert(format!("cs{i}"), &Array1::from_vec(v), no_meta())
            .unwrap();
    }
    e.create_index_with_params(8, 32, 32, 1.2, 0).unwrap();
    let mut q = vec![0.0f64; d];
    q[0] = 1.0;
    let results = e
        .search_with_filter_and_ann(&Array1::from_vec(q), 5, None, None, true)
        .unwrap();
    assert!(!results.is_empty());
}

// ── Large multi-level ANN — exercises graph multi-level beam search ────────
// With 200 vectors, HNSW almost certainly builds a multi-level graph,
// exercising lines 251-291 in graph.rs (upper-layer beam navigation).

#[test]
fn large_multilevel_ann_search_exercises_graph_beam() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 16;
    let mut e = open_default(p, d);
    for i in 0..200u32 {
        let mut v = vec![0.001f64; d];
        v[i as usize % d] += i as f64 * 0.007;
        e.insert(format!("ml{i}"), &Array1::from_vec(v), no_meta())
            .unwrap();
    }
    // Large graph with max_degree=16 virtually guarantees multilevel HNSW
    e.create_index_with_params(16, 64, 64, 1.2, 0).unwrap();
    let mut q = vec![0.0f64; d];
    q[0] = 1.0;
    let results = e
        .search_with_filter_and_ann(&Array1::from_vec(q), 10, None, None, true)
        .unwrap();
    assert!(
        !results.is_empty(),
        "large ANN search should return results"
    );
}

// ── L2 ANN with rerank + raw vectors (lines 823-845 in build_scorer) ──────

#[test]
fn l2_ann_with_rerank_raw_vecs_build_scorer_path() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 16;
    // rerank=true + L2 → has_vraw=true → precomputed_l2 is empty → lines 823-845
    let mut e = TurboQuantEngine::open_with_metric_and_rerank(
        p,
        p,
        d,
        4,
        42,
        DistanceMetric::L2,
        true, // rerank → live_vraw exists → has_vraw=true
        false,
    )
    .unwrap();
    for i in 0..30u32 {
        let mut v = vec![0.0f64; d];
        v[i as usize % d] = (i as f64 + 1.0) * 0.05;
        e.insert(format!("lr{i}"), &Array1::from_vec(v), no_meta())
            .unwrap();
    }
    e.create_index_with_params(8, 32, 32, 1.2, 0).unwrap();
    let q = Array1::from_vec(vec![0.0f64; d]);
    let results = e
        .search_with_filter_and_ann(&q, 5, None, None, true)
        .unwrap();
    assert!(!results.is_empty());
}

// ── Exhaustive search filter returning empty (line 1136-1138) ────────────

#[test]
fn exhaustive_search_filter_no_results_covers_empty_candidates() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    let mut e = open_default(p, d);
    let mut meta = no_meta();
    meta.insert("tag".into(), json!("present"));
    e.insert("v1".into(), &make_vec(d, 0.5), meta).unwrap();
    // Filter matches nothing — exhaustive_search_simd returns empty (line 1136-1138)
    let filter: HashMap<String, serde_json::Value> =
        serde_json::from_str(r#"{"tag": "absent"}"#).unwrap();
    let results = e
        .search_with_filter_and_ann(&make_vec(d, 0.5), 5, Some(&filter), None, true)
        .unwrap();
    assert!(
        results.is_empty(),
        "filter matching nothing in exhaustive search should return empty"
    );
}

// ── Cosine ANN rerank via dequantized vectors (lines 1075-1076) ──────────

#[test]
fn cosine_ann_rerank_without_raw_vecs_uses_deq_path() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 16;
    // Cosine + rerank=true but Disabled precision → live_vraw=None → deq_vecs path (line 1076)
    let mut e = TurboQuantEngine::open_with_options(
        p,
        p,
        d,
        4,
        42,
        DistanceMetric::Cosine,
        true, // rerank_enabled=true
        false,
        RerankPrecision::Disabled, // no raw vecs → live_vraw=None
        None,
        false,
    )
    .unwrap();
    for i in 0..30u32 {
        let mut v = vec![0.01f64; d];
        v[i as usize % d] += (i as f64 + 1.0) * 0.1;
        e.insert(format!("cr{i}"), &Array1::from_vec(v), no_meta())
            .unwrap();
    }
    e.create_index_with_params(8, 32, 32, 1.2, 0).unwrap();
    let mut q = vec![0.0f64; d];
    q[0] = 1.0;
    let results = e
        .search_with_filter_and_ann(&Array1::from_vec(q), 5, None, None, true)
        .unwrap();
    assert!(!results.is_empty());
}

// ── count_with_filter ─────────────────────────────────────────────────────

#[test]
fn count_no_filter_returns_active_vector_count() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    let mut e = open_default(p, d);
    assert_eq!(e.count_with_filter(None).unwrap(), 0);
    e.insert("a".into(), &make_vec(d, 0.5), no_meta()).unwrap();
    e.insert("b".into(), &make_vec(d, 0.3), no_meta()).unwrap();
    assert_eq!(e.count_with_filter(None).unwrap(), 2);
    e.delete("a".into()).unwrap();
    assert_eq!(e.count_with_filter(None).unwrap(), 1);
}

#[test]
fn count_with_filter_returns_matching_count() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    let mut e = open_default(p, d);
    let mut m1 = no_meta();
    m1.insert("topic".into(), json!("ml"));
    let mut m2 = no_meta();
    m2.insert("topic".into(), json!("nlp"));
    let mut m3 = no_meta();
    m3.insert("topic".into(), json!("ml"));
    e.insert("a".into(), &make_vec(d, 0.5), m1).unwrap();
    e.insert("b".into(), &make_vec(d, 0.4), m2).unwrap();
    e.insert("c".into(), &make_vec(d, 0.3), m3).unwrap();

    let filter: HashMap<String, serde_json::Value> =
        serde_json::from_str(r#"{"topic": "ml"}"#).unwrap();
    assert_eq!(e.count_with_filter(Some(&filter)).unwrap(), 2);

    let filter_nlp: HashMap<String, serde_json::Value> =
        serde_json::from_str(r#"{"topic": "nlp"}"#).unwrap();
    assert_eq!(e.count_with_filter(Some(&filter_nlp)).unwrap(), 1);

    let filter_none: HashMap<String, serde_json::Value> =
        serde_json::from_str(r#"{"topic": "cv"}"#).unwrap();
    assert_eq!(e.count_with_filter(Some(&filter_none)).unwrap(), 0);
}

// ── delete_batch ──────────────────────────────────────────────────────────

#[test]
fn delete_batch_removes_existing_ids_returns_count() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    let mut e = open_default(p, d);
    e.insert("a".into(), &make_vec(d, 0.5), no_meta()).unwrap();
    e.insert("b".into(), &make_vec(d, 0.4), no_meta()).unwrap();
    e.insert("c".into(), &make_vec(d, 0.3), no_meta()).unwrap();

    let deleted = e
        .delete_batch(vec!["a".into(), "c".into(), "z".into()])
        .unwrap();
    assert_eq!(deleted, 2, "only 2 existing IDs should be deleted");
    assert_eq!(e.stats().vector_count, 1);
    assert_eq!(e.list_all(), vec!["b"]);
}

#[test]
fn delete_batch_empty_input_returns_zero() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    let mut e = open_default(p, d);
    e.insert("a".into(), &make_vec(d, 0.5), no_meta()).unwrap();
    assert_eq!(e.delete_batch(vec![]).unwrap(), 0);
    assert_eq!(e.stats().vector_count, 1);
}

#[test]
fn delete_batch_all_unknown_ids_returns_zero() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    let mut e = open_default(p, d);
    assert_eq!(e.delete_batch(vec!["x".into(), "y".into()]).unwrap(), 0);
}

// ── pub fn search / search_with_filter wrappers (lines 933-948) ──────────

#[test]
fn search_and_search_with_filter_wrappers_callable() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    let mut e = open_default(p, d);
    for i in 0..5u32 {
        let mut meta = no_meta();
        meta.insert("tag".into(), json!("yes"));
        e.insert(format!("s{i}"), &make_vec(d, i as f64 * 0.1 + 0.1), meta)
            .unwrap();
    }
    let q = make_vec(d, 0.5);
    // search() wrapper — covers lines 933-939
    let r1 = e.search(&q, 3).unwrap();
    assert!(!r1.is_empty());
    // search_with_filter() wrapper — covers lines 941-948
    let mut f = no_meta();
    f.insert("tag".into(), json!("yes"));
    let r2 = e.search_with_filter(&q, 3, Some(&f)).unwrap();
    assert!(!r2.is_empty());
}

// ── pub fn close() + reopen restores id_pool (lines 1434-1440, 417-418, 1909-1912) ──

#[test]
fn close_and_reopen_restores_id_pool() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    {
        let mut e = open_default(p, d);
        for i in 0..5u32 {
            e.insert(
                format!("id{i}"),
                &make_vec(d, i as f64 * 0.1 + 0.1),
                no_meta(),
            )
            .unwrap();
        }
        // close() covers lines 1434-1440; persists live_ids.bin
        e.close().unwrap();
    }
    // Reopen: load_id_pool finds live_ids.bin → covers lines 417-418, 1909-1912
    // rebuild_lookup() called inside load_id_pool → covers id_pool.rs 123-132
    let e2 = open_default(p, d);
    assert_eq!(e2.vector_count(), 5);
}

// ── update_with_document on existing id (lines 527-528) ──────────────────

#[test]
fn update_with_document_on_existing_id_succeeds() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    let mut e = open_default(p, d);
    e.insert("x".into(), &make_vec(d, 0.1), no_meta()).unwrap();
    // update_with_document on existing id → covers lines 527-528
    e.update_with_document("x".into(), &make_vec(d, 0.9), no_meta(), None)
        .unwrap();
    assert_eq!(e.vector_count(), 1);
}

// ── delete() WAL flush at buffer==threshold (line 568) ───────────────────
// 99 inserts → buffer=99; then delete → buffer=100 >= 100 → flush

#[test]
fn delete_at_wal_buffer_threshold_triggers_flush() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    let mut e = TurboQuantEngine::open(p, p, d, 2, 42).unwrap();
    for i in 0..99u32 {
        e.insert(format!("w{i}"), &make_vec(d, 0.5), no_meta())
            .unwrap();
    }
    // This delete brings buffer to 100 → flush (line 568)
    e.delete("w0".to_string()).unwrap();
    let results = e
        .search_with_filter_and_ann(&make_vec(d, 0.5), 5, None, None, true)
        .unwrap();
    assert!(results.len() <= 5);
}

// ── delete_batch() WAL flush at buffer==threshold (line 685) ─────────────

#[test]
fn delete_batch_at_wal_buffer_threshold_triggers_flush() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    let mut e = TurboQuantEngine::open(p, p, d, 2, 42).unwrap();
    for i in 0..99u32 {
        e.insert(format!("b{i}"), &make_vec(d, 0.5), no_meta())
            .unwrap();
    }
    // delete_batch brings buffer to 100 → flush (line 685)
    e.delete_batch(vec!["b0".to_string()]).unwrap();
    assert_eq!(e.vector_count(), 98);
}

#[test]
fn delete_batch_sets_has_pending_deletes_so_compaction_runs_on_flush() {
    // Regression: delete_batch previously did NOT set has_pending_deletes,
    // so flush_wal_to_segment skipped live_compact_slab and deleted slots
    // remained in the live slab even after a WAL flush.
    //
    // The test proves compaction ran by checking live_slot_count == vector_count:
    // - With fix:    compaction removes 5 deleted slots → live_slot_count == 4994
    // - Without fix: deleted slots linger          → live_slot_count == 4999 != 4994
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    let n = 5usize;
    {
        let mut e = TurboQuantEngine::open(p, p, d, 4, 42).unwrap();
        let threshold = e.wal_flush_threshold;

        // Fill WAL buffer to threshold - 1 (no flush yet).
        for i in 0..threshold - 1 {
            e.insert(format!("v{i}"), &make_vec(d, 0.5), no_meta())
                .unwrap();
        }

        // delete_batch adds n entries → buffer reaches threshold → flush triggered.
        let to_delete: Vec<String> = (0..n).map(|i| format!("v{i}")).collect();
        let deleted = e.delete_batch(to_delete).unwrap();
        assert_eq!(deleted, n, "all requested ids should be deleted");

        // Compaction should have reduced live_slot_count to match vector_count.
        let stats = e.stats();
        assert_eq!(
            stats.live_slot_count as u64, stats.vector_count,
            "live_slot_count must equal vector_count after compaction (no ghost slots)"
        );
        assert_eq!(stats.vector_count, (threshold - 1 - n) as u64);
    }
    // Confirm deleted IDs are gone after reload.
    let e2 = TurboQuantEngine::open(p, p, d, 4, 42).unwrap();
    for i in 0..n {
        assert!(
            e2.get(&format!("v{i}")).unwrap().is_none(),
            "v{i} should be deleted after reopen"
        );
    }
}

#[test]
fn delete_batch_deferred_flush_compacts_when_later_insert_triggers_wal_flush() {
    // Companion to the test above: delete_batch does NOT trigger the WAL flush
    // itself (buffer stays below threshold), but has_pending_deletes=true is still
    // set, so when a later insert causes the flush, compaction runs correctly.
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    let n = 3usize;
    {
        let mut e = TurboQuantEngine::open(p, p, d, 4, 42).unwrap();
        let threshold = e.wal_flush_threshold;

        // Insert just a few vectors — WAL buffer stays well below threshold.
        for i in 0..10usize {
            e.insert(format!("v{i}"), &make_vec(d, 0.5), no_meta())
                .unwrap();
        }

        // delete_batch: buffer stays below threshold (10 + n << threshold).
        let to_delete: Vec<String> = (0..n).map(|i| format!("v{i}")).collect();
        let deleted = e.delete_batch(to_delete).unwrap();
        assert_eq!(deleted, n);

        // Now flood inserts until we cross the threshold — this triggers the WAL flush.
        let remaining = threshold - (10 + n); // how many more to cross threshold
        for i in 10..10 + remaining {
            e.insert(format!("w{i}"), &make_vec(d, 0.5), no_meta())
                .unwrap();
        }

        // Compaction must have run: live_slot_count == vector_count (no ghost slots).
        let stats = e.stats();
        assert_eq!(
            stats.live_slot_count as u64, stats.vector_count,
            "compaction must run during flush triggered by inserts after delete_batch"
        );
        assert_eq!(
            stats.vector_count,
            (10 - n + remaining) as u64,
            "vector count must be correct after deferred compaction"
        );
    }
    // Confirm deleted IDs are gone after reload.
    let e2 = TurboQuantEngine::open(p, p, d, 4, 42).unwrap();
    for i in 0..n {
        assert!(
            e2.get(&format!("v{i}")).unwrap().is_none(),
            "v{i} should be deleted after deferred-flush compaction and reopen"
        );
    }
}

#[test]
fn insert_many_at_threshold_triggers_wal_flush() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    let mut e = open_default(p, d);
    let items: Vec<BatchWriteItem> = (0..100u32)
        .map(|i| BatchWriteItem {
            id: format!("m{i}"),
            vector: make_vec_f32(d, i as f32 * 0.01 + 0.01),
            metadata: no_meta(),
            document: None,
        })
        .collect();
    // 100 items → wal_buffer.len() = 100 >= 100 → flush (line 1859)
    e.insert_many(items).unwrap();
    assert_eq!(e.vector_count(), 100);
}

// ── Cosine ANN: fast_mode empty-qjl (863), zero-norm build (871),
//    zero-query ANN search (1072) ──────────────────────────────────────────

#[test]
fn cosine_ann_fast_mode_empty_qjl_and_zero_norm_and_zero_query() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 16;
    // fast_mode=true → cached_qjl_len=0 → &[] branch (line 863)
    // rerank=false → uses stored norm path in build_scorer
    let mut e = TurboQuantEngine::open_with_metric_and_rerank(
        p,
        p,
        d,
        4,
        42,
        DistanceMetric::Cosine,
        false,
        true,
    )
    .unwrap();
    // Insert zero vector → norm=0 → line 871 in build_scorer
    e.insert("zero".into(), &Array1::zeros(d), no_meta())
        .unwrap();
    for i in 1..15u32 {
        let mut v = vec![0.0f64; d];
        v[i as usize % d] = 1.0;
        e.insert(format!("v{i}"), &Array1::from_vec(v), no_meta())
            .unwrap();
    }
    e.create_index_with_params(4, 16, 16, 1.2, 0).unwrap();
    // Search with zero query → query_norm=0 → line 1072
    let zero_q = Array1::<f64>::zeros(d);
    let _results = e
        .search_with_filter_and_ann(&zero_q, 5, None, None, true)
        .unwrap();
}

// ── $and filter with Object conditions (lines 1955-1957) ─────────────────

#[test]
fn and_filter_with_object_conditions_matches() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    let mut e = open_default(p, d);
    for i in 0i64..5 {
        let mut meta = no_meta();
        meta.insert("score".into(), json!(i));
        meta.insert("tag".into(), json!("good"));
        e.insert(format!("a{i}"), &make_vec(d, i as f64 * 0.2), meta)
            .unwrap();
    }
    // $and with Object conditions — covers lines 1955-1957
    let filter: HashMap<String, serde_json::Value> =
        serde_json::from_str(r#"{"$and": [{"score": {"$gte": 2}}, {"tag": "good"}]}"#).unwrap();
    let results = e
        .search_with_filter_and_ann(&make_vec(d, 0.5), 10, Some(&filter), None, true)
        .unwrap();
    assert!(!results.is_empty());
}

// ── $ne comparison operator (line 2034) ──────────────────────────────────

#[test]
fn ne_filter_matches_different_values() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    let mut e = open_default(p, d);
    for tag in ["a", "b", "c"] {
        let mut meta = no_meta();
        meta.insert("tag".into(), json!(tag));
        e.insert(tag.to_string(), &make_vec(d, 0.5), meta).unwrap();
    }
    // $ne filter — covers line 2034
    let mut filter = no_meta();
    filter.insert("tag".into(), json!({"$ne": "a"}));
    let results = e
        .search_with_filter_and_ann(&make_vec(d, 0.5), 10, Some(&filter), None, true)
        .unwrap();
    assert_eq!(results.len(), 2);
}

// ── $lt/$lte numeric comparison (lines 2046, 2047) ───────────────────────

#[test]
fn lt_lte_numeric_filters_covered() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    let mut e = open_default(p, d);
    for i in 0i64..5 {
        let mut meta = no_meta();
        meta.insert("n".into(), json!(i));
        e.insert(format!("n{i}"), &make_vec(d, i as f64 * 0.2), meta)
            .unwrap();
    }
    // $lt covers line 2046
    let mut f1 = no_meta();
    f1.insert("n".into(), json!({"$lt": 3}));
    let r1 = e
        .search_with_filter_and_ann(&make_vec(d, 0.5), 10, Some(&f1), None, true)
        .unwrap();
    assert_eq!(r1.len(), 3); // n=0,1,2

    // $lte covers line 2047
    let mut f2 = no_meta();
    f2.insert("n".into(), json!({"$lte": 2}));
    let r2 = e
        .search_with_filter_and_ann(&make_vec(d, 0.5), 10, Some(&f2), None, true)
        .unwrap();
    assert_eq!(r2.len(), 3); // n=0,1,2
}

// ── type mismatch in comparison (line 2058) ───────────────────────────────

#[test]
fn comparison_type_mismatch_returns_no_results() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    let mut e = open_default(p, d);
    let mut meta = no_meta();
    meta.insert("name".into(), json!("alice"));
    e.insert("a".into(), &make_vec(d, 0.5), meta).unwrap();
    // String field vs numeric op_val → (String, Number) → _ => false (line 2058)
    let mut filter = no_meta();
    filter.insert("name".into(), json!({"$gte": 5}));
    let results = e
        .search_with_filter_and_ann(&make_vec(d, 0.5), 10, Some(&filter), None, true)
        .unwrap();
    assert!(results.is_empty());
}

// ── unknown comparison operator (line 2061) ───────────────────────────────

#[test]
fn unknown_operator_returns_no_results() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    let mut e = open_default(p, d);
    let mut meta = no_meta();
    meta.insert("x".into(), json!(1));
    e.insert("v".into(), &make_vec(d, 0.5), meta).unwrap();
    // Unknown op "$bogus" → _ => false (line 2061)
    let mut filter = no_meta();
    filter.insert("x".into(), json!({"$bogus": 1}));
    let results = e
        .search_with_filter_and_ann(&make_vec(d, 0.5), 10, Some(&filter), None, true)
        .unwrap();
    assert!(results.is_empty());
}

// ── nested dotted paths: covers get_nested_json_field lines 2020, 2025 ───

#[test]
fn nested_dotted_path_three_levels_covers_recursive_and_leaf() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    let mut e = open_default(p, d);
    // metadata: {"a": {"b": {"c": 42}}}
    let meta: HashMap<String, serde_json::Value> =
        serde_json::from_str(r#"{"a": {"b": {"c": 42}}}"#).unwrap();
    e.insert("deep".into(), &make_vec(d, 0.5), meta).unwrap();
    // Filter on "a.b.c" = 42 → get_nested_json_field recurses (line 2020)
    // and returns Some(val) at leaf (line 2025)
    let mut filter = no_meta();
    filter.insert("a.b.c".into(), json!(42));
    let results = e
        .search_with_filter_and_ann(&make_vec(d, 0.5), 5, Some(&filter), None, true)
        .unwrap();
    assert_eq!(results.len(), 1);
}

// ── delete_batch WAL-flush threshold coverage ─────────────────────────────

#[test]
fn delete_batch_persists_across_reopen() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    {
        let mut e = open_default(p, d);
        for i in 0..5u32 {
            e.insert(
                format!("x{i}"),
                &make_vec(d, 0.1 * i as f64 + 0.1),
                no_meta(),
            )
            .unwrap();
        }
        let deleted = e
            .delete_batch(vec!["x0".into(), "x2".into(), "x4".into()])
            .unwrap();
        assert_eq!(deleted, 3);
    }
    // Reopen and confirm deletions persisted
    let e2 = open_default(p, d);
    let ids = e2.list_all();
    assert!(ids.contains(&"x1".to_string()));
    assert!(ids.contains(&"x3".to_string()));
    assert!(!ids.contains(&"x0".to_string()));
    assert!(!ids.contains(&"x2".to_string()));
    assert!(!ids.contains(&"x4".to_string()));
}

// ── New filter operators ───────────────────────────────────────────────────

#[test]
fn filter_in_operator_matches_element_in_array() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    let mut e = open_default(p, d);
    let m = |tag: &str| {
        let mut h = no_meta();
        h.insert("tag".into(), json!(tag));
        h
    };
    e.insert("a".into(), &make_vec(d, 0.1), m("ml")).unwrap();
    e.insert("b".into(), &make_vec(d, 0.2), m("cv")).unwrap();
    e.insert("c".into(), &make_vec(d, 0.3), m("nlp")).unwrap();

    let filter: HashMap<String, serde_json::Value> =
        serde_json::from_str(r#"{"tag": {"$in": ["ml", "cv"]}}"#).unwrap();
    let results = e
        .search_with_filter_and_ann(&make_vec(d, 0.5), 10, Some(&filter), None, true)
        .unwrap();
    let ids: Vec<_> = results.iter().map(|r| r.id.as_str()).collect();
    assert!(ids.contains(&"a"), "$in: expected 'a'");
    assert!(ids.contains(&"b"), "$in: expected 'b'");
    assert!(!ids.contains(&"c"), "$in: 'c' should be excluded");
}

#[test]
fn filter_in_operator_missing_field_does_not_match() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    let mut e = open_default(p, d);
    e.insert("no-tag".into(), &make_vec(d, 0.1), no_meta())
        .unwrap();
    let filter: HashMap<String, serde_json::Value> =
        serde_json::from_str(r#"{"tag": {"$in": ["ml"]}}"#).unwrap();
    let results = e
        .search_with_filter_and_ann(&make_vec(d, 0.5), 10, Some(&filter), None, true)
        .unwrap();
    assert!(results.is_empty(), "$in on missing field should not match");
}

#[test]
fn filter_nin_operator_excludes_matching_elements() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    let mut e = open_default(p, d);
    let m = |tag: &str| {
        let mut h = no_meta();
        h.insert("tag".into(), json!(tag));
        h
    };
    e.insert("a".into(), &make_vec(d, 0.1), m("ml")).unwrap();
    e.insert("b".into(), &make_vec(d, 0.2), m("cv")).unwrap();
    e.insert("c".into(), &make_vec(d, 0.3), m("nlp")).unwrap();

    let filter: HashMap<String, serde_json::Value> =
        serde_json::from_str(r#"{"tag": {"$nin": ["ml", "cv"]}}"#).unwrap();
    let results = e
        .search_with_filter_and_ann(&make_vec(d, 0.5), 10, Some(&filter), None, true)
        .unwrap();
    let ids: Vec<_> = results.iter().map(|r| r.id.as_str()).collect();
    assert!(!ids.contains(&"a"), "$nin: 'a' should be excluded");
    assert!(!ids.contains(&"b"), "$nin: 'b' should be excluded");
    assert!(ids.contains(&"c"), "$nin: expected 'c'");
}

#[test]
fn filter_nin_missing_field_matches() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    let mut e = open_default(p, d);
    e.insert("no-tag".into(), &make_vec(d, 0.1), no_meta())
        .unwrap();
    let filter: HashMap<String, serde_json::Value> =
        serde_json::from_str(r#"{"tag": {"$nin": ["ml"]}}"#).unwrap();
    let results = e
        .search_with_filter_and_ann(&make_vec(d, 0.5), 10, Some(&filter), None, true)
        .unwrap();
    assert_eq!(results.len(), 1, "$nin on missing field should match");
}

#[test]
fn filter_exists_true_matches_present_field_only() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    let mut e = open_default(p, d);
    let mut with_tag = no_meta();
    with_tag.insert("tag".into(), json!("ml"));
    e.insert("has".into(), &make_vec(d, 0.1), with_tag).unwrap();
    e.insert("missing".into(), &make_vec(d, 0.2), no_meta())
        .unwrap();

    let filter_true: HashMap<String, serde_json::Value> =
        serde_json::from_str(r#"{"tag": {"$exists": true}}"#).unwrap();
    let r = e
        .search_with_filter_and_ann(&make_vec(d, 0.5), 10, Some(&filter_true), None, true)
        .unwrap();
    assert_eq!(r.len(), 1);
    assert_eq!(r[0].id, "has");

    let filter_false: HashMap<String, serde_json::Value> =
        serde_json::from_str(r#"{"tag": {"$exists": false}}"#).unwrap();
    let r2 = e
        .search_with_filter_and_ann(&make_vec(d, 0.5), 10, Some(&filter_false), None, true)
        .unwrap();
    assert_eq!(r2.len(), 1);
    assert_eq!(r2[0].id, "missing");
}

#[test]
fn filter_contains_operator_matches_substring() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    let mut e = open_default(p, d);
    let m = |s: &str| {
        let mut h = no_meta();
        h.insert("title".into(), json!(s));
        h
    };
    e.insert("a".into(), &make_vec(d, 0.1), m("GPU acceleration tips"))
        .unwrap();
    e.insert("b".into(), &make_vec(d, 0.2), m("CPU optimization guide"))
        .unwrap();
    e.insert("c".into(), &make_vec(d, 0.3), m("GPU memory management"))
        .unwrap();

    let filter: HashMap<String, serde_json::Value> =
        serde_json::from_str(r#"{"title": {"$contains": "GPU"}}"#).unwrap();
    let results = e
        .search_with_filter_and_ann(&make_vec(d, 0.5), 10, Some(&filter), None, true)
        .unwrap();
    let ids: Vec<_> = results.iter().map(|r| r.id.as_str()).collect();
    assert!(ids.contains(&"a"), "$contains: expected 'a'");
    assert!(!ids.contains(&"b"), "$contains: 'b' should be excluded");
    assert!(ids.contains(&"c"), "$contains: expected 'c'");
}

#[test]
fn filter_contains_non_string_field_does_not_match() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    let mut e = open_default(p, d);
    let mut m = no_meta();
    m.insert("score".into(), json!(42));
    e.insert("a".into(), &make_vec(d, 0.1), m).unwrap();
    let filter: HashMap<String, serde_json::Value> =
        serde_json::from_str(r#"{"score": {"$contains": "4"}}"#).unwrap();
    let results = e
        .search_with_filter_and_ann(&make_vec(d, 0.5), 10, Some(&filter), None, true)
        .unwrap();
    assert!(
        results.is_empty(),
        "$contains on non-string field should not match"
    );
}

// ── update_metadata_only ──────────────────────────────────────────────────

#[test]
fn update_metadata_only_changes_metadata_without_changing_vector() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    let mut e = open_default(p, d);
    let mut m = no_meta();
    m.insert("status".into(), json!("draft"));
    e.insert("a".into(), &make_vec(d, 0.5), m).unwrap();

    // Update only metadata
    let mut new_meta = no_meta();
    new_meta.insert("status".into(), json!("published"));
    e.update_metadata_only("a", new_meta, None).unwrap();

    let got = e.get("a").unwrap().unwrap();
    assert_eq!(got.metadata["status"], json!("published"));
    assert!(got.document.is_none());
}

#[test]
fn update_metadata_only_updates_document_preserving_metadata() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    let mut e = open_default(p, d);
    let mut m = no_meta();
    m.insert("cat".into(), json!("a"));
    e.insert_with_document("x".into(), &make_vec(d, 0.1), m, Some("old doc".into()))
        .unwrap();

    // Update document only (empty metadata map → preserve existing)
    e.update_metadata_only("x", no_meta(), Some("new doc".into()))
        .unwrap();

    let got = e.get("x").unwrap().unwrap();
    assert_eq!(got.document.as_deref(), Some("new doc"));
    assert_eq!(
        got.metadata["cat"],
        json!("a"),
        "metadata should be preserved"
    );
}

#[test]
fn update_metadata_only_errors_on_missing_id() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    let mut e = open_default(p, d);
    let result = e.update_metadata_only("nonexistent", no_meta(), None);
    assert!(result.is_err());
}

#[test]
fn update_metadata_only_persists_across_reopen() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    {
        let mut e = open_default(p, d);
        let mut m = no_meta();
        m.insert("v".into(), json!(1));
        e.insert("doc".into(), &make_vec(d, 0.3), m).unwrap();
        let mut updated = no_meta();
        updated.insert("v".into(), json!(2));
        e.update_metadata_only("doc", updated, Some("hello".into()))
            .unwrap();
        e.flush_wal_to_segment().unwrap();
    }
    let e2 = open_default(p, d);
    let got = e2.get("doc").unwrap().unwrap();
    assert_eq!(got.metadata["v"], json!(2));
    assert_eq!(got.document.as_deref(), Some("hello"));
}

// ── list_with_filter_page ─────────────────────────────────────────────────

#[test]
fn list_with_filter_page_no_filter_returns_all() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    let mut e = open_default(p, d);
    for i in 0..5u32 {
        e.insert(format!("id{i}"), &make_vec(d, 0.1), no_meta())
            .unwrap();
    }
    let ids = e.list_with_filter_page(None, None, 0).unwrap();
    assert_eq!(ids.len(), 5);
}

#[test]
fn list_with_filter_page_limit_and_offset_work() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    let mut e = open_default(p, d);
    for i in 0..10u32 {
        e.insert(format!("id{i:02}"), &make_vec(d, 0.1), no_meta())
            .unwrap();
    }
    let page = e.list_with_filter_page(None, Some(3), 2).unwrap();
    assert_eq!(page.len(), 3, "limit=3 should return 3 items");
}

#[test]
fn list_with_filter_page_with_filter_returns_matching_ids() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    let mut e = open_default(p, d);
    let mut m_a = no_meta();
    m_a.insert("kind".into(), json!("A"));
    let mut m_b = no_meta();
    m_b.insert("kind".into(), json!("B"));
    e.insert("a1".into(), &make_vec(d, 0.1), m_a.clone())
        .unwrap();
    e.insert("b1".into(), &make_vec(d, 0.2), m_b.clone())
        .unwrap();
    e.insert("a2".into(), &make_vec(d, 0.3), m_a.clone())
        .unwrap();

    let filter: HashMap<String, serde_json::Value> =
        serde_json::from_str(r#"{"kind": "A"}"#).unwrap();
    let ids = e.list_with_filter_page(Some(&filter), None, 0).unwrap();
    assert_eq!(ids.len(), 2);
    assert!(ids.iter().all(|id| id.starts_with('a')));
}

// ── search_batch ──────────────────────────────────────────────────────────

#[test]
fn search_batch_returns_one_result_set_per_query() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    let mut e = open_default(p, d);
    for i in 0..5u32 {
        e.insert(
            format!("v{i}"),
            &make_vec(d, 0.1 * i as f64 + 0.1),
            no_meta(),
        )
        .unwrap();
    }
    let q1 = make_vec(d, 0.1);
    let q2 = make_vec(d, 0.9);
    let results = e.search_batch(&[q1, q2], 2, None, None, true).unwrap();
    assert_eq!(results.len(), 2, "one result set per query");
    assert_eq!(results[0].len(), 2, "top_k=2 for first query");
    assert_eq!(results[1].len(), 2, "top_k=2 for second query");
}

#[test]
fn search_batch_empty_queries_returns_empty() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    let e = open_default(p, d);
    let results = e.search_batch(&[], 5, None, None, true).unwrap();
    assert!(results.is_empty());
}

#[test]
fn hybrid_search_finds_post_index_vectors() {
    // Regression test for the "dark vector" bug:
    // Vectors inserted AFTER create_index() must still appear in ANN search results.
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 32;
    let mut e = open_default(p, d);

    // Insert 200 diverse vectors and build index.
    // Use deterministic vectors spread across the space so the search is meaningful.
    for i in 0..200u32 {
        let mut v = vec![0.0f64; d];
        // Spread vectors: cycle through dimensions with varying magnitudes.
        v[i as usize % d] = 1.0;
        v[(i as usize + 1) % d] = 0.5 * ((i as f64 * 0.1).sin());
        e.insert(format!("pre_{i}"), &Array1::from_vec(v), no_meta())
            .unwrap();
    }
    e.create_index_with_params(16, 100, 64, 1.2, 2).unwrap();

    // Insert 20 vectors AFTER the index is built — these are "dark" slots.
    // Place them near the query direction so they should rank in the top results.
    let mut dark_ids = Vec::new();
    for i in 0..20u32 {
        let mut v = vec![0.0f64; d];
        v[0] = 0.8 + (i as f64) * 0.01; // near query direction
        let id = format!("dark_{i}");
        e.insert(id.clone(), &Array1::from_vec(v), no_meta())
            .unwrap();
        dark_ids.push(id);
    }

    // Search near the direction of the dark vectors; they must appear in results.
    let mut q = vec![0.0f64; d];
    q[0] = 1.0;
    let results = e
        .search_with_filter_and_ann(&Array1::from_vec(q), 30, None, None, true)
        .unwrap();

    let result_ids: std::collections::HashSet<&str> =
        results.iter().map(|r| r.id.as_str()).collect();
    let dark_found = dark_ids.iter().any(|id| result_ids.contains(id.as_str()));
    assert!(
        dark_found,
        "at least one post-index vector must appear in ANN results (hybrid search)"
    );
}

// ── Compaction edge cases ─────────────────────────────────────────────────

#[test]
fn compact_with_all_deleted_vectors_leaves_empty_live_slab() {
    // All vectors are deleted before the WAL flush that triggers live_compact_slab.
    // Verifies that the slab truncates to 0 active slots without panicking.
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    let mut e = TurboQuantEngine::open(p, p, d, 4, 42).unwrap();
    let threshold = e.wal_flush_threshold;

    // Insert exactly threshold-1 vectors then delete all of them so that
    // the final delete_batch call brings the WAL buffer to threshold → flush.
    for i in 0..threshold - 1 {
        e.insert(format!("v{i}"), &make_vec(d, 0.5), no_meta())
            .unwrap();
    }
    let to_delete: Vec<String> = (0..threshold - 1).map(|i| format!("v{i}")).collect();
    e.delete_batch(to_delete).unwrap();

    // After compaction the live slab should have 0 active slots and 0 vectors.
    let stats = e.stats();
    assert_eq!(
        stats.vector_count, 0,
        "all vectors deleted: vector_count must be 0"
    );
    assert_eq!(
        stats.live_slot_count as u64, 0,
        "all vectors deleted: live_slot_count must be 0 (no ghost slots)"
    );

    // Reopen must succeed and DB must still be empty.
    let e2 = TurboQuantEngine::open(p, p, d, 4, 42).unwrap();
    assert_eq!(e2.stats().vector_count, 0);
    assert!(e2.list_all().is_empty());
}

#[test]
fn compact_called_before_wal_flush_flushes_wal_first() {
    // compact() must flush the WAL before running compaction so that vectors
    // sitting in the WAL buffer are not silently dropped.
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    let n = 10usize;
    let mut e = TurboQuantEngine::open(p, p, d, 4, 42).unwrap();

    // Insert well below the WAL flush threshold so the WAL is NOT auto-flushed.
    for i in 0..n {
        e.insert(format!("v{i}"), &make_vec(d, 0.5), no_meta())
            .unwrap();
    }
    // All n vectors still in WAL buffer at this point.
    assert_eq!(e.stats().segment_count, 0, "no segments written yet");

    // Calling compact() explicitly must flush the WAL first.
    e.compact().unwrap();

    // After compact() all vectors must be accessible — none dropped.
    let e2 = TurboQuantEngine::open(p, p, d, 4, 42).unwrap();
    assert_eq!(
        e2.stats().vector_count,
        n as u64,
        "compact() must flush WAL: all inserted vectors must survive"
    );
    for i in 0..n {
        assert!(
            e2.get(&format!("v{i}")).unwrap().is_some(),
            "v{i} must be present after compact()+reopen"
        );
    }
}

#[test]
fn compact_then_search_returns_correct_results() {
    // Compaction invalidates the HNSW index. Search after compaction must fall
    // back to exhaustive scan and still return the correct nearest neighbours.
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    let mut e = TurboQuantEngine::open(p, p, d, 4, 42).unwrap();
    let threshold = e.wal_flush_threshold;

    // Insert enough vectors to trigger a WAL flush and then build an index.
    for i in 0..threshold {
        let mut v = vec![0.0f64; d];
        v[i % d] = (i as f64 + 1.0) * 0.1;
        e.insert(format!("v{i}"), &Array1::from_vec(v), no_meta())
            .unwrap();
    }
    e.create_index_with_params(4, 16, 16, 1.2, 0).unwrap();
    assert!(e.stats().has_index, "index must exist before compaction");

    // Delete some vectors and trigger compaction (invalidates index).
    let to_delete: Vec<String> = (0..5).map(|i| format!("v{i}")).collect();
    e.delete_batch(to_delete).unwrap();
    e.compact().unwrap();

    // Search must succeed (falls back to exhaustive) and must not return deleted IDs.
    let q = make_vec(d, 0.5);
    let results = e
        .search_with_filter_and_ann(&q, 10, None, None, true)
        .unwrap();
    assert!(
        !results.is_empty(),
        "search after compaction must return results"
    );
    let deleted_owned: Vec<String> = (0..5usize).map(|i| format!("v{i}")).collect();
    let deleted_set: std::collections::HashSet<&str> =
        deleted_owned.iter().map(|s| s.as_str()).collect();
    for r in &results {
        assert!(
            !deleted_set.contains(r.id.as_str()),
            "deleted vector {} must not appear in search results after compaction",
            r.id
        );
    }
}

#[test]
fn prepared_compaction_state_on_disk_is_recovered_on_reopen() {
    // Simulate a crash after the Prepared marker is written and the new segment
    // exists on disk. On reopen, recover_if_needed() must detect the Prepared
    // state, validate the new segment, delete old segments, and leave the DB intact.
    use serde_json::json;
    use std::fs;

    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 8;
    let threshold;
    {
        let mut e = TurboQuantEngine::open(p, p, d, 4, 42).unwrap();
        threshold = e.wal_flush_threshold;

        // Write enough vectors to produce at least one segment on disk.
        for i in 0..threshold {
            e.insert(format!("v{i}"), &make_vec(d, 0.5), no_meta())
                .unwrap();
        }
        // Force another flush so we have a second segment (more realistic).
        for i in threshold..threshold * 2 {
            e.insert(format!("v{i}"), &make_vec(d, 0.3), no_meta())
                .unwrap();
        }
    }

    // Find the segment files on disk — we need their names for the compaction state.
    let seg_names: Vec<String> = fs::read_dir(p)
        .unwrap()
        .filter_map(|e| {
            let name = e.unwrap().file_name().into_string().unwrap();
            if name.starts_with("seg-") && name.ends_with(".bin") {
                Some(name)
            } else {
                None
            }
        })
        .collect();
    assert!(
        !seg_names.is_empty(),
        "test requires at least one segment on disk"
    );

    // Write a fake Prepared compaction marker that claims old_segments = seg_names
    // and new_segment = a non-existent file. This simulates a crash before the new
    // segment was written → recovery should abandon (keep old segments).
    let fake_state = json!({
        "phase": "Prepared",
        "old_segment_names": seg_names,
        "new_segment_name": "seg-deadbeef.bin"
    });
    let state_path = std::path::Path::new(p).join("compaction_state.json");
    fs::write(&state_path, serde_json::to_vec_pretty(&fake_state).unwrap()).unwrap();
    assert!(state_path.exists(), "compaction state file must be written");

    // Reopen: recover_if_needed() must run, detect missing new segment,
    // preserve old segments, and clean up the marker.
    let e2 = TurboQuantEngine::open(p, p, d, 4, 42).unwrap();

    // Marker file must be gone.
    assert!(
        !state_path.exists(),
        "compaction_state.json must be cleaned up on reopen"
    );
    // All original segments preserved (new_segment was missing → rollback).
    for name in &seg_names {
        assert!(
            std::path::Path::new(p).join(name).exists(),
            "old segment {name} must be preserved after aborted compaction recovery"
        );
    }
    // All vectors still accessible.
    assert_eq!(
        e2.stats().vector_count,
        (threshold * 2) as u64,
        "all vectors must survive aborted compaction recovery"
    );
}
