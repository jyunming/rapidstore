use ndarray::Array1;
use serde_json::json;
use std::collections::HashMap;
use tempfile::tempdir;
use tqdb::storage::engine::TurboQuantEngine;

fn make_vec(d: usize, val: f64) -> Array1<f64> {
    Array1::from_elem(d, val)
}

fn no_meta() -> HashMap<String, serde_json::Value> {
    HashMap::new()
}

#[test]
fn delete_then_reinsert_persists_after_reopen() {
    // Repro mirrors Python behavior:
    // insert/upsert same ID, delete it, then reinsert same ID before close.
    // After reopen, the ID should still exist.
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 16;

    let mut e = TurboQuantEngine::open(p, p, d, 4, 42).unwrap();
    let mut first_meta = no_meta();
    first_meta.insert("phase".into(), json!(1));
    e.upsert_with_document(
        "x".into(),
        &make_vec(d, 1.0),
        first_meta,
        Some("first".into()),
    )
    .unwrap();
    assert_eq!(e.stats().vector_count, 1);

    let deleted = e.delete("x".into()).unwrap();
    assert!(deleted);
    assert_eq!(e.stats().vector_count, 0);

    let mut second_meta = no_meta();
    second_meta.insert("phase".into(), json!(2));
    e.upsert_with_document(
        "x".into(),
        &make_vec(d, 2.0),
        second_meta,
        Some("second".into()),
    )
    .unwrap();
    assert_eq!(e.stats().vector_count, 1);
    assert!(e.get("x").unwrap().is_some());

    e.close().unwrap();

    let reopened = TurboQuantEngine::open(p, p, d, 4, 42).unwrap();
    assert_eq!(
        reopened.stats().vector_count,
        1,
        "reinserted id should persist after reopen",
    );
    assert!(
        reopened.get("x").unwrap().is_some(),
        "reinserted id missing after reopen",
    );
    let got = reopened.get("x").unwrap().unwrap();
    assert_eq!(got.metadata.get("phase"), Some(&json!(2)));
    assert_eq!(got.document.as_deref(), Some("second"));
}
