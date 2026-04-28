//! Engine-level integration tests for the BM25 sparse-retrieval path.
//!
//! These tests drive the public engine API the same way `Database` (the Python
//! shim) does — insert with documents, search, delete, reopen — and assert that
//! the BM25 sidecar stays consistent with the dense store across every path.

use ndarray::Array1;
use std::collections::HashMap;
use tempfile::tempdir;

use tqdb::storage::engine::TurboQuantEngine;

fn make_vec(d: usize, seed: f64) -> Array1<f64> {
    Array1::<f64>::from_iter((0..d).map(|j| seed + (j as f64) * 0.001))
}

#[test]
fn search_bm25_returns_keyword_matches() {
    let dir = tempdir().unwrap();
    let db = dir.path().to_str().unwrap();
    let d = 32;
    let mut engine = TurboQuantEngine::open(db, db, d, 4, 1).unwrap();

    engine
        .insert_with_document(
            "fox".into(),
            &make_vec(d, 0.1),
            HashMap::new(),
            Some("the quick brown fox".into()),
        )
        .unwrap();
    engine
        .insert_with_document(
            "dog".into(),
            &make_vec(d, 0.2),
            HashMap::new(),
            Some("the lazy dog sleeps".into()),
        )
        .unwrap();
    engine
        .insert_with_document(
            "rabbit".into(),
            &make_vec(d, 0.3),
            HashMap::new(),
            Some("a quick rabbit hops".into()),
        )
        .unwrap();

    assert_eq!(engine.bm25_doc_count(), 3);

    let r = engine.search_bm25("quick", 10, None).unwrap();
    let ids: Vec<&str> = r.iter().map(|(id, _)| id.as_str()).collect();
    assert!(ids.contains(&"fox"), "fox doc must match 'quick'");
    assert!(ids.contains(&"rabbit"), "rabbit doc must match 'quick'");
    assert!(!ids.contains(&"dog"), "dog doc must not match 'quick'");
}

#[test]
fn delete_removes_from_bm25() {
    let dir = tempdir().unwrap();
    let db = dir.path().to_str().unwrap();
    let d = 16;
    let mut engine = TurboQuantEngine::open(db, db, d, 2, 7).unwrap();

    for (id, doc) in [("a", "alpha bravo"), ("b", "alpha gamma"), ("c", "delta")] {
        engine
            .insert_with_document(
                id.into(),
                &make_vec(d, 0.1),
                HashMap::new(),
                Some(doc.into()),
            )
            .unwrap();
    }
    assert_eq!(engine.search_bm25("alpha", 10, None).unwrap().len(), 2);

    engine.delete("a".into()).unwrap();

    let r = engine.search_bm25("alpha", 10, None).unwrap();
    assert_eq!(r.len(), 1);
    assert_eq!(r[0].0, "b");
}

#[test]
fn upsert_replaces_doc_in_bm25() {
    let dir = tempdir().unwrap();
    let db = dir.path().to_str().unwrap();
    let d = 16;
    let mut engine = TurboQuantEngine::open(db, db, d, 2, 7).unwrap();

    engine
        .insert_with_document(
            "x".into(),
            &make_vec(d, 0.1),
            HashMap::new(),
            Some("first version of the story".into()),
        )
        .unwrap();
    assert_eq!(engine.search_bm25("first", 10, None).unwrap().len(), 1);

    // Upsert: same id, new document — `first` must no longer match this slot.
    engine
        .upsert_with_document(
            "x".into(),
            &make_vec(d, 0.1),
            HashMap::new(),
            Some("second draft completely rewritten".into()),
        )
        .unwrap();
    assert!(engine.search_bm25("first", 10, None).unwrap().is_empty());
    assert_eq!(engine.search_bm25("rewritten", 10, None).unwrap().len(), 1);
}

#[test]
fn bm25_survives_close_reopen() {
    let dir = tempdir().unwrap();
    let db = dir.path().to_str().unwrap();
    let d = 32;

    {
        let mut engine = TurboQuantEngine::open(db, db, d, 4, 99).unwrap();
        for i in 0..20u32 {
            engine
                .insert_with_document(
                    format!("doc-{i}"),
                    &make_vec(d, i as f64 * 0.01),
                    HashMap::new(),
                    Some(format!("token-{i} shared-keyword payload-{i}")),
                )
                .unwrap();
        }
        engine.close().unwrap();
    }

    // Reopen: BM25 must come back populated, either from `bm25.idx` (persistence
    // path) or rebuilt from metadata docs (cold-start fallback). Either way,
    // the user observes a queryable index.
    let engine = TurboQuantEngine::open(db, db, d, 4, 99).unwrap();
    assert_eq!(engine.bm25_doc_count(), 20);
    let r = engine.search_bm25("shared-keyword", 100, None).unwrap();
    assert_eq!(r.len(), 20);

    // Query "token-3" tokenizes to {"token", "3"} after the splitter; "token"
    // matches every doc but "3" only matches doc-3 (and any doc whose payload
    // contains a literal `3`). doc-3 must therefore rank first because it picks
    // up the rare-term IDF on "3" while every other doc only scores "token".
    let r = engine.search_bm25("token-3", 10, None).unwrap();
    assert!(!r.is_empty());
    assert_eq!(r[0].0, "doc-3", "doc-3 must rank first; got {r:?}");
}

#[test]
fn empty_query_returns_empty_results() {
    let dir = tempdir().unwrap();
    let db = dir.path().to_str().unwrap();
    let d = 16;
    let mut engine = TurboQuantEngine::open(db, db, d, 2, 1).unwrap();
    engine
        .insert_with_document(
            "a".into(),
            &make_vec(d, 0.1),
            HashMap::new(),
            Some("hello world".into()),
        )
        .unwrap();
    // Whitespace, punctuation, and the empty string all tokenize to nothing.
    assert!(engine.search_bm25("", 5, None).unwrap().is_empty());
    assert!(engine.search_bm25("   ", 5, None).unwrap().is_empty());
    assert!(engine.search_bm25("???", 5, None).unwrap().is_empty());
}

#[test]
fn bm25_doc_count_is_zero_when_no_documents_inserted() {
    let dir = tempdir().unwrap();
    let db = dir.path().to_str().unwrap();
    let d = 16;
    let mut engine = TurboQuantEngine::open(db, db, d, 2, 1).unwrap();
    engine
        .insert("plain".into(), &make_vec(d, 0.5), HashMap::new())
        .unwrap();
    assert_eq!(engine.bm25_doc_count(), 0);
    assert!(engine.search_bm25("anything", 5, None).unwrap().is_empty());
}
