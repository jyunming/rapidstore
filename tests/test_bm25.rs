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
fn hybrid_recovers_keyword_dense_misses() {
    // Construct a corpus where the dense-best doc has no rare term and the
    // rare-term doc is far in dense space. Pure dense → A; pure BM25 → B;
    // hybrid should surface B alongside A because the rare-term IDF dominates
    // the BM25 ranking and RRF rewards the consensus.
    fn aligned(d: usize, seed: f64) -> Array1<f64> {
        // First component drives IP; remaining dims are tiny noise.
        let mut v = Array1::<f64>::zeros(d);
        v[0] = seed;
        for j in 1..d {
            v[j] = (j as f64) * 1e-6;
        }
        v
    }

    let dir = tempdir().unwrap();
    let db = dir.path().to_str().unwrap();
    let d = 32;
    let mut engine = TurboQuantEngine::open(db, db, d, 4, 99).unwrap();

    // Doc-A: aligned with the query (high IP). No rare word.
    engine
        .insert_with_document(
            "doc-A".into(),
            &aligned(d, 1.0),
            HashMap::new(),
            Some("ordinary common everyday text content".into()),
        )
        .unwrap();
    // Doc-B: nearly orthogonal to the query (tiny IP). Carries the rare token.
    let mut doc_b_vec = Array1::<f64>::zeros(d);
    doc_b_vec[1] = 1.0;
    engine
        .insert_with_document(
            "doc-B".into(),
            &doc_b_vec,
            HashMap::new(),
            Some("ordinary common rare-token-xyz appears here".into()),
        )
        .unwrap();
    // Filler docs in between so the rare term has meaningful IDF.
    for i in 0..5u32 {
        let mut v = Array1::<f64>::zeros(d);
        v[2 + (i as usize) % (d - 2)] = 1.0;
        engine
            .insert_with_document(
                format!("filler-{i}"),
                &v,
                HashMap::new(),
                Some(format!("ordinary common filler-{i} payload")),
            )
            .unwrap();
    }

    let q = aligned(d, 1.0);
    let dense_only = engine.search(&q, 5).unwrap();
    assert_eq!(
        dense_only[0].id, "doc-A",
        "sanity: dense-only top-1 must be doc-A; got {dense_only:?}"
    );

    let hybrid = engine
        .search_hybrid(
            &q,
            "rare-token-xyz",
            5,
            None,
            None,
            false,
            None,
            Some(0.5),
            Some(60.0),
            Some(4),
            true,
            true,
            true,
        )
        .unwrap();
    let hybrid_ids: Vec<&str> = hybrid.iter().map(|r| r.id.as_str()).collect();
    assert!(
        hybrid_ids.contains(&"doc-B"),
        "hybrid must surface doc-B (the only carrier of the rare token); got {hybrid_ids:?}"
    );
}

#[test]
fn hybrid_weight_zero_matches_dense_only() {
    // weight=0.0 collapses RRF to the dense list alone, so the top hit must
    // match what `search()` returns. (Order below the top may differ because
    // RRF still ranks consensus picks, but the top-1 should be stable.)
    let dir = tempdir().unwrap();
    let db = dir.path().to_str().unwrap();
    let d = 16;
    let mut engine = TurboQuantEngine::open(db, db, d, 2, 7).unwrap();
    for i in 0..6u32 {
        engine
            .insert_with_document(
                format!("v{i}"),
                &make_vec(d, i as f64 * 0.1),
                HashMap::new(),
                Some(format!("text {i}")),
            )
            .unwrap();
    }
    let q = make_vec(d, 0.0);
    let dense = engine.search(&q, 3).unwrap();
    let hybrid = engine
        .search_hybrid(
            &q,
            "anything",
            3,
            None,
            None,
            false,
            None,
            Some(0.0),
            Some(60.0),
            Some(4),
            true,
            true,
            true,
        )
        .unwrap();
    assert_eq!(hybrid[0].id, dense[0].id);
}

#[test]
fn slot_reuse_does_not_leak_doc() {
    // Insert id "alice" with doc-A, delete alice, then insert "bob" with doc-B.
    // Bob may end up in the same slot alice vacated; whether or not the slot is
    // physically reused, BM25 must not associate alice's tokens with bob.
    let dir = tempdir().unwrap();
    let db = dir.path().to_str().unwrap();
    let d = 16;
    let mut engine = TurboQuantEngine::open(db, db, d, 2, 5).unwrap();

    // Tokens must be single alphanumeric words; the tokenizer splits on
    // non-alphanumeric chars, so "alpha-uniq" would become ["alpha", "uniq"]
    // and any shared substring would create false positives.
    engine
        .insert_with_document(
            "alice".into(),
            &make_vec(d, 0.1),
            HashMap::new(),
            Some("alphaword alicemarker".into()),
        )
        .unwrap();
    engine.delete("alice".into()).unwrap();
    engine
        .insert_with_document(
            "bob".into(),
            &make_vec(d, 0.2),
            HashMap::new(),
            Some("betaword bobmarker".into()),
        )
        .unwrap();

    // Alice's tokens must not surface bob.
    let r_alpha = engine.search_bm25("alphaword", 10, None).unwrap();
    let alpha_ids: Vec<&str> = r_alpha.iter().map(|(id, _)| id.as_str()).collect();
    assert!(
        !alpha_ids.contains(&"bob"),
        "bob inherited alice's tokens; alphaword returned {alpha_ids:?}"
    );
    let r_marker = engine.search_bm25("alicemarker", 10, None).unwrap();
    assert!(
        r_marker.is_empty(),
        "alicemarker should hit nothing after alice was deleted; got {r_marker:?}"
    );
    // Bob's tokens must work normally.
    let r_beta = engine.search_bm25("betaword", 10, None).unwrap();
    let beta_ids: Vec<&str> = r_beta.iter().map(|(id, _)| id.as_str()).collect();
    assert!(
        beta_ids.contains(&"bob"),
        "betaword did not return bob; got {beta_ids:?}"
    );
}

#[test]
fn cold_start_rebuild_matches_persisted() {
    // Open, insert, close → bm25.idx is persisted. Reopen capture A.
    // Delete bm25.idx, reopen → rebuild from metadata.iter_docs(). Capture B.
    // The two captures must yield the same top-K IDs in the same order.
    let dir = tempdir().unwrap();
    let db = dir.path().to_str().unwrap();
    let d = 16;
    {
        let mut engine = TurboQuantEngine::open(db, db, d, 2, 99).unwrap();
        for i in 0..20u32 {
            engine
                .insert_with_document(
                    format!("doc-{i}"),
                    &make_vec(d, i as f64 * 0.01),
                    HashMap::new(),
                    Some(format!("shared-tag rare-{i} mid-{}", i % 3)),
                )
                .unwrap();
        }
        engine.close().unwrap();
    }

    // Reopen with persisted bm25.idx — the "fast path".
    let persisted_results = {
        let engine = TurboQuantEngine::open(db, db, d, 2, 99).unwrap();
        engine.search_bm25("shared-tag mid-1", 10, None).unwrap()
    };

    // Now delete bm25.idx and reopen — forces the cold-start rebuild path.
    let bm25_path = dir.path().join("bm25.idx");
    assert!(
        bm25_path.exists(),
        "bm25.idx should have been written by close()"
    );
    std::fs::remove_file(&bm25_path).unwrap();

    let rebuilt_results = {
        let engine = TurboQuantEngine::open(db, db, d, 2, 99).unwrap();
        engine.search_bm25("shared-tag mid-1", 10, None).unwrap()
    };

    // Compare by ID→score map: the partial sort breaks ties unstably, so the
    // exact ordering may differ across runs even when the underlying index is
    // identical. The right correctness check is "same docs surface, with the
    // same scores."
    let p_map: std::collections::HashMap<String, f32> = persisted_results
        .iter()
        .map(|(s, sc)| (s.clone(), *sc))
        .collect();
    let r_map: std::collections::HashMap<String, f32> = rebuilt_results
        .iter()
        .map(|(s, sc)| (s.clone(), *sc))
        .collect();

    // Every doc in the persisted result whose score is strictly above the
    // tie threshold (the lowest score in the rebuilt set) must also appear
    // in the rebuilt set with the same score. Tied docs at the cutoff may
    // swap which side of top-10 they land on; that's acceptable.
    let cutoff_rebuilt = rebuilt_results.last().map(|(_, s)| *s).unwrap_or(0.0);
    for (id_p, score_p) in &p_map {
        if *score_p > cutoff_rebuilt + 1e-5 {
            let score_r = r_map.get(id_p).unwrap_or_else(|| {
                panic!(
                    "doc {id_p} (score {score_p}) is strictly above cutoff {cutoff_rebuilt} \
                     but missing from rebuilt set; rebuild must have produced different data"
                );
            });
            assert!(
                (score_p - score_r).abs() < 1e-5,
                "score drift on {id_p}: persisted={score_p} rebuilt={score_r}"
            );
        }
    }
    // And the top-1 must be identical (no ties at the very top in this corpus).
    assert_eq!(
        persisted_results[0].0, rebuilt_results[0].0,
        "top-1 differs: persisted={} rebuilt={}",
        persisted_results[0].0, rebuilt_results[0].0
    );
}

#[test]
fn compaction_with_docs_preserves_search() {
    // Insert 20 docs with documents, delete half, force a compaction via
    // flush_wal_to_segment (which calls live_compact_slab when has_pending_deletes).
    // After compaction, BM25 has been remapped/rebuilt; search results from the
    // surviving docs must still be returnable by their unique tokens.
    let dir = tempdir().unwrap();
    let db = dir.path().to_str().unwrap();
    let d = 16;
    let mut engine = TurboQuantEngine::open(db, db, d, 2, 7).unwrap();

    for i in 0..20u32 {
        engine
            .insert_with_document(
                format!("d{i}"),
                &make_vec(d, i as f64 * 0.01),
                HashMap::new(),
                Some(format!("common-word unique-tag-{i}")),
            )
            .unwrap();
    }
    // Delete every other doc.
    for i in (0..20u32).step_by(2) {
        let ok = engine.delete(format!("d{i}")).unwrap();
        assert!(ok, "delete d{i} returned false");
    }

    // Force compaction. This must run live_compact_slab under the hood
    // (has_pending_deletes is set by delete()).
    engine.flush_wal_to_segment().unwrap();

    // Surviving docs (odd-numbered) must still be findable by their unique tag.
    let r = engine.search_bm25("unique-tag-7", 10, None).unwrap();
    let ids: Vec<&str> = r.iter().map(|(id, _)| id.as_str()).collect();
    assert!(
        ids.contains(&"d7"),
        "d7 should survive compaction and be findable; got {ids:?}"
    );
    // Deleted docs must not surface.
    let r0 = engine.search_bm25("unique-tag-0", 10, None).unwrap();
    let ids0: Vec<&str> = r0.iter().map(|(id, _)| id.as_str()).collect();
    assert!(
        !ids0.contains(&"d0"),
        "d0 was deleted before compaction but still appears in search; got {ids0:?}"
    );
}

#[test]
fn hybrid_respects_metadata_filter_on_bm25_only_slots() {
    // Regression: search_hybrid used to pass `bm25_filter=None` AND skip the
    // post-filter, so a slot found only by the BM25 leg could leak into results
    // even when its metadata violated the user's filter. Verify it doesn't.
    let dir = tempdir().unwrap();
    let db = dir.path().to_str().unwrap();
    let d = 16;
    let mut engine = TurboQuantEngine::open(db, db, d, 2, 13).unwrap();

    // Two docs share a rare BM25 token but differ in metadata.
    // Doc-allow has color=red and a vector aligned with the query.
    // Doc-deny has color=blue and an orthogonal vector — only BM25 will surface it.
    let mut q = Array1::<f64>::zeros(d);
    q[0] = 1.0;
    let mut allow_v = Array1::<f64>::zeros(d);
    allow_v[0] = 0.99;
    let mut deny_v = Array1::<f64>::zeros(d);
    deny_v[1] = 1.0;

    let mut allow_meta = HashMap::new();
    allow_meta.insert("color".to_string(), serde_json::Value::String("red".into()));
    let mut deny_meta = HashMap::new();
    deny_meta.insert(
        "color".to_string(),
        serde_json::Value::String("blue".into()),
    );

    engine
        .insert_with_document(
            "allow".into(),
            &allow_v,
            allow_meta,
            Some("rare-shared-tag allow-marker".into()),
        )
        .unwrap();
    engine
        .insert_with_document(
            "deny".into(),
            &deny_v,
            deny_meta,
            Some("rare-shared-tag deny-marker".into()),
        )
        .unwrap();

    // Filter on color=red. The dense leg will surface "allow" (and only allow,
    // because the filter is enforced in the dense pipeline). BM25 will surface
    // both; without the post-filter, "deny" would slip through into the fused
    // output even though its metadata fails the predicate.
    let mut filter = HashMap::new();
    filter.insert("color".to_string(), serde_json::json!("red"));
    let r = engine
        .search_hybrid(
            &q,
            "rare-shared-tag",
            10,
            Some(&filter),
            None,
            false,
            None,
            Some(0.5),
            Some(60.0),
            Some(4),
            true,
            true,
            true,
        )
        .unwrap();
    let ids: Vec<&str> = r.iter().map(|x| x.id.as_str()).collect();
    assert!(
        ids.contains(&"allow"),
        "allow must pass the filter; got {ids:?}"
    );
    assert!(
        !ids.contains(&"deny"),
        "deny must be filtered out even though BM25 matches its document; got {ids:?}"
    );
}

#[test]
fn hybrid_with_empty_bm25_falls_back_to_dense() {
    // Database with vectors but zero documents → BM25 is empty. search_hybrid
    // must not panic; it should return the dense leg's hits.
    let dir = tempdir().unwrap();
    let db = dir.path().to_str().unwrap();
    let d = 16;
    let mut engine = TurboQuantEngine::open(db, db, d, 2, 11).unwrap();
    for i in 0..5u32 {
        // Insert without a document.
        engine
            .insert(
                format!("v{i}"),
                &make_vec(d, i as f64 * 0.05),
                HashMap::new(),
            )
            .unwrap();
    }
    assert_eq!(engine.bm25_doc_count(), 0);

    let q = make_vec(d, 0.0);
    let results = engine
        .search_hybrid(
            &q,
            "anything",
            3,
            None,
            None,
            false,
            None,
            Some(0.5),
            Some(60.0),
            Some(4),
            true,
            true,
            true,
        )
        .unwrap();
    // Empty BM25 contributes nothing to RRF; output is the dense leg's top-K.
    assert!(
        !results.is_empty(),
        "hybrid with empty BM25 should still return dense results"
    );
    assert!(results.len() <= 3);
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
