use ndarray::Array1;
use serde_json::json;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;
use tqdb::storage::engine::TurboQuantEngine;

#[test]
fn test_benchmark_compaction_and_search() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let total_vectors = std::env::var("TQ_BENCH_TOTAL_VECTORS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(500);
    let batches = std::env::var("TQ_BENCH_BATCHES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(5)
        .max(1);
    let per_batch = total_vectors.div_ceil(batches);

    let tmp = tempfile::tempdir()?;
    let local_path = tmp.path().to_str().unwrap();
    let uri = format!("file://{}", local_path.replace("\\", "/"));

    let mut engine = TurboQuantEngine::open(&uri, local_path, 128, 8, 42)?;

    println!("Inserting {total_vectors} vectors in {batches} batches...");
    let start_insert = Instant::now();
    let mut inserted = 0usize;
    for batch in 0..batches {
        for i in 0..per_batch {
            if inserted >= total_vectors {
                break;
            }
            let id = format!("id-{}-{}", batch, i);
            let mut v = Array1::zeros(128);
            v[0] = inserted as f64;
            engine.insert(id, &v, HashMap::new())?;
            inserted += 1;
        }
        engine.flush_wal_to_segment()?;
        if inserted >= total_vectors {
            break;
        }
    }
    println!("Insertion took: {:?}", start_insert.elapsed());
    let insert_total_ms = start_insert.elapsed().as_secs_f64() * 1000.0;

    let query = Array1::zeros(128);

    // 1. Brute-force Search (10 segments)
    let start_brute = Instant::now();
    let results_brute = engine.search(&query, 10)?;
    let elapsed_brute = start_brute.elapsed();
    println!("Brute-force (10 segments) took: {:?}", elapsed_brute);
    assert_eq!(results_brute.len(), 10);

    // 2. Flush WAL to segment
    println!("Flushing WAL to segment...");
    let start_flush = Instant::now();
    engine.flush_wal_to_segment()?;
    println!("Flush took: {:?}", start_flush.elapsed());
    let compaction_ms = start_flush.elapsed().as_secs_f64() * 1000.0;

    // 3. Brute-force Search (1 segment)
    let start_brute_post = Instant::now();
    let _ = engine.search(&query, 10)?;
    let elapsed_brute_post = start_brute_post.elapsed();
    println!("Brute-force (1 segment) took: {:?}", elapsed_brute_post);

    // 4. Indexing
    println!("Building Vamana Index...");
    let start_index = Instant::now();
    engine.create_index_with_params(32, 64, 64, 1.2, 5)?;
    println!("Indexing took: {:?}", start_index.elapsed());
    let indexing_ms = start_index.elapsed().as_secs_f64() * 1000.0;

    // 5. ANN Search
    let start_ann = Instant::now();
    for _ in 0..100 {
        let _ = engine.search(&query, 10)?;
    }
    let elapsed_ann = start_ann.elapsed() / 100;
    println!("ANN Search (average) took: {:?}", elapsed_ann);
    let ann_avg_ms = elapsed_ann.as_secs_f64() * 1000.0;

    let stats = engine.stats();
    let insertion_throughput = inserted as f64 / (insert_total_ms / 1000.0);

    let artifact_dir =
        std::env::var("TQ_BENCH_ARTIFACT_DIR").unwrap_or_else(|_| "target/benchmarks".to_string());
    fs::create_dir_all(&artifact_dir)?;
    let artifact_path = PathBuf::from(artifact_dir).join("bench_search.json");

    let report = json!({
        "benchmark": "bench_search",
        "vectors": inserted,
        "dimension": 128,
        "bits": 8,
        "insert_total_ms": insert_total_ms,
        "insert_throughput_vectors_per_sec": insertion_throughput,
        "brute_ms_pre_compaction": elapsed_brute.as_secs_f64() * 1000.0,
        "compaction_ms": compaction_ms,
        "brute_ms_post_compaction": elapsed_brute_post.as_secs_f64() * 1000.0,
        "indexing_ms": indexing_ms,
        "ann_avg_ms": ann_avg_ms,
        "stats": {
            "vector_count": stats.vector_count,
            "segment_count": stats.segment_count,
            "has_index": stats.has_index,
            "index_nodes": stats.index_nodes,
            "total_disk_bytes": stats.total_disk_bytes,
        }
    });
    fs::write(&artifact_path, serde_json::to_vec_pretty(&report)?)?;
    println!("Wrote benchmark artifact: {}", artifact_path.display());

    Ok(())
}
