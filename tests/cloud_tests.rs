use ndarray::Array1;
use std::collections::HashMap;
use turboquantdb::storage::engine::TurboQuantEngine;

#[test]
fn test_local_file_uri_storage() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let temp_dir = tempfile::tempdir()?;
    let uri = url::Url::from_file_path(temp_dir.path())
        .unwrap()
        .to_string();
    let local_work_dir = tempfile::tempdir()?;
    let local_path = local_work_dir.path().to_str().unwrap();

    let mut engine = TurboQuantEngine::open(&uri, local_path, 128, 8, 42)?;

    // Insert some data (distinct vectors)
    for i in 0..50 {
        let v = Array1::from_iter((0..128).map(|j| (i + j) as f64 / 1000.0));
        engine.insert(format!("id-{}", i), &v, HashMap::new())?;
    }

    // Flush to segment
    engine.flush_wal_to_segment()?;

    // Verify manifest was uploaded (by reopening)
    drop(engine);

    let engine2 = TurboQuantEngine::open(&uri, local_path, 128, 8, 42)?;
    assert_eq!(engine2.vector_count(), 50);

    // Linear search (no index yet)
    let query = Array1::from_iter((0..128).map(|j| j as f64 / 1000.0));
    let linear_results = engine2.search(&query, 5)?;
    assert_eq!(
        linear_results.len(),
        5,
        "Linear search should return 5 results"
    );
    // With dot product and these vectors, id-49 is the most similar
    assert_eq!(linear_results[0].id, "id-49");

    // Build index and verify upload
    let mut engine2 = engine2;
    engine2.create_index_with_params(32, 200, 64, 1.2, 5)?;

    // ANN Search
    let results = engine2.search(&query, 5)?;
    assert_eq!(results.len(), 5, "ANN search should return 5 results");
    assert_eq!(results[0].id, "id-49");

    Ok(())
}
