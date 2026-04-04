use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::time::Instant;
use tqdb::storage::engine::{BatchWriteItem, TurboQuantEngine};

#[test]
#[ignore]
fn bench_batch_crud_large() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let tmp = tempfile::tempdir()?;
    let db_path = tmp.path().to_str().unwrap();
    let mut engine = TurboQuantEngine::open(db_path, db_path, 64, 4, 42)?;

    let total = 2_000usize;
    let batch = 1_000usize;

    let start_insert = Instant::now();
    for chunk in 0..(total / batch) {
        let mut items = Vec::with_capacity(batch);
        for i in 0..batch {
            let idx = chunk * batch + i;
            let mut meta = HashMap::new();
            meta.insert("tenant".to_string(), JsonValue::String("bench".to_string()));
            meta.insert("idx".to_string(), JsonValue::Number((idx as i64).into()));
            items.push(BatchWriteItem {
                id: format!("id-{idx}"),
                vector: (0..64).map(|d| (idx + d) as f32 / 1000.0).collect(),
                metadata: meta,
                document: None,
            });
        }
        engine.insert_many(items)?;
        engine.flush_wal_to_segment()?;
    }
    let insert_elapsed = start_insert.elapsed();

    let start_get = Instant::now();
    let ids: Vec<String> = (0..500).map(|i| format!("id-{i}")).collect();
    let got = engine.get_many(&ids)?;
    let get_elapsed = start_get.elapsed();

    let start_delete = Instant::now();
    let delete_ids: Vec<String> = (0..250).map(|i| format!("id-{i}")).collect();
    let deleted = engine.delete_batch(delete_ids)?;
    engine.flush_wal_to_segment()?;
    let delete_elapsed = start_delete.elapsed();

    println!(
        "batch benchmark: inserted={} in {:?}, get_many(500)={} in {:?}, delete_many(250)={} in {:?}",
        total,
        insert_elapsed,
        got.len(),
        get_elapsed,
        deleted,
        delete_elapsed
    );

    assert_eq!(got.len(), 500);
    assert_eq!(deleted, 250);
    Ok(())
}
