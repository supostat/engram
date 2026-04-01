use criterion::{Criterion, criterion_group, criterion_main};
use engram_storage::{Database, Memory};

fn make_memory(id: &str) -> Memory {
    Memory {
        id: id.to_string(),
        memory_type: "bugfix".to_string(),
        context: format!("Context for memory {id}"),
        action: format!("Action taken for {id}"),
        result: format!("Result achieved for {id}"),
        score: 0.5,
        embedding_context: None,
        embedding_action: None,
        embedding_result: None,
        indexed: false,
        tags: Some("[\"test\"]".to_string()),
        project: Some("bench".to_string()),
        parent_id: None,
        source_ids: None,
        insight_type: None,
        created_at: "2025-01-01T00:00:00Z".to_string(),
        updated_at: "2025-01-01T00:00:00Z".to_string(),
        used_count: 0,
        last_used_at: None,
        superseded_by: None,
    }
}

fn bench_insert_memory(criterion: &mut Criterion) {
    criterion.bench_function("insert_memory", |bencher| {
        let database = Database::in_memory().unwrap();
        let mut counter = 0u64;
        bencher.iter(|| {
            counter += 1;
            let memory = make_memory(&format!("bench-{counter}"));
            database.insert_memory(&memory).unwrap();
        });
    });
}

fn bench_batch_insert_1k(criterion: &mut Criterion) {
    criterion.bench_function("batch_insert_1k", |bencher| {
        bencher.iter(|| {
            let database = Database::in_memory().unwrap();
            let memories: Vec<Memory> = (0..1000)
                .map(|i| make_memory(&format!("batch-{i}")))
                .collect();
            database.bulk_insert_memories(&memories).unwrap();
        });
    });
}

fn bench_fts_search_10k(criterion: &mut Criterion) {
    let database = Database::in_memory().unwrap();
    let memories: Vec<Memory> = (0..10_000)
        .map(|i| {
            let mut memory = make_memory(&format!("fts-{i}"));
            if i % 100 == 0 {
                memory.context = format!("Special keyword findme in memory {i}");
            }
            memory
        })
        .collect();
    database.bulk_insert_memories(&memories).unwrap();

    criterion.bench_function("fts_search_10k", |bencher| {
        bencher.iter(|| database.search_fts("findme", 5).unwrap());
    });
}

fn bench_full_table_scan_10k(criterion: &mut Criterion) {
    let database = Database::in_memory().unwrap();
    let memories: Vec<Memory> = (0..10_000)
        .map(|i| make_memory(&format!("scan-{i}")))
        .collect();
    database.bulk_insert_memories(&memories).unwrap();

    criterion.bench_function("full_table_scan_10k", |bencher| {
        bencher.iter(|| database.get_unindexed_memories(10_000).unwrap());
    });
}

criterion_group!(
    benches,
    bench_insert_memory,
    bench_batch_insert_1k,
    bench_fts_search_10k,
    bench_full_table_scan_10k
);
criterion_main!(benches);
