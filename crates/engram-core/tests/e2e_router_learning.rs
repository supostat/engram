use std::sync::{Arc, Mutex, RwLock};

use serde_json::json;

use engram_core::config::Config;
use engram_core::dispatch;
use engram_core::indexes::IndexSet;
use engram_core::server::ServerState;
use engram_embeddings::Embedder;
use engram_llm_client::{EmbeddingProvider, TextGenerator};
use engram_router::Router;
use engram_storage::Database;

fn build_deterministic_state() -> Arc<ServerState> {
    let database = Database::in_memory().expect("in-memory database");
    let mut config = Config::default();
    config.embedding.provider = "deterministic".into();
    let indexes = IndexSet::new(|| config.build_hnsw_params()).expect("index set");
    let embedder = Embedder::new(0);
    let router = Router::new(0.1, 0.15);
    let embedding_provider: Arc<dyn EmbeddingProvider + Send + Sync> = Arc::from(
        config
            .build_embedding_provider()
            .expect("embedding provider"),
    );
    let text_generator: Option<Arc<dyn TextGenerator + Send + Sync>> =
        config.build_text_generator().ok().map(Arc::from);
    Arc::new(ServerState {
        database: Mutex::new(database),
        indexes: RwLock::new(indexes),
        embedder,
        router: Mutex::new(router),
        config,
        database_path: String::new(),
        embedding_provider,
        text_generator,
    })
}

// Write-time deduplication folds away near-identical memories, so a learning
// series must store memories that are genuinely distinct in all three fields.
// The deterministic provider concentrates byte signal in low index positions
// and would treat field text differing only by a trailing digit as a ~0.999
// duplicate; using a different keyword per memory keeps every pairwise field
// similarity well below the 0.95 dedup threshold.
const LEARNING_CONTEXTS: [&str; 5] = [
    "configured connection pooling for the primary datastore",
    "tuned the background compaction scheduler thresholds",
    "introduced read-through caching at the gateway edge",
    "partitioned the event log across regional shards",
    "enabled cross-zone replication for durability",
];
const LEARNING_ACTIONS: [&str; 5] = [
    "raised the maximum pool size to fifty handles",
    "lowered the compaction trigger to nightly windows",
    "added a least-recently-used eviction policy",
    "hashed routing keys onto eight independent shards",
    "streamed the write-ahead log to two standby zones",
];
const LEARNING_RESULTS: [&str; 5] = [
    "connection saturation disappeared under peak load",
    "disk amplification dropped by half within a week",
    "gateway tail latency improved noticeably for reads",
    "hotspotting on the busiest partition was eliminated",
    "failover recovery time shrank to a few seconds",
];

#[tokio::test]
async fn router_state_changes_after_judge_series() {
    let state = build_deterministic_state();
    let scores = [0.2, 0.4, 0.6, 0.8, 1.0];
    let mut memory_ids = Vec::new();
    for (index, score) in scores.iter().enumerate() {
        let stored = dispatch::route(
            "memory_store",
            &state,
            json!({
                "memory_type": "decision",
                "context": LEARNING_CONTEXTS[index],
                "action": LEARNING_ACTIONS[index],
                "result": LEARNING_RESULTS[index],
            }),
        )
        .await
        .expect("store should succeed");
        let memory_id = stored["id"].as_str().expect("id").to_string();

        dispatch::route(
            "memory_search",
            &state,
            json!({
                "query": LEARNING_CONTEXTS[index],
                "limit": 5,
            }),
        )
        .await
        .expect("search should succeed");

        dispatch::route(
            "memory_judge",
            &state,
            json!({
                "memory_id": &memory_id,
                "score": *score,
            }),
        )
        .await
        .expect("judge should succeed");

        memory_ids.push(memory_id);
    }

    let status = dispatch::route("memory_status", &state, json!({}))
        .await
        .expect("status should succeed");
    assert_eq!(status["memory_count"], 5);
    assert_eq!(status["indexed_count"], 5);

    let final_search = dispatch::route(
        "memory_search",
        &state,
        json!({
            "query": "datastore connection pooling and replication tuning",
            "limit": 10,
        }),
    )
    .await
    .expect("post-learning search should succeed");
    let results = final_search["results"].as_array().expect("results array");
    assert!(
        !results.is_empty(),
        "search must return results after learning"
    );
}

#[tokio::test]
async fn feedback_tracking_records_searches() {
    let state = build_deterministic_state();

    let stored = dispatch::route(
        "memory_store",
        &state,
        json!({
            "memory_type": "decision",
            "context": "feedback tracking context for search recording",
            "action": "feedback tracking action for search recording",
            "result": "feedback tracking result for search recording",
        }),
    )
    .await
    .expect("store should succeed");
    let memory_id = stored["id"].as_str().expect("id").to_string();

    for _ in 0..3 {
        dispatch::route(
            "memory_search",
            &state,
            json!({
                "query": "feedback tracking context for search recording",
                "limit": 5,
            }),
        )
        .await
        .expect("search should succeed");
    }

    let status_before = dispatch::route("memory_status", &state, json!({}))
        .await
        .expect("status should succeed");
    let pending_before = status_before["pending_judgments"]
        .as_u64()
        .expect("pending_judgments");
    assert!(
        pending_before > 0,
        "searches without judgment should create pending entries"
    );

    dispatch::route(
        "memory_judge",
        &state,
        json!({
            "memory_id": &memory_id,
            "score": 0.9,
        }),
    )
    .await
    .expect("judge should succeed");

    let status_after = dispatch::route("memory_status", &state, json!({}))
        .await
        .expect("status after judge should succeed");
    let pending_after = status_after["pending_judgments"]
        .as_u64()
        .expect("pending_judgments");
    assert!(
        pending_after <= pending_before,
        "judging should not increase pending count"
    );
}
