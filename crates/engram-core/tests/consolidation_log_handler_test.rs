use std::sync::{Arc, Mutex, RwLock};

use serde_json::json;

use engram_core::config::Config;
use engram_core::dispatch;
use engram_core::indexes::IndexSet;
use engram_core::lock_helpers;
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

fn memory_count(state: &Arc<ServerState>) -> i64 {
    let database = lock_helpers::lock_db(state);
    database
        .connection()
        .query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))
        .expect("count memories")
}

async fn store_pair_sharing_context(state: &Arc<ServerState>) {
    let shared_context = "consolidation log shared context tokens";
    dispatch::route(
        "memory_store",
        state,
        json!({
            "memory_type": "decision",
            "context": shared_context,
            "action": "first distinct action describing one specific operational procedure",
            "result": "first distinct result capturing a particular measured production outcome",
        }),
    )
    .await
    .expect("first store");
    dispatch::route(
        "memory_store",
        state,
        json!({
            "memory_type": "decision",
            "context": shared_context,
            "action": "second distinct action describing an entirely different remediation workflow",
            "result": "second distinct result documenting an unrelated downstream metric recovery",
        }),
    )
    .await
    .expect("second store");
}

#[tokio::test]
async fn consolidation_log_empty_returns_zero_count() {
    let state = build_deterministic_state();
    let result = dispatch::route("memory_consolidate_log", &state, json!({}))
        .await
        .expect("log should succeed");
    assert_eq!(result["count"], 0);
    assert!(result["log"].as_array().expect("array").is_empty());
}

#[tokio::test]
async fn consolidation_log_records_a_real_merge() {
    let state = build_deterministic_state();
    store_pair_sharing_context(&state).await;
    let before = memory_count(&state);

    dispatch::route("memory_consolidate_apply", &state, json!({}))
        .await
        .expect("consolidate apply succeeds");

    let result = dispatch::route("memory_consolidate_log", &state, json!({}))
        .await
        .expect("log should succeed");

    let count = result["count"].as_u64().expect("count is a number");
    assert!(count >= 1, "a merge must produce at least one log entry");
    let entries = result["log"].as_array().expect("log is an array");
    let newest = &entries[0];
    assert_eq!(newest["action"], "merge");
    assert!(
        newest["memory_ids"].is_array(),
        "memory_ids must be a JSON array, not a stringified blob"
    );

    // Reading the log is read-only: no rows created or removed.
    assert_eq!(
        memory_count(&state),
        before,
        "listing the consolidation log must not mutate the memory store"
    );
}

#[tokio::test]
async fn consolidation_log_honors_limit() {
    let state = build_deterministic_state();
    store_pair_sharing_context(&state).await;
    dispatch::route("memory_consolidate_apply", &state, json!({}))
        .await
        .expect("first apply");

    let result = dispatch::route("memory_consolidate_log", &state, json!({ "limit": 1 }))
        .await
        .expect("log should succeed");
    assert_eq!(result["count"], 1);
    assert_eq!(result["log"].as_array().expect("array").len(), 1);
}
