use std::sync::{Arc, Mutex};

use serde_json::json;

use engram_core::config::Config;
use engram_core::dispatch;
use engram_core::indexes::IndexSet;
use engram_core::server::ServerState;
use engram_embeddings::Embedder;
use engram_router::Router;
use engram_storage::Database;

fn build_deterministic_state() -> Arc<ServerState> {
    let database = Database::in_memory().expect("in-memory database");
    let mut config = Config::default();
    config.embedding.provider = "deterministic".into();
    let indexes = IndexSet::new(|| config.build_hnsw_params()).expect("index set");
    let embedder = Embedder::new();
    let router = Router::new(0.1, 0.15);
    Arc::new(ServerState {
        database: Mutex::new(database),
        indexes: Mutex::new(indexes),
        embedder: Mutex::new(embedder),
        router: Mutex::new(router),
        config,
    })
}

#[tokio::test]
async fn insights_list_empty() {
    let state = build_deterministic_state();
    let result = dispatch::route(
        "memory_insights",
        &state,
        json!({"action": "list"}),
    )
    .await;
    let data = result.expect("list should succeed");
    assert_eq!(data["count"], 0);
    assert!(data["insights"].as_array().expect("array").is_empty());
}

#[tokio::test]
async fn insights_generate_returns_stub_error() {
    let state = build_deterministic_state();
    let result = dispatch::route(
        "memory_insights",
        &state,
        json!({"action": "generate"}),
    )
    .await;
    let error = result.expect_err("generate should fail with stub");
    assert!(error.to_string().contains("engram-trainer"));
}

#[tokio::test]
async fn insights_delete_nonexistent_returns_error() {
    let state = build_deterministic_state();
    let result = dispatch::route(
        "memory_insights",
        &state,
        json!({"action": "delete", "id": "nonexistent-id"}),
    )
    .await;
    let error = result.expect_err("delete nonexistent should fail");
    assert!(error.to_string().contains("not found"));
}

#[tokio::test]
async fn insights_delete_missing_id_returns_error() {
    let state = build_deterministic_state();
    let result = dispatch::route(
        "memory_insights",
        &state,
        json!({"action": "delete"}),
    )
    .await;
    let error = result.expect_err("missing id should fail");
    assert!(error.to_string().contains("[6007]"));
}

#[tokio::test]
async fn insights_invalid_action_returns_error() {
    let state = build_deterministic_state();
    let result = dispatch::route(
        "memory_insights",
        &state,
        json!({"action": "unknown"}),
    )
    .await;
    let error = result.expect_err("unknown action should fail");
    assert!(error.to_string().contains("invalid insights action"));
}

#[tokio::test]
async fn insights_list_returns_only_insight_type() {
    let state = build_deterministic_state();

    let store_params = json!({
        "memory_type": "decision",
        "context": "regular memory context",
        "action": "regular memory action",
        "result": "regular memory result",
    });
    dispatch::route("memory_store", &state, store_params)
        .await
        .expect("store should succeed");

    let result = dispatch::route(
        "memory_insights",
        &state,
        json!({"action": "list"}),
    )
    .await;
    let data = result.expect("list should succeed");
    assert_eq!(data["count"], 0);
}
