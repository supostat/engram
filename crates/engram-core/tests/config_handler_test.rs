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
async fn config_get_returns_sanitized_config() {
    let state = build_deterministic_state();
    let result = dispatch::route("memory_config", &state, json!({"action": "get"})).await;
    let data = result.expect("get should succeed");
    assert!(data["database"]["path"].is_string());
    assert!(data["embedding"]["provider"].is_string());
    assert!(data["embedding"].get("api_key").is_none());
    assert_eq!(data["embedding"]["has_api_key"], false);
    assert!(data["llm"]["provider"].is_string());
    assert!(data["llm"].get("api_key").is_none());
    assert_eq!(data["llm"]["has_api_key"], false);
    assert!(data["server"]["socket_path"].is_string());
    assert!(data["hnsw"]["max_connections"].is_u64());
    assert!(data["consolidation"]["stale_days"].is_u64());
}

#[tokio::test]
async fn config_get_reports_api_key_presence() {
    let mut config = Config::default();
    config.embedding.provider = "deterministic".into();
    config.embedding.api_key = Some("test-voyage-key".into());
    config.llm.api_key = Some("test-openai-key".into());
    let database = Database::in_memory().expect("in-memory database");
    let indexes = IndexSet::new(|| config.build_hnsw_params()).expect("index set");
    let state = Arc::new(ServerState {
        database: Mutex::new(database),
        indexes: Mutex::new(indexes),
        embedder: Mutex::new(Embedder::new()),
        router: Mutex::new(Router::new(0.1, 0.15)),
        config,
    });
    let data = dispatch::route("memory_config", &state, json!({"action": "get"}))
        .await
        .expect("get should succeed");
    assert_eq!(data["embedding"]["has_api_key"], true);
    assert_eq!(data["llm"]["has_api_key"], true);
    assert!(data["embedding"].get("api_key").is_none());
    assert!(data["llm"].get("api_key").is_none());
}

#[tokio::test]
async fn config_set_returns_read_only_error() {
    let state = build_deterministic_state();
    let result = dispatch::route(
        "memory_config",
        &state,
        json!({"action": "set", "key": "foo", "value": "bar"}),
    )
    .await;
    let error = result.expect_err("set should fail");
    assert!(error.to_string().contains("[6008]"));
}

#[tokio::test]
async fn config_invalid_action_returns_dispatch_error() {
    let state = build_deterministic_state();
    let result = dispatch::route("memory_config", &state, json!({"action": "delete"})).await;
    let error = result.expect_err("invalid action should fail");
    assert!(error.to_string().contains("[6007]"));
    assert!(error.to_string().contains("invalid config action"));
}

#[tokio::test]
async fn config_missing_action_returns_dispatch_error() {
    let state = build_deterministic_state();
    let result = dispatch::route("memory_config", &state, json!({})).await;
    let error = result.expect_err("missing action should fail");
    assert!(error.to_string().contains("[6007]"));
}
