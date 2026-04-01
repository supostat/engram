use std::sync::{Arc, Mutex};

use serde_json::json;

use engram_core::config::Config;
use engram_core::dispatch;
use engram_core::error::CoreError;
use engram_core::indexes::IndexSet;
use engram_core::server::ServerState;
use engram_embeddings::Embedder;
use engram_router::Router;
use engram_storage::Database;

fn build_test_state() -> Arc<ServerState> {
    let database = Database::in_memory().expect("in-memory database");
    let config = Config::default();
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

fn build_test_state_with_deterministic_embeddings() -> Arc<ServerState> {
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
async fn dispatch_unknown_method_returns_error() {
    let state = build_test_state();
    let result = dispatch::route("nonexistent_method", &state, json!({})).await;
    let error = result.unwrap_err();
    match error {
        CoreError::DispatchError(message) => {
            assert!(message.contains("nonexistent_method"));
        }
        other => panic!("expected DispatchError, got: {other}"),
    }
}

#[tokio::test]
async fn status_handler_returns_counts() {
    let state = build_test_state();
    let result = dispatch::route("memory_status", &state, json!({})).await;
    let data = result.expect("status should succeed");
    assert_eq!(data["memory_count"], 0);
    assert_eq!(data["indexed_count"], 0);
    assert_eq!(data["pending_judgments"], 0);
    assert_eq!(data["index_size"], 0);
}

#[tokio::test]
async fn status_handler_reflects_inserted_memory() {
    let state = build_test_state();
    insert_test_memory(&state, "test-001");
    let result = dispatch::route("memory_status", &state, json!({})).await;
    let data = result.expect("status should succeed");
    assert_eq!(data["memory_count"], 1);
}

#[tokio::test]
async fn consolidate_preview_returns_empty_on_fresh_database() {
    let state = build_test_state();
    let params = json!({ "stale_days": 90, "min_score": 0.3 });
    let result = dispatch::route("memory_consolidate_preview", &state, params).await;
    let data = result.expect("preview should succeed");
    assert_eq!(data["duplicates"], 0);
    assert_eq!(data["stale"], 0);
    assert_eq!(data["garbage"], 0);
}

#[tokio::test]
async fn judge_handler_explicit_score_updates_memory() {
    let state = build_test_state();
    insert_test_memory(&state, "judge-test-001");
    let params = json!({
        "memory_id": "judge-test-001",
        "score": 0.85,
    });
    let result = dispatch::route("memory_judge", &state, params).await;
    let data = result.expect("judge should succeed");
    let returned_score = data["score"].as_f64().expect("score is a number");
    assert!((returned_score - 0.85).abs() < 0.001);
    assert_eq!(data["degraded"], false);
    let database = state.database.lock().unwrap();
    let memory = database.get_memory("judge-test-001").expect("memory exists");
    assert!((memory.score - 0.85).abs() < 0.001);
}

#[tokio::test]
async fn judge_handler_missing_memory_returns_error() {
    let state = build_test_state();
    let params = json!({
        "memory_id": "nonexistent",
        "score": 0.5,
    });
    let result = dispatch::route("memory_judge", &state, params).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn store_handler_rejects_invalid_params() {
    let state = build_test_state();
    let params = json!({ "invalid": "data" });
    let result = dispatch::route("memory_store", &state, params).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn search_handler_rejects_missing_query() {
    let state = build_test_state();
    let params = json!({ "limit": 5 });
    let result = dispatch::route("memory_search", &state, params).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn store_handler_creates_memory() {
    let state = build_test_state_with_deterministic_embeddings();
    let params = json!({
        "memory_type": "decision",
        "context": "user configured the database connection for production deployment",
        "action": "set up connection pooling with maximum fifty connections per instance",
        "result": "database connections are now stable and performant under load",
    });
    let result = dispatch::route("memory_store", &state, params).await;
    let data = result.expect("store should succeed");
    let memory_id = data["id"].as_str().expect("id is a string");
    assert!(!memory_id.is_empty());
    assert_eq!(data["indexed"], true);

    let database = state.database.lock().unwrap();
    let memory = database.get_memory(memory_id).expect("memory exists in database");
    assert_eq!(memory.memory_type, "decision");
    assert_eq!(memory.context, "user configured the database connection for production deployment");
    assert!(memory.indexed);
}

#[tokio::test]
async fn search_handler_finds_stored_memory() {
    let state = build_test_state_with_deterministic_embeddings();
    let store_params = json!({
        "memory_type": "decision",
        "context": "the deployment pipeline runs automated integration tests before merging",
        "action": "configured continuous integration with three parallel test runners",
        "result": "all tests pass within five minutes on every pull request submission",
    });
    let store_result = dispatch::route("memory_store", &state, store_params).await;
    let stored = store_result.expect("store should succeed");
    let stored_id = stored["id"].as_str().expect("stored id");

    let search_params = json!({
        "query": "deployment pipeline integration tests continuous integration runners",
        "limit": 10,
    });
    let search_result = dispatch::route("memory_search", &state, search_params).await;
    let results = search_result.expect("search should succeed");
    let results_array = results.as_array().expect("results is an array");

    let found = results_array.iter().any(|entry| entry["id"] == stored_id);
    assert!(found, "stored memory should appear in search results");
}

fn insert_test_memory(state: &Arc<ServerState>, id: &str) {
    let memory = engram_storage::Memory {
        id: id.to_string(),
        memory_type: "decision".to_string(),
        context: "test context".to_string(),
        action: "test action".to_string(),
        result: "test result".to_string(),
        score: 0.0,
        embedding_context: None,
        embedding_action: None,
        embedding_result: None,
        indexed: false,
        tags: None,
        project: None,
        parent_id: None,
        source_ids: None,
        insight_type: None,
        created_at: "2025-01-01T00:00:00Z".to_string(),
        updated_at: "2025-01-01T00:00:00Z".to_string(),
        used_count: 0,
        last_used_at: None,
        superseded_by: None,
    };
    let database = state.database.lock().unwrap();
    database.insert_memory(&memory).expect("insert memory");
}
