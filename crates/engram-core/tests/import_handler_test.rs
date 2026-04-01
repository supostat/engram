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
async fn import_single_memory() {
    let state = build_deterministic_state();
    let params = json!({
        "version": 1,
        "memories": [{
            "id": "imported-001",
            "memory_type": "decision",
            "context": "imported context",
            "action": "imported action",
            "result": "imported result",
            "score": 0.5,
            "tags": "import",
            "project": "engram",
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z",
            "used_count": 0,
        }]
    });
    let result = dispatch::route("memory_import", &state, params).await;
    let data = result.expect("import should succeed");
    assert_eq!(data["imported"], 1);
    assert_eq!(data["skipped"], 0);
}

#[tokio::test]
async fn import_skips_duplicate_ids() {
    let state = build_deterministic_state();
    let store_params = json!({
        "memory_type": "decision",
        "context": "existing context",
        "action": "existing action",
        "result": "existing result",
    });
    let stored = dispatch::route("memory_store", &state, store_params)
        .await
        .expect("store should succeed");
    let existing_id = stored["id"].as_str().expect("id").to_string();

    let params = json!({
        "version": 1,
        "memories": [{
            "id": existing_id,
            "memory_type": "decision",
            "context": "duplicate context",
            "action": "duplicate action",
            "result": "duplicate result",
            "score": 0.0,
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z",
            "used_count": 0,
        }]
    });
    let result = dispatch::route("memory_import", &state, params).await;
    let data = result.expect("import should succeed");
    assert_eq!(data["imported"], 0);
    assert_eq!(data["skipped"], 1);
}

#[tokio::test]
async fn import_wrong_version_returns_error() {
    let state = build_deterministic_state();
    let params = json!({
        "version": 99,
        "memories": []
    });
    let result = dispatch::route("memory_import", &state, params).await;
    let error = result.expect_err("wrong version should fail");
    assert!(error.to_string().contains("[6010]"));
}

#[tokio::test]
async fn import_empty_memories_array() {
    let state = build_deterministic_state();
    let params = json!({
        "version": 1,
        "memories": []
    });
    let result = dispatch::route("memory_import", &state, params).await;
    let data = result.expect("empty import should succeed");
    assert_eq!(data["imported"], 0);
    assert_eq!(data["skipped"], 0);
}

#[tokio::test]
async fn import_roundtrip_with_export() {
    let state = build_deterministic_state();
    let store_params = json!({
        "memory_type": "decision",
        "context": "roundtrip context",
        "action": "roundtrip action",
        "result": "roundtrip result",
    });
    dispatch::route("memory_store", &state, store_params)
        .await
        .expect("store should succeed");

    let exported = dispatch::route("memory_export", &state, json!({}))
        .await
        .expect("export should succeed");

    let fresh_state = build_deterministic_state();
    let result = dispatch::route("memory_import", &fresh_state, exported).await;
    let data = result.expect("import should succeed");
    assert_eq!(data["imported"], 1);
    assert_eq!(data["skipped"], 0);
}
