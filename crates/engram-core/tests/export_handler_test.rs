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
async fn export_empty_database_returns_zero_count() {
    let state = build_deterministic_state();
    let result = dispatch::route("memory_export", &state, json!({})).await;
    let data = result.expect("export should succeed");
    assert_eq!(data["version"], 1);
    assert_eq!(data["count"], 0);
    assert!(data["memories"].as_array().expect("memories array").is_empty());
    assert!(data["exported_at"].is_string());
}

#[tokio::test]
async fn export_includes_stored_memories() {
    let state = build_deterministic_state();
    let store_params = json!({
        "memory_type": "decision",
        "context": "export test context",
        "action": "export test action",
        "result": "export test result",
        "tags": "test,export",
        "project": "engram",
    });
    dispatch::route("memory_store", &state, store_params)
        .await
        .expect("store should succeed");

    let result = dispatch::route("memory_export", &state, json!({})).await;
    let data = result.expect("export should succeed");
    assert_eq!(data["version"], 1);
    assert_eq!(data["count"], 1);
    let memories = data["memories"].as_array().expect("memories array");
    assert_eq!(memories.len(), 1);
    let memory = &memories[0];
    assert_eq!(memory["memory_type"], "decision");
    assert_eq!(memory["context"], "export test context");
    assert_eq!(memory["tags"], "test,export");
    assert_eq!(memory["project"], "engram");
    assert!(memory.get("embedding_context").is_none());
    assert!(memory.get("embedding_action").is_none());
    assert!(memory.get("embedding_result").is_none());
    assert!(memory.get("indexed").is_none());
    assert!(memory.get("superseded_by").is_none());
}

#[tokio::test]
async fn export_excludes_superseded_memories() {
    let state = build_deterministic_state();
    let store_original = json!({
        "memory_type": "decision",
        "context": "original context",
        "action": "original action",
        "result": "original result",
    });
    let stored = dispatch::route("memory_store", &state, store_original)
        .await
        .expect("store original should succeed");
    let original_id = stored["id"].as_str().expect("original id").to_string();

    let store_replacement = json!({
        "memory_type": "decision",
        "context": "replacement context",
        "action": "replacement action",
        "result": "replacement result",
    });
    let replacement = dispatch::route("memory_store", &state, store_replacement)
        .await
        .expect("store replacement should succeed");
    let replacement_id = replacement["id"].as_str().expect("replacement id").to_string();

    {
        let database = state.database.lock().unwrap();
        database
            .set_superseded_by(&original_id, &replacement_id)
            .expect("set superseded");
    }

    let result = dispatch::route("memory_export", &state, json!({})).await;
    let data = result.expect("export should succeed");
    assert_eq!(data["count"], 1);
    let memories = data["memories"].as_array().expect("memories array");
    assert_eq!(memories[0]["id"].as_str().expect("id"), replacement_id);
}
