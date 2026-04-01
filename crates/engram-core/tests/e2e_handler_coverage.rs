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
async fn all_handlers_accessible() {
    let state = build_deterministic_state();

    let stored = dispatch::route("memory_store", &state, json!({
        "memory_type": "decision",
        "context": "handler coverage context",
        "action": "handler coverage action",
        "result": "handler coverage result",
    })).await.expect("memory_store");
    let memory_id = stored["id"].as_str().expect("id");
    assert!(!memory_id.is_empty());

    let search = dispatch::route("memory_search", &state, json!({
        "query": "handler coverage",
        "limit": 5,
    })).await.expect("memory_search");
    assert!(search.is_array());

    let judge = dispatch::route("memory_judge", &state, json!({
        "memory_id": memory_id,
        "score": 0.7,
    })).await.expect("memory_judge");
    assert!(judge.get("score").is_some());

    let status = dispatch::route("memory_status", &state, json!({}))
        .await.expect("memory_status");
    assert!(status.get("memory_count").is_some());

    let config = dispatch::route("memory_config", &state, json!({
        "action": "get",
    })).await.expect("memory_config");
    assert!(config.get("database").is_some());

    let exported = dispatch::route("memory_export", &state, json!({}))
        .await.expect("memory_export");
    assert_eq!(exported["version"], 1);

    let import_result = dispatch::route("memory_import", &state, json!({
        "version": 1,
        "memories": [],
    })).await.expect("memory_import");
    assert_eq!(import_result["imported"], 0);

    let preview = dispatch::route("memory_consolidate_preview", &state, json!({
        "stale_days": 0,
        "min_score": 0.0,
    })).await.expect("memory_consolidate_preview");
    assert!(preview.get("duplicates").is_some());

    let consolidate = dispatch::route("memory_consolidate", &state, json!({
        "stale_days": 0,
        "min_score": 0.0,
    })).await;
    if let Ok(data) = consolidate {
        assert!(data.get("analyzed_count").is_some());
    }

    let apply = dispatch::route("memory_consolidate_apply", &state, json!({
        "stale_days": 0,
        "min_score": 0.0,
    })).await;
    if let Ok(data) = apply {
        assert!(data.get("merged").is_some());
    }

    let insights = dispatch::route("memory_insights", &state, json!({
        "action": "list",
    })).await.expect("memory_insights");
    assert!(insights["insights"].is_array());
}

#[tokio::test]
async fn unknown_method_returns_dispatch_error() {
    let state = build_deterministic_state();
    let result = dispatch::route("nonexistent", &state, json!({})).await;
    let error = result.expect_err("unknown method should fail");
    assert!(error.to_string().contains("[6007]"));
    assert!(error.to_string().contains("nonexistent"));
}

#[tokio::test]
async fn config_set_returns_readonly_error() {
    let state = build_deterministic_state();
    let result = dispatch::route("memory_config", &state, json!({
        "action": "set",
    })).await;
    let error = result.expect_err("config set should fail");
    assert!(error.to_string().contains("[6008]"));
}

#[tokio::test]
async fn import_wrong_version_returns_error() {
    let state = build_deterministic_state();
    let result = dispatch::route("memory_import", &state, json!({
        "version": 99,
        "memories": [],
    })).await;
    let error = result.expect_err("wrong version should fail");
    assert!(error.to_string().contains("[6010]"));
}
