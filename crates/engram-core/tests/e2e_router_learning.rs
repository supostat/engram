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
                "context": format!("learning context number {index}"),
                "action": format!("learning action number {index}"),
                "result": format!("learning result number {index}"),
            }),
        )
        .await
        .expect("store should succeed");
        let memory_id = stored["id"].as_str().expect("id").to_string();

        dispatch::route(
            "memory_search",
            &state,
            json!({
                "query": format!("learning context number {index}"),
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
            "query": "learning context",
            "limit": 10,
        }),
    )
    .await
    .expect("post-learning search should succeed");
    let results = final_search.as_array().expect("results array");
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
