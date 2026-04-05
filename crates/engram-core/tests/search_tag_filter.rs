use std::sync::{Arc, Mutex};

use serde_json::{Value, json};

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

async fn store_with_tags(state: &Arc<ServerState>, context: &str, tags: &str) -> String {
    let stored = dispatch::route(
        "memory_store",
        state,
        json!({
            "memory_type": "decision",
            "context": context,
            "action": format!("action for {context}"),
            "result": format!("result for {context}"),
            "tags": tags,
        }),
    )
    .await
    .expect("store should succeed");
    stored["id"].as_str().expect("stored id").to_string()
}

fn extract_ids(search_result: &Value) -> Vec<String> {
    search_result
        .as_array()
        .expect("results array")
        .iter()
        .map(|entry| entry["id"].as_str().expect("id").to_string())
        .collect()
}

#[tokio::test]
async fn search_filters_by_single_tag() {
    let state = build_deterministic_state();
    let rust_id =
        store_with_tags(&state, "rust compiler optimization", r#"["rust","bugfix"]"#).await;
    let _python_id = store_with_tags(
        &state,
        "python runtime optimization",
        r#"["python","feature"]"#,
    )
    .await;

    let results = dispatch::route(
        "memory_search",
        &state,
        json!({
            "query": "optimization",
            "limit": 10,
            "tags": ["rust"],
        }),
    )
    .await
    .expect("search with single tag");
    let ids = extract_ids(&results);
    assert!(ids.contains(&rust_id), "rust memory must be returned");
    assert_eq!(ids.len(), 1, "only rust memory should match");
}

#[tokio::test]
async fn search_filters_by_multiple_tags() {
    let state = build_deterministic_state();
    let rust_id =
        store_with_tags(&state, "rust compiler optimization", r#"["rust","bugfix"]"#).await;
    let _python_id = store_with_tags(
        &state,
        "python runtime optimization",
        r#"["python","feature"]"#,
    )
    .await;

    let results = dispatch::route(
        "memory_search",
        &state,
        json!({
            "query": "optimization",
            "limit": 10,
            "tags": ["rust", "bugfix"],
        }),
    )
    .await
    .expect("search with multiple tags");
    let ids = extract_ids(&results);
    assert!(ids.contains(&rust_id), "rust memory must match both tags");
    assert_eq!(ids.len(), 1, "only rust memory has both tags");
}

#[tokio::test]
async fn search_without_tags_returns_all() {
    let state = build_deterministic_state();
    let rust_id =
        store_with_tags(&state, "rust compiler optimization", r#"["rust","bugfix"]"#).await;
    let python_id = store_with_tags(
        &state,
        "python runtime optimization",
        r#"["python","feature"]"#,
    )
    .await;

    let results = dispatch::route(
        "memory_search",
        &state,
        json!({
            "query": "optimization",
            "limit": 10,
        }),
    )
    .await
    .expect("search without tags");
    let ids = extract_ids(&results);
    assert!(ids.contains(&rust_id), "rust memory must be returned");
    assert!(ids.contains(&python_id), "python memory must be returned");
}

#[tokio::test]
async fn search_tag_matching_is_case_insensitive() {
    let state = build_deterministic_state();
    let rust_id =
        store_with_tags(&state, "rust compiler optimization", r#"["Rust","BugFix"]"#).await;

    let results = dispatch::route(
        "memory_search",
        &state,
        json!({
            "query": "optimization",
            "limit": 10,
            "tags": ["rust", "BUGFIX"],
        }),
    )
    .await
    .expect("case-insensitive tag search");
    let ids = extract_ids(&results);
    assert!(ids.contains(&rust_id), "case-insensitive match must work");
}

#[tokio::test]
async fn search_nonexistent_tag_returns_empty() {
    let state = build_deterministic_state();
    let _rust_id =
        store_with_tags(&state, "rust compiler optimization", r#"["rust","bugfix"]"#).await;

    let results = dispatch::route(
        "memory_search",
        &state,
        json!({
            "query": "optimization",
            "limit": 10,
            "tags": ["golang"],
        }),
    )
    .await
    .expect("search with nonexistent tag");
    let ids = extract_ids(&results);
    assert!(ids.is_empty(), "no memories should match unknown tag");
}
