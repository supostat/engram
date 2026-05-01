use std::sync::{Arc, Mutex, RwLock};

use serde_json::{Value, json};

use engram_core::config::Config;
use engram_core::dispatch;
use engram_core::indexes::IndexSet;
use engram_core::migrations;
use engram_core::server::ServerState;
use engram_embeddings::Embedder;
use engram_llm_client::{EmbeddingProvider, TextGenerator};
use engram_router::Router;
use engram_storage::{Database, Memory};

fn build_deterministic_state() -> Arc<ServerState> {
    let database = Database::in_memory().expect("in-memory database");
    let mut config = Config::default();
    config.embedding.provider = "deterministic".into();
    let indexes = IndexSet::new(|| config.build_hnsw_params()).expect("index set");
    let embedder = Embedder::new();
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

async fn store_with_tags(state: &Arc<ServerState>, context: &str, tags: &[&str]) -> String {
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

fn seed_memory_directly(state: &Arc<ServerState>, id: &str, context: &str, raw_tags: &str) {
    let memory = Memory {
        id: id.to_string(),
        memory_type: "decision".to_string(),
        context: context.to_string(),
        action: format!("action for {context}"),
        result: format!("result for {context}"),
        score: 0.0,
        embedding_context: None,
        embedding_action: None,
        embedding_result: None,
        indexed: false,
        tags: Some(raw_tags.to_string()),
        project: None,
        parent_id: None,
        source_ids: None,
        insight_type: None,
        created_at: "2026-05-01T00:00:00Z".to_string(),
        updated_at: "2026-05-01T00:00:00Z".to_string(),
        used_count: 0,
        last_used_at: None,
        superseded_by: None,
    };
    let database = state.database.lock().unwrap();
    database.insert_memory(&memory).expect("seed legacy row");
}

#[tokio::test]
async fn search_filters_by_single_tag() {
    let state = build_deterministic_state();
    let rust_id = store_with_tags(&state, "rust compiler optimization", &["rust", "bugfix"]).await;
    let _python_id = store_with_tags(
        &state,
        "python runtime optimization",
        &["python", "feature"],
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
    let rust_id = store_with_tags(&state, "rust compiler optimization", &["rust", "bugfix"]).await;
    let _python_id = store_with_tags(
        &state,
        "python runtime optimization",
        &["python", "feature"],
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
    let rust_id = store_with_tags(&state, "rust compiler optimization", &["rust", "bugfix"]).await;
    let python_id = store_with_tags(
        &state,
        "python runtime optimization",
        &["python", "feature"],
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
    let rust_id = store_with_tags(&state, "rust compiler optimization", &["Rust", "BugFix"]).await;

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
    let _rust_id = store_with_tags(&state, "rust compiler optimization", &["rust", "bugfix"]).await;

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

#[tokio::test]
async fn search_finds_record_after_csv_tag_migration() {
    let state = build_deterministic_state();
    seed_memory_directly(&state, "csv-1", "rust compiler csv", "rust,bugfix");
    {
        let database = state.database.lock().unwrap();
        migrations::run_pending(&database).expect("migration runs");
    }

    let results = dispatch::route(
        "memory_search",
        &state,
        json!({
            "query": "rust compiler csv",
            "limit": 10,
            "tags": ["rust"],
        }),
    )
    .await
    .expect("search after csv migration");
    let ids = extract_ids(&results);
    assert!(
        ids.contains(&"csv-1".to_string()),
        "csv-migrated row must be findable by tag"
    );
}

#[tokio::test]
async fn search_finds_record_after_naked_tag_migration() {
    let state = build_deterministic_state();
    seed_memory_directly(&state, "naked-1", "naked rust note", "rust");
    {
        let database = state.database.lock().unwrap();
        migrations::run_pending(&database).expect("migration runs");
    }

    let results = dispatch::route(
        "memory_search",
        &state,
        json!({
            "query": "naked rust note",
            "limit": 10,
            "tags": ["rust"],
        }),
    )
    .await
    .expect("search after naked migration");
    let ids = extract_ids(&results);
    assert!(
        ids.contains(&"naked-1".to_string()),
        "naked-migrated row must be findable by tag"
    );
}

#[tokio::test]
async fn store_accepts_encoded_string_for_backward_compat() {
    let state = build_deterministic_state();
    let stored = dispatch::route(
        "memory_store",
        &state,
        json!({
            "memory_type": "decision",
            "context": "encoded string compat",
            "action": "store via encoded",
            "result": "accepted",
            "tags": "[\"rust\"]",
        }),
    )
    .await
    .expect("store with encoded-string tags");
    let stored_id = stored["id"].as_str().expect("id").to_string();

    let results = dispatch::route(
        "memory_search",
        &state,
        json!({
            "query": "encoded string compat",
            "limit": 10,
            "tags": ["rust"],
        }),
    )
    .await
    .expect("search after encoded-string store");
    let ids = extract_ids(&results);
    assert!(
        ids.contains(&stored_id),
        "encoded-string wire path must be searchable by tag"
    );
}
