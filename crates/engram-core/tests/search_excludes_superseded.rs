use std::sync::{Arc, Mutex, RwLock};

use serde_json::{Value, json};

use engram_core::config::Config;
use engram_core::dispatch;
use engram_core::indexes::IndexSet;
use engram_core::lock_helpers;
use engram_core::server::ServerState;
use engram_embeddings::Embedder;
use engram_llm_client::{ApiError, EmbeddingProvider, TextGenerator};
use engram_router::Router;
use engram_storage::{Database, Memory};

struct UnavailableEmbeddingProvider;

impl EmbeddingProvider for UnavailableEmbeddingProvider {
    fn embed(&self, _text: &str, _input_type: Option<&str>) -> Result<Vec<f32>, ApiError> {
        Err(ApiError::EmbeddingApiUnavailable(
            "provider down for test".into(),
        ))
    }

    fn dimension(&self) -> usize {
        1024
    }

    fn model_name(&self) -> &str {
        "unavailable-test-provider"
    }
}

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

fn build_degraded_state() -> Arc<ServerState> {
    let database = Database::in_memory().expect("in-memory database");
    let mut config = Config::default();
    config.embedding.provider = "deterministic".into();
    let indexes = IndexSet::new(|| config.build_hnsw_params()).expect("index set");
    let embedder = Embedder::new(0);
    let router = Router::new(0.1, 0.15);
    let embedding_provider: Arc<dyn EmbeddingProvider + Send + Sync> =
        Arc::new(UnavailableEmbeddingProvider);
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

fn seed_memory_directly(state: &Arc<ServerState>, id: &str, shared_text: &str) {
    let memory = Memory {
        id: id.to_string(),
        memory_type: "decision".to_string(),
        context: shared_text.to_string(),
        action: format!("action {shared_text}"),
        result: format!("result {shared_text}"),
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
        created_at: "2026-05-01T00:00:00Z".to_string(),
        updated_at: "2026-05-01T00:00:00Z".to_string(),
        used_count: 0,
        last_used_at: None,
        superseded_by: None,
    };
    let database = lock_helpers::lock_db(state);
    database.insert_memory(&memory).expect("seed row");
}

fn supersede(state: &Arc<ServerState>, retired_id: &str, survivor_id: &str) {
    let database = lock_helpers::lock_db(state);
    database
        .set_superseded_by(retired_id, survivor_id)
        .expect("mark superseded");
}

async fn search_ids(state: &Arc<ServerState>, query: &str) -> Vec<String> {
    let response = dispatch::route(
        "memory_search",
        state,
        json!({ "query": query, "limit": 50 }),
    )
    .await
    .expect("search succeeds");
    response["results"]
        .as_array()
        .expect("results array")
        .iter()
        .filter_map(|entry| entry["id"].as_str().map(str::to_string))
        .collect()
}

fn used_count(state: &Arc<ServerState>, id: &str) -> i64 {
    lock_helpers::lock_db(state)
        .get_memory(id)
        .expect("row exists")
        .used_count
}

// A retired (superseded) row must never surface on live vector search, and must NOT
// accrue track_search usage even if the index/FTS still references it.
#[tokio::test]
async fn superseded_row_hidden_from_live_search() {
    let state = build_deterministic_state();
    let shared = "reciprocal rank fusion ranking algorithm superseded guard";
    seed_memory_directly(&state, "survivor-a", shared);
    seed_memory_directly(&state, "retired-b", shared);
    supersede(&state, "retired-b", "survivor-a");

    let ids = search_ids(&state, "reciprocal rank fusion ranking algorithm").await;
    assert!(
        ids.contains(&"survivor-a".to_string()),
        "the surviving row must remain searchable"
    );
    assert!(
        !ids.contains(&"retired-b".to_string()),
        "a superseded row must be absent from live search results"
    );
    assert_eq!(
        used_count(&state, "retired-b"),
        0,
        "a skipped superseded row must not accrue track_search usage"
    );
}

// The same guard must hold when the search degrades to FTS-only (embeddings down):
// the FTS path also runs through load_memories, so the superseded row stays hidden.
#[tokio::test]
async fn superseded_row_hidden_in_degraded_fts_search() {
    let state = build_degraded_state();
    let shared = "wal mode concurrent write throughput superseded degraded";
    seed_memory_directly(&state, "survivor-c", shared);
    seed_memory_directly(&state, "retired-d", shared);
    supersede(&state, "retired-d", "survivor-c");

    let response: Value = dispatch::route(
        "memory_search",
        &state,
        json!({ "query": "wal mode concurrent write throughput", "limit": 50 }),
    )
    .await
    .expect("degraded search succeeds");
    assert_eq!(
        response["degraded"], true,
        "embeddings down forces FTS mode"
    );

    let ids: Vec<String> = response["results"]
        .as_array()
        .expect("results array")
        .iter()
        .filter_map(|entry| entry["id"].as_str().map(str::to_string))
        .collect();
    assert!(
        ids.contains(&"survivor-c".to_string()),
        "FTS fallback must surface the surviving row"
    );
    assert!(
        !ids.contains(&"retired-d".to_string()),
        "a superseded row must be hidden in FTS-only mode too"
    );
    assert_eq!(
        used_count(&state, "retired-d"),
        0,
        "degraded path must not accrue usage on the superseded row"
    );
}
