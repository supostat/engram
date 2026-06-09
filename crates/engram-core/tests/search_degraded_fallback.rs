use std::sync::{Arc, Mutex, RwLock};

use serde_json::json;

use engram_core::config::Config;
use engram_core::dispatch;
use engram_core::indexes::IndexSet;
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

fn build_state_with_failing_embeddings() -> Arc<ServerState> {
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
async fn search_degrades_to_fts_when_embeddings_unavailable() {
    let state = build_state_with_failing_embeddings();
    seed_memory_directly(
        &state,
        "fts-1",
        "reciprocal rank fusion ranking algorithm",
        "search",
    );

    let response = dispatch::route(
        "memory_search",
        &state,
        json!({ "query": "reciprocal rank fusion", "limit": 10 }),
    )
    .await
    .expect("must succeed via FTS fallback");

    assert_eq!(response["degraded"], true);
    let ids: Vec<&str> = response["results"]
        .as_array()
        .expect("results array")
        .iter()
        .filter_map(|entry| entry["id"].as_str())
        .collect();
    assert!(
        ids.contains(&"fts-1"),
        "FTS fallback must surface the seeded memory"
    );
}

#[tokio::test]
async fn search_degrades_with_empty_results_on_empty_corpus() {
    let state = build_state_with_failing_embeddings();

    let response = dispatch::route(
        "memory_search",
        &state,
        json!({ "query": "nothing seeded here", "limit": 10 }),
    )
    .await
    .expect("must succeed via FTS fallback");

    assert_eq!(response["degraded"], true);
    assert!(
        response["results"]
            .as_array()
            .expect("results array")
            .is_empty(),
        "empty corpus yields no results"
    );
}
