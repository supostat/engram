use std::sync::{Arc, Mutex, RwLock};

use serde_json::json;

use engram_core::config::Config;
use engram_core::dispatch;
use engram_core::indexes::IndexSet;
use engram_core::server::ServerState;
use engram_embeddings::Embedder;
use engram_llm_client::error::ApiError;
use engram_llm_client::{EmbeddingProvider, TextGenerator};
use engram_router::Router;
use engram_storage::{Database, Memory};

struct FailingEmbeddingProvider {
    dimension: usize,
}

impl EmbeddingProvider for FailingEmbeddingProvider {
    fn embed(&self, _text: &str, _input_type: Option<&str>) -> Result<Vec<f32>, ApiError> {
        Err(ApiError::EmbeddingApiUnavailable(
            "reembed test: provider intentionally unavailable".into(),
        ))
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn model_name(&self) -> &str {
        "failing-embedding"
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

async fn store_memory(
    state: &Arc<ServerState>,
    context: &str,
    action: &str,
    result: &str,
) -> String {
    let response = dispatch::route(
        "memory_store",
        state,
        json!({
            "memory_type": "decision",
            "context": context,
            "action": action,
            "result": result,
        }),
    )
    .await
    .expect("store should succeed");
    response["id"]
        .as_str()
        .expect("store returns id")
        .to_string()
}

#[tokio::test]
async fn reembed_empty_database_returns_zero() {
    let state = build_deterministic_state();

    let response = dispatch::route("memory_reembed", &state, json!({}))
        .await
        .expect("reembed should succeed");

    assert_eq!(response["total"], 0);
    assert_eq!(response["succeeded"], 0);
    assert_eq!(response["failed"], 0);
    assert_eq!(response["model"], "deterministic");
}

#[tokio::test]
async fn reembed_processes_all_memories() {
    let state = build_deterministic_state();
    let id_a = store_memory(
        &state,
        "user opened the editor and started typing a document about rust",
        "typed several paragraphs about the rust programming language and its memory model",
        "the document was saved successfully to the local filesystem without errors",
    )
    .await;
    let id_b = store_memory(
        &state,
        "developer reviewing pull request for new feature branch implementation",
        "added inline comments suggesting changes to the authentication module logic",
        "review submitted and the pull request was approved by all reviewers on the team",
    )
    .await;

    let response = dispatch::route("memory_reembed", &state, json!({}))
        .await
        .expect("reembed should succeed");

    assert_eq!(response["total"], 2);
    assert_eq!(response["succeeded"], 2);
    assert_eq!(response["failed"], 0);

    let database = state.database.lock().unwrap();
    let memory_a = database.get_memory(&id_a).unwrap();
    let memory_b = database.get_memory(&id_b).unwrap();
    assert!(memory_a.indexed);
    assert!(memory_b.indexed);
    assert!(
        memory_a
            .embedding_context
            .as_deref()
            .is_some_and(|b| !b.is_empty())
    );
    assert!(
        memory_b
            .embedding_context
            .as_deref()
            .is_some_and(|b| !b.is_empty())
    );
}

#[tokio::test]
async fn reembed_clears_stale_cache_entries() {
    let state = build_deterministic_state();
    store_memory(
        &state,
        "engineer wrote tests for the new caching layer in the embedding pipeline today",
        "added unit tests covering happy path edge cases and error propagation through the layer",
        "all tests passed and the caching layer is now ready for code review by team members",
    )
    .await;
    let stale_query = "caching layer tests written today by the engineer";
    dispatch::route(
        "memory_search",
        &state,
        json!({ "query": stale_query, "limit": 5 }),
    )
    .await
    .expect("search should succeed");
    assert!(
        state
            .embedder
            .cache()
            .get(stale_query, Some("query"))
            .is_some(),
        "search primes the cache for its query"
    );

    dispatch::route("memory_reembed", &state, json!({}))
        .await
        .expect("reembed should succeed");

    assert!(
        state
            .embedder
            .cache()
            .get(stale_query, Some("query"))
            .is_none(),
        "reembed must purge the stale search query from the cache"
    );
}

#[tokio::test]
async fn reembed_refreshes_hnsw_so_search_returns_memory() {
    let state = build_deterministic_state();
    let id = store_memory(
        &state,
        "data scientist exploring the distribution of embeddings across the corpus today",
        "ran cluster analysis on the vector store and identified three distinct topic groups",
        "saved the cluster labels back to the database for downstream consumers to use",
    )
    .await;

    dispatch::route("memory_reembed", &state, json!({}))
        .await
        .expect("reembed should succeed");

    let search = dispatch::route(
        "memory_search",
        &state,
        json!({ "query": "cluster analysis vector store", "limit": 5 }),
    )
    .await
    .expect("search after reembed should succeed");

    let results = search
        .as_array()
        .expect("search returns an array of memories");
    let found = results
        .iter()
        .any(|hit| hit["id"].as_str() == Some(id.as_str()));
    assert!(found, "reembedded memory must be findable via search");
}

fn build_state_with_provider(
    provider: Arc<dyn EmbeddingProvider + Send + Sync>,
) -> Arc<ServerState> {
    let database = Database::in_memory().expect("in-memory database");
    let mut config = Config::default();
    config.embedding.provider = "deterministic".into();
    let indexes = IndexSet::new(|| config.build_hnsw_params()).expect("index set");
    let embedder = Embedder::new(0);
    let router = Router::new(0.1, 0.15);
    let text_generator: Option<Arc<dyn TextGenerator + Send + Sync>> =
        config.build_text_generator().ok().map(Arc::from);
    Arc::new(ServerState {
        database: Mutex::new(database),
        indexes: RwLock::new(indexes),
        embedder,
        router: Mutex::new(router),
        config,
        database_path: String::new(),
        embedding_provider: provider,
        text_generator,
    })
}

fn raw_memory(id: &str) -> Memory {
    Memory {
        id: id.to_string(),
        memory_type: "decision".into(),
        context: format!("context body for {id} sufficient length to skip hyde threshold zero"),
        action: format!("action body for {id} sufficient length to skip hyde threshold zero"),
        result: format!("result body for {id} sufficient length to skip hyde threshold zero"),
        score: 0.0,
        embedding_context: Some(vec![0u8; 4]),
        embedding_action: Some(vec![0u8; 4]),
        embedding_result: Some(vec![0u8; 4]),
        indexed: true,
        tags: None,
        project: None,
        parent_id: None,
        source_ids: None,
        insight_type: None,
        created_at: "2026-05-14T00:00:00Z".into(),
        updated_at: "2026-05-14T00:00:00Z".into(),
        used_count: 0,
        last_used_at: None,
        superseded_by: None,
    }
}

#[tokio::test]
async fn reembed_recoverable_failure_marks_memory_for_retry() {
    let provider: Arc<dyn EmbeddingProvider + Send + Sync> =
        Arc::new(FailingEmbeddingProvider { dimension: 1024 });
    let state = build_state_with_provider(provider);
    {
        let database = state.database.lock().unwrap();
        database.insert_memory(&raw_memory("mem-a")).unwrap();
        database.insert_memory(&raw_memory("mem-b")).unwrap();
    }

    let response = dispatch::route("memory_reembed", &state, json!({}))
        .await
        .expect("reembed should not propagate recoverable failures");

    assert_eq!(response["total"], 2);
    assert_eq!(response["succeeded"], 0);
    assert_eq!(response["failed"], 2);

    let database = state.database.lock().unwrap();
    assert!(
        !database.get_memory("mem-a").unwrap().indexed,
        "recoverable failure must reset indexed=0 so background reindex retries"
    );
    assert!(!database.get_memory("mem-b").unwrap().indexed);
}

#[tokio::test]
async fn reembed_accepts_force_flag_as_no_op() {
    let state = build_deterministic_state();
    store_memory(
        &state,
        "context describing the force flag behavior in the reembed handler interface",
        "passed force=true through the dispatch route and observed the handler response",
        "handler accepts the flag without altering behavior in the current implementation",
    )
    .await;

    let response = dispatch::route("memory_reembed", &state, json!({ "force": true }))
        .await
        .expect("reembed with force=true should succeed");

    assert_eq!(response["total"], 1);
    assert_eq!(response["succeeded"], 1);
}
