use std::sync::{Arc, Mutex, RwLock};

use serde_json::json;

use engram_core::config::Config;
use engram_core::dispatch;
use engram_core::indexes::IndexSet;
use engram_core::server::ServerState;
use engram_embeddings::Embedder;
use engram_llm_client::{EmbeddingProvider, TextGenerator};
use engram_router::Router;
use engram_storage::Database;

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

#[tokio::test]
async fn delete_removes_memory_from_db_and_search() {
    let state = build_deterministic_state();

    let stored = dispatch::route(
        "memory_store",
        &state,
        json!({
            "memory_type": "decision",
            "context": "delete handler context for removal coverage",
            "action": "delete handler action for removal coverage",
            "result": "delete handler result for removal coverage",
        }),
    )
    .await
    .expect("store should succeed");
    let memory_id = stored["id"].as_str().expect("id").to_string();

    // Record a search so feedback_tracking has a child row to clear (FK-safe path).
    dispatch::route(
        "memory_search",
        &state,
        json!({
            "query": "delete handler context for removal coverage",
            "limit": 5,
        }),
    )
    .await
    .expect("search should succeed");

    let deleted = dispatch::route("memory_delete", &state, json!({ "id": &memory_id }))
        .await
        .expect("delete should succeed");
    assert_eq!(deleted["deleted"], memory_id);

    {
        let database = state.database.lock().unwrap();
        let get_result = database.get_memory(&memory_id);
        assert!(get_result.is_err(), "memory should be gone after delete");
    }

    let search_after = dispatch::route(
        "memory_search",
        &state,
        json!({
            "query": "delete handler context for removal coverage",
            "limit": 10,
        }),
    )
    .await
    .expect("post-delete search should succeed");
    let results = search_after["results"].as_array().expect("results array");
    assert!(
        results
            .iter()
            .all(|result| result["id"] != json!(memory_id)),
        "deleted memory must not appear in search results"
    );
}

#[tokio::test]
async fn delete_missing_id_returns_not_found() {
    let state = build_deterministic_state();

    let result = dispatch::route("memory_delete", &state, json!({ "id": "nonexistent-id" })).await;
    let error = result.expect_err("delete of missing id should fail");
    assert!(
        error.to_string().contains("not found"),
        "expected not-found error, got: {error}"
    );
}

#[tokio::test]
async fn delete_missing_params_is_dispatch_error() {
    let state = build_deterministic_state();

    let result = dispatch::route("memory_delete", &state, json!({})).await;
    let error = result.expect_err("delete with missing params should fail");
    assert!(
        error.to_string().contains("dispatch error"),
        "expected dispatch error, got: {error}"
    );
}

#[tokio::test]
async fn delete_non_indexed_memory_succeeds() {
    let state = build_deterministic_state();

    // Insert a row straight into the DB without indexing it into the HNSW set,
    // so the handler's `indexes.contains(hash)` guard is false — the unindexed
    // path (e.g. an insight, indexed = FALSE) must still delete cleanly.
    {
        let database = state.database.lock().unwrap();
        database
            .connection()
            .execute(
                "INSERT INTO memories (id, memory_type, context, action, result, created_at, updated_at)
                 VALUES ('insight-1', 'insight', 'ctx', 'act', 'res', '2026-01-01', '2026-01-01')",
                [],
            )
            .expect("insert unindexed memory");
    }

    let deleted = dispatch::route("memory_delete", &state, json!({ "id": "insight-1" }))
        .await
        .expect("delete of a non-indexed memory should succeed");
    assert_eq!(deleted["deleted"], "insight-1");

    let database = state.database.lock().unwrap();
    assert!(
        database.get_memory("insight-1").is_err(),
        "non-indexed memory should be gone after delete"
    );
}
