use std::sync::{Arc, Mutex, RwLock};

use serde_json::{Value, json};

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

fn make_store_params(context: &str, action: &str, result: &str) -> Value {
    json!({
        "memory_type": "decision",
        "context": context,
        "action": action,
        "result": result,
    })
}

#[tokio::test]
async fn store_search_judge_status_cycle() {
    let state = build_deterministic_state();

    let stored = dispatch::route(
        "memory_store",
        &state,
        make_store_params(
            "configured database connection pooling",
            "set maximum connections to fifty",
            "connections stable under load",
        ),
    )
    .await
    .expect("store should succeed");
    let stored_id = stored["id"].as_str().expect("stored id");

    let search_result = dispatch::route(
        "memory_search",
        &state,
        json!({
            "query": "database connection pooling",
            "limit": 10,
        }),
    )
    .await
    .expect("search should succeed");
    assert_eq!(search_result["degraded"], false);
    let results = search_result["results"].as_array().expect("results array");
    let found = results.iter().any(|entry| entry["id"] == stored_id);
    assert!(found, "stored memory must appear in search results");

    let judge_result = dispatch::route(
        "memory_judge",
        &state,
        json!({
            "memory_id": stored_id,
            "score": 0.8,
        }),
    )
    .await
    .expect("judge should succeed");
    let judged_score = judge_result["score"].as_f64().expect("score");
    assert!(
        (judged_score - 0.8).abs() < 0.01,
        "score should be ~0.8, got {judged_score}"
    );
    assert!(!judge_result["degraded"].as_bool().unwrap());

    let status = dispatch::route("memory_status", &state, json!({}))
        .await
        .expect("status should succeed");
    assert_eq!(status["memory_count"], 1);
    assert_eq!(status["indexed_count"], 1);
}

// Write-time deduplication folds away near-identical memories. The deterministic
// provider would treat field text differing only by a trailing digit as a ~0.999
// duplicate, so a roundtrip of N memories must store N genuinely distinct ones.
const ROUNDTRIP_MEMORIES: [(&str, &str, &str); 3] = [
    (
        "migrated the build pipeline to a hermetic sandbox",
        "pinned every toolchain version inside the container image",
        "reproducible builds eliminated works-on-my-machine reports",
    ),
    (
        "adopted structured logging across the request handlers",
        "emitted correlation identifiers on every span boundary",
        "incident triage time dropped sharply for distributed traces",
    ),
    (
        "replaced polling with a websocket push notification channel",
        "subscribed clients to topic-scoped server-sent updates",
        "perceived freshness improved while backend load decreased",
    ),
];

#[tokio::test]
async fn store_export_import_roundtrip() {
    let state = build_deterministic_state();
    for (context, action, result) in ROUNDTRIP_MEMORIES {
        dispatch::route(
            "memory_store",
            &state,
            make_store_params(context, action, result),
        )
        .await
        .expect("store should succeed");
    }

    let exported = dispatch::route("memory_export", &state, json!({}))
        .await
        .expect("export should succeed");
    assert_eq!(exported["version"], 1);
    assert_eq!(exported["count"], 3);

    let fresh_state = build_deterministic_state();
    let import_result = dispatch::route("memory_import", &fresh_state, exported)
        .await
        .expect("import should succeed");
    assert_eq!(import_result["imported"], 3);
    assert_eq!(import_result["skipped"], 0);

    let search_result = dispatch::route(
        "memory_search",
        &fresh_state,
        json!({
            "query": "hermetic reproducible build pipeline toolchain",
            "limit": 10,
        }),
    )
    .await
    .expect("search in fresh state should succeed");
    let results = search_result["results"].as_array().expect("results array");
    assert!(!results.is_empty(), "imported memories must be searchable");
}

#[tokio::test]
async fn consolidation_preview_cycle() {
    let state = build_deterministic_state();
    dispatch::route(
        "memory_store",
        &state,
        make_store_params(
            "database optimization strategy for production",
            "enable query caching at application layer",
            "reduced query latency by forty percent",
        ),
    )
    .await
    .expect("store first should succeed");
    dispatch::route(
        "memory_store",
        &state,
        make_store_params(
            "database optimization approach for production",
            "enable query caching at service layer",
            "reduced query latency by thirty percent",
        ),
    )
    .await
    .expect("store second should succeed");

    let preview = dispatch::route(
        "memory_consolidate_preview",
        &state,
        json!({
            "stale_days": 0,
            "min_score": 0.0,
        }),
    )
    .await
    .expect("preview should succeed");
    assert!(preview.get("duplicates").is_some());
    assert!(preview.get("stale").is_some());
    assert!(preview.get("garbage").is_some());

    let status = dispatch::route("memory_status", &state, json!({}))
        .await
        .expect("status should succeed");
    assert_eq!(status["memory_count"], 2);
}
