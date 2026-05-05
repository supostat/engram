use std::sync::{Arc, Mutex, RwLock};

use serde_json::json;

use engram_core::config::{
    Config, deterministic_provider_max_concurrent, disable_deterministic_provider_instrumentation,
    enable_deterministic_provider_instrumentation, reset_deterministic_provider_counters,
};
use engram_core::dispatch;
use engram_core::indexes::IndexSet;
use engram_core::server::ServerState;
use engram_embeddings::Embedder;
use engram_llm_client::{EmbeddingProvider, TextGenerator};
use engram_router::Router;
use engram_storage::Database;

// Serializes these tests against any other test that reads the global
// deterministic-provider counters. Poisoning is tolerated — the guard
// exists solely to prevent interleaving.
static COUNTER_LOCK: Mutex<()> = Mutex::new(());

// RAII guard: enables instrumentation on construction, disables on drop
// (including panic). This ensures the deterministic provider's 20ms sleep
// never leaks outside the test scope into any subsequent test or binary.
struct InstrumentationGuard;

impl InstrumentationGuard {
    fn new() -> Self {
        reset_deterministic_provider_counters();
        enable_deterministic_provider_instrumentation();
        Self
    }
}

impl Drop for InstrumentationGuard {
    fn drop(&mut self) {
        disable_deterministic_provider_instrumentation();
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

// The `COUNTER_LOCK` guard must span the awaits — it serializes the entire
// test against any other test that also reads the global
// deterministic-provider counters. There is no blocking I/O under the lock.
#[allow(clippy::await_holding_lock)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn parallel_store_requests_do_not_serialize_on_embedder() {
    let _counter_lock = COUNTER_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let _instrumentation = InstrumentationGuard::new();

    let state = build_deterministic_state();
    let mut handles = Vec::new();
    for index in 0..5 {
        let state_clone = Arc::clone(&state);
        let params = json!({
            "memory_type": "decision",
            "context": format!("parallel-context-{index}"),
            "action": format!("parallel-action-{index}"),
            "result": format!("parallel-result-{index}"),
        });
        handles.push(tokio::spawn(async move {
            dispatch::route("memory_store", &state_clone, params).await
        }));
    }
    for handle in handles {
        handle
            .await
            .expect("task joined")
            .expect("dispatch succeeded");
    }

    let max_concurrent = deterministic_provider_max_concurrent();
    assert!(
        max_concurrent >= 2,
        "expected concurrent embedder calls, got max={max_concurrent}"
    );
}

// `memory_search` takes the same embedder path as `memory_store` via
// `config.build_embedding_provider()` inside a `spawn_blocking` closure.
// This test confirms that concurrent search requests also run the provider
// in parallel (no accidental serialization behind a shared mutex).
#[allow(clippy::await_holding_lock)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn parallel_search_requests_do_not_serialize_on_embedder() {
    let _counter_lock = COUNTER_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let _instrumentation = InstrumentationGuard::new();

    let state = build_deterministic_state();
    let mut handles = Vec::new();
    for index in 0..5 {
        let state_clone = Arc::clone(&state);
        let params = json!({
            "query": format!("parallel-search-query-{index}"),
            "limit": 3,
        });
        handles.push(tokio::spawn(async move {
            dispatch::route("memory_search", &state_clone, params).await
        }));
    }
    for handle in handles {
        handle
            .await
            .expect("task joined")
            .expect("dispatch succeeded");
    }

    let max_concurrent = deterministic_provider_max_concurrent();
    assert!(
        max_concurrent >= 2,
        "expected concurrent embedder calls during search, got max={max_concurrent}"
    );
}
