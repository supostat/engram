use std::sync::{Arc, Mutex, RwLock};

use serde_json::json;

use engram_core::config::{
    Config, deterministic_provider_max_concurrent, disable_deterministic_provider_instrumentation,
    enable_deterministic_provider_instrumentation, reset_deterministic_provider_counters,
};
use engram_core::dispatch;
use engram_core::indexes::IndexSet;
use engram_core::indexes::instrumentation::{
    self, disable_reader_tracking, enable_reader_tracking, reset_reader_counters,
};
use engram_core::server::ServerState;
use engram_embeddings::{Embedder, ThreeFieldEmbedding};
use engram_llm_client::{EmbeddingProvider, TextGenerator};
use engram_router::Router;
use engram_storage::Database;

// Serializes these tests against any other test that reads the global
// deterministic-provider counters. Poisoning is tolerated — the guard
// exists solely to prevent interleaving.
static COUNTER_LOCK: Mutex<()> = Mutex::new(());

// Tests that assert on `concurrent_readers_max >= 2` are timing-sensitive:
// coverage instrumentation (cargo-tarpaulin) slows each call enough that
// parallel readers complete sequentially and the peak overlap counter
// stays at 1. CI sets `ENGRAM_SKIP_TIMING_TESTS=1` for the coverage job;
// local `cargo test` keeps the assertion live.
fn skip_under_coverage() -> bool {
    std::env::var("ENGRAM_SKIP_TIMING_TESTS").is_ok()
}

// RAII guard: enables deterministic-provider + reader-tracking instrumentation
// on construction, disables both on drop (including panic). Counters are
// reset before enabling to clear any stale state from prior test runs.
struct InstrumentationGuard;

impl InstrumentationGuard {
    fn new() -> Self {
        reset_deterministic_provider_counters();
        reset_reader_counters();
        enable_deterministic_provider_instrumentation();
        enable_reader_tracking();
        Self
    }
}

impl Drop for InstrumentationGuard {
    fn drop(&mut self) {
        disable_reader_tracking();
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

fn seed_one_memory(state: &Arc<ServerState>) {
    let embedding = ThreeFieldEmbedding {
        context: vec![0.1; 1024],
        action: vec![0.1; 1024],
        result: vec![0.1; 1024],
    };
    let mut indexes = state.indexes.write().unwrap();
    indexes
        .insert_atomic(1, "seed-memory-id", &embedding, 0.5)
        .expect("seed insert");
}

// The `COUNTER_LOCK` guard must span the awaits — it serializes the entire
// test against any other test that also reads the global instrumentation
// counters. There is no blocking I/O under the lock.
#[allow(clippy::await_holding_lock)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn parallel_searches_overlap_with_rwlock() {
    if skip_under_coverage() {
        return;
    }
    let _counter_lock = COUNTER_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let _instrumentation = InstrumentationGuard::new();

    let state = build_deterministic_state();
    seed_one_memory(&state);

    let mut handles = Vec::new();
    for index in 0..10 {
        let state_clone = Arc::clone(&state);
        let params = json!({
            "query": format!("rwlock-search-query-{index}"),
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

    let embedder_max = deterministic_provider_max_concurrent();
    assert!(
        embedder_max >= 2,
        "expected concurrent embedder calls during search, got max={embedder_max}"
    );
    let reader_max = instrumentation::concurrent_readers_max();
    assert!(
        reader_max >= 2,
        "expected concurrent index readers under RwLock, got max={reader_max}"
    );
}

#[allow(clippy::await_holding_lock)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn search_continues_during_store_with_rwlock() {
    let _counter_lock = COUNTER_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let _instrumentation = InstrumentationGuard::new();

    let state = build_deterministic_state();
    seed_one_memory(&state);

    let mut handles = Vec::new();
    for index in 0..5 {
        let state_clone = Arc::clone(&state);
        let params = json!({
            "query": format!("rwlock-mixed-search-{index}"),
            "limit": 3,
        });
        handles.push(tokio::spawn(async move {
            dispatch::route("memory_search", &state_clone, params).await
        }));
    }
    {
        let state_clone = Arc::clone(&state);
        let params = json!({
            "memory_type": "decision",
            "context": "rwlock-mixed-context",
            "action": "rwlock-mixed-action",
            "result": "rwlock-mixed-result",
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
}
