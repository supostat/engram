use std::sync::{Arc, Mutex, RwLock};

use serde_json::json;

use engram_core::config::Config;
use engram_core::dispatch;
use engram_core::indexes::IndexSet;
use engram_core::server::ServerState;
use engram_embeddings::Embedder;
use engram_llm_client::openai::instrumentation as openai_instr;
use engram_llm_client::voyage::instrumentation as voyage_instr;
use engram_llm_client::{
    EmbeddingProvider, OpenAITextGenerator, RetryConfig, TextGenerator, VoyageEmbeddingProvider,
};
use engram_router::Router;
use engram_storage::Database;

// Serializes tests that read or mutate the global construction counters in
// voyage/openai instrumentation. Poisoning is tolerated — the guard exists
// only to prevent counter interleaving with other tests.
static COUNTER_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn voyage_construction_counter_increments_once_per_with_config() {
    let _guard = COUNTER_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    voyage_instr::reset_client_construction_count();
    voyage_instr::enable_client_construction_tracking();

    let provider = VoyageEmbeddingProvider::with_config(
        "test-key".into(),
        "voyage-code-3".into(),
        1024,
        RetryConfig::default(),
        "http://localhost:1".into(),
    )
    .expect("provider");
    assert_eq!(voyage_instr::client_construction_count(), 1);

    let arc: Arc<dyn EmbeddingProvider + Send + Sync> = Arc::new(provider);
    for _ in 0..100 {
        let _clone = Arc::clone(&arc);
    }
    assert_eq!(voyage_instr::client_construction_count(), 1);

    voyage_instr::disable_client_construction_tracking();
}

#[test]
fn openai_construction_counter_increments_once_per_with_config() {
    let _guard = COUNTER_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    openai_instr::reset_client_construction_count();
    openai_instr::enable_client_construction_tracking();

    let generator = OpenAITextGenerator::with_config(
        "test-key".into(),
        "gpt-4o-mini".into(),
        RetryConfig::default(),
        "http://localhost:1".into(),
    )
    .expect("generator");
    assert_eq!(openai_instr::client_construction_count(), 1);

    let arc: Arc<dyn TextGenerator + Send + Sync> = Arc::new(generator);
    for _ in 0..100 {
        let _clone = Arc::clone(&arc);
    }
    assert_eq!(openai_instr::client_construction_count(), 1);

    openai_instr::disable_client_construction_tracking();
}

// The `COUNTER_LOCK` guard must span the awaits — it serializes the entire
// test against any other test that reads the global construction counters.
// There is no blocking I/O under the lock.
#[allow(clippy::await_holding_lock)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dispatch_search_does_not_construct_voyage_per_call() {
    let _guard = COUNTER_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    voyage_instr::reset_client_construction_count();
    voyage_instr::enable_client_construction_tracking();

    // Construct (and drop) the Voyage client inside spawn_blocking. Voyage's
    // reqwest::blocking::Client owns an internal tokio runtime, which panics
    // if dropped from an async context.
    tokio::task::spawn_blocking(|| {
        let _voyage = VoyageEmbeddingProvider::with_config(
            "test-key".into(),
            "voyage-code-3".into(),
            1024,
            RetryConfig::default(),
            "http://localhost:1".into(),
        )
        .expect("provider");
    })
    .await
    .expect("voyage construct task");
    assert_eq!(voyage_instr::client_construction_count(), 1);

    let state = build_deterministic_state();
    for index in 0..100 {
        let params = json!({ "query": format!("unique-query-{index}") });
        let _ = dispatch::route("memory_search", &state, params).await;
    }
    assert_eq!(voyage_instr::client_construction_count(), 1);

    voyage_instr::disable_client_construction_tracking();
}

#[test]
fn initialize_state_succeeds_with_text_generator_unavailable() {
    let mut config = Config::default();
    config.embedding.provider = "deterministic".into();
    config.llm.api_key = None;

    let embedding_provider: Arc<dyn EmbeddingProvider + Send + Sync> =
        Arc::from(config.build_embedding_provider().expect("provider"));
    let text_generator: Option<Arc<dyn TextGenerator + Send + Sync>> =
        config.build_text_generator().ok().map(Arc::from);
    assert!(text_generator.is_none(), "openai without key must Err");

    let database = Database::in_memory().expect("db");
    let indexes = IndexSet::new(|| config.build_hnsw_params()).expect("indexes");
    let _state = ServerState {
        database: Mutex::new(database),
        indexes: RwLock::new(indexes),
        embedder: Embedder::new(0),
        router: Mutex::new(Router::new(0.1, 0.15)),
        config,
        database_path: String::new(),
        embedding_provider,
        text_generator,
    };
}

#[test]
fn initialize_state_fails_with_empty_voyage_key() {
    let mut config = Config::default();
    config.embedding.provider = "voyage".into();
    config.embedding.api_key = Some(String::new());
    let result = config.build_embedding_provider();
    assert!(result.is_err(), "empty voyage api_key must Err");
}

#[test]
fn judge_with_text_generator_via_arc_coercion_compiles() {
    // Sanity-check trait coercion: Arc<dyn _ + Send + Sync> exposes via
    // .as_deref() a &(dyn _ + Send + Sync), which coerces to &dyn _ where
    // call-sites need the bare trait object (e.g. CombinedJudge::with_llm).
    let _guard = COUNTER_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let generator = OpenAITextGenerator::with_config(
        "test-key".into(),
        "gpt-4o-mini".into(),
        RetryConfig::default(),
        "http://localhost:1".into(),
    )
    .expect("generator");
    let opt: Option<Arc<dyn TextGenerator + Send + Sync>> = Some(Arc::new(generator));
    let coerced: Option<&(dyn TextGenerator + Send + Sync)> = opt.as_deref();
    let _bare: Option<&dyn TextGenerator> =
        coerced.map(|generator| generator as &dyn TextGenerator);
}

fn build_deterministic_state() -> Arc<ServerState> {
    let database = Database::in_memory().expect("db");
    let mut config = Config::default();
    config.embedding.provider = "deterministic".into();
    let indexes = IndexSet::new(|| config.build_hnsw_params()).expect("indexes");
    let embedding_provider: Arc<dyn EmbeddingProvider + Send + Sync> =
        Arc::from(config.build_embedding_provider().expect("provider"));
    let text_generator: Option<Arc<dyn TextGenerator + Send + Sync>> =
        config.build_text_generator().ok().map(Arc::from);
    Arc::new(ServerState {
        database: Mutex::new(database),
        indexes: RwLock::new(indexes),
        embedder: Embedder::new(0),
        router: Mutex::new(Router::new(0.1, 0.15)),
        config,
        database_path: String::new(),
        embedding_provider,
        text_generator,
    })
}
