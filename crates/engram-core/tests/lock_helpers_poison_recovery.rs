//! Regression guard for [`engram_core::lock_helpers`].
//!
//! The pre-Phase-8 code acquired every `ServerState` lock with a raw
//! `.lock().unwrap()` / `.write().unwrap()`. A panic while holding such a lock
//! poisons it, and the next `.unwrap()` panics in turn — cascading a single
//! recoverable hiccup into a daemon-wide crash. The helpers recover via
//! `PoisonError::into_inner`, so a poisoned lock must still hand back a usable
//! guard. These tests poison the real locks and assert the helpers recover.

use std::sync::{Arc, Mutex, RwLock};

use engram_core::config::Config;
use engram_core::indexes::IndexSet;
use engram_core::lock_helpers::{lock_db, lock_router, read_indexes, write_indexes};
use engram_core::server::ServerState;
use engram_embeddings::Embedder;
use engram_llm_client::{EmbeddingProvider, TextGenerator};
use engram_router::{Mode, Router};
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

#[test]
fn lock_db_recovers_after_a_holder_thread_panics() {
    let state = build_deterministic_state();

    let poisoning_state = Arc::clone(&state);
    let panicked = std::thread::spawn(move || {
        let _guard = poisoning_state.database.lock().unwrap();
        panic!("intentional panic while holding the database mutex");
    })
    .join();
    assert!(panicked.is_err(), "holder thread should have panicked");
    assert!(
        state.database.is_poisoned(),
        "database mutex should be poisoned after the holder panicked"
    );

    let database = lock_db(&state);
    database
        .list_all_memories()
        .expect("recovered database guard must be usable");
}

#[test]
fn write_indexes_recovers_after_a_holder_thread_panics() {
    let state = build_deterministic_state();

    let poisoning_state = Arc::clone(&state);
    let panicked = std::thread::spawn(move || {
        let _guard = poisoning_state.indexes.write().unwrap();
        panic!("intentional panic while holding the indexes write lock");
    })
    .join();
    assert!(panicked.is_err(), "holder thread should have panicked");
    assert!(
        state.indexes.is_poisoned(),
        "indexes lock should be poisoned after the holder panicked"
    );

    let indexes = write_indexes(&state);
    let _ = indexes.len();
}

#[test]
fn lock_router_recovers_after_a_holder_thread_panics() {
    let state = build_deterministic_state();

    let poisoning_state = Arc::clone(&state);
    let panicked = std::thread::spawn(move || {
        let _guard = poisoning_state.router.lock().unwrap();
        panic!("intentional panic while holding the router mutex");
    })
    .join();
    assert!(panicked.is_err(), "holder thread should have panicked");
    assert!(
        state.router.is_poisoned(),
        "router mutex should be poisoned after the holder panicked"
    );

    let router = lock_router(&state);
    let decision = router.decide(Mode::Coding, 0.5);
    assert_eq!(
        decision.mode,
        Mode::Coding,
        "recovered router guard must produce a usable decision"
    );
}

#[test]
fn read_indexes_recovers_after_a_holder_thread_panics() {
    let state = build_deterministic_state();

    // An `RwLock` is poisoned only by a panic under a *write* guard — read
    // guards never poison. We poison via the write side, then assert the read
    // accessor still hands back a usable shared guard.
    let poisoning_state = Arc::clone(&state);
    let panicked = std::thread::spawn(move || {
        let _guard = poisoning_state.indexes.write().unwrap();
        panic!("intentional panic while holding the indexes write lock");
    })
    .join();
    assert!(panicked.is_err(), "holder thread should have panicked");
    assert!(
        state.indexes.is_poisoned(),
        "indexes lock should be poisoned after the holder panicked"
    );

    let indexes = read_indexes(&state);
    let _ = indexes.len();
}
