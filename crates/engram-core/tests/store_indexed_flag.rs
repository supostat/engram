use std::sync::{Arc, Mutex, RwLock};

use serde_json::{Value, json};

use engram_core::config::Config;
use engram_core::dispatch;
use engram_core::indexes::IndexSet;
use engram_core::lock_helpers;
use engram_core::persistence::hash_string_to_u64;
use engram_core::server::{ServerState, reindex_unindexed_memories};
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

// Builds a state whose embedding provider emits vectors of one dimension while
// the HNSW index expects another. The SQLite write succeeds, but the HNSW
// `insert_atomic` fails with a non-collision `DimensionMismatch` — a transient
// HNSW error that must NOT fail the store and must leave the row unindexed.
fn build_dimension_mismatch_state() -> Arc<ServerState> {
    let database = Database::in_memory().expect("in-memory database");
    let mut config = Config::default();
    config.embedding.provider = "deterministic".into();
    config.embedding.dimension = Some(16);
    config.hnsw.dimension = 32;
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

fn make_store_params(marker: &str) -> Value {
    json!({
        "memory_type": "decision",
        "context": format!("indexed-flag context {marker}"),
        "action": format!("indexed-flag action {marker}"),
        "result": format!("indexed-flag result {marker}"),
    })
}

async fn store(state: &Arc<ServerState>, marker: &str) -> Value {
    dispatch::route("memory_store", state, make_store_params(marker))
        .await
        .expect("memory_store")
}

async fn search(state: &Arc<ServerState>, marker: &str) -> Vec<Value> {
    dispatch::route(
        "memory_search",
        state,
        json!({ "query": format!("indexed-flag {marker}"), "limit": 10 }),
    )
    .await
    .expect("memory_search")["results"]
        .as_array()
        .expect("search returns array")
        .clone()
}

fn results_contain(results: &[Value], memory_id: &str) -> bool {
    results
        .iter()
        .any(|result| result["id"].as_str() == Some(memory_id))
}

fn index_contains(state: &Arc<ServerState>, memory_id: &str) -> bool {
    let indexes = lock_helpers::read_indexes(state);
    indexes.contains(hash_string_to_u64(memory_id))
}

// Happy path: a successful store marks the SQLite row indexed only AFTER the
// HNSW write is confirmed. The response reports `indexed: true`, the row is
// indexed=1, and the memory is searchable.
#[tokio::test]
async fn successful_store_marks_indexed_after_hnsw_write() {
    let state = build_deterministic_state();

    let stored = store(&state, "happy").await;
    let memory_id = stored["id"].as_str().expect("id").to_string();

    assert_eq!(
        stored["indexed"],
        json!(true),
        "response must report indexed=true on the success path"
    );

    let row_indexed = {
        let database = lock_helpers::lock_db(&state);
        database.get_memory(&memory_id).expect("row exists").indexed
    };
    assert!(row_indexed, "SQLite row must be indexed=1 after success");

    let results = search(&state, "happy").await;
    assert!(
        results_contain(&results, &memory_id),
        "stored memory must be searchable"
    );
}

// Recovery / Done-when (c): a record present in SQLite with indexed=0 but
// absent from the HNSW index (its indexing never completed) is recovered by
// the background reindex routine — afterwards the row is indexed=1 and the
// memory is searchable. Under the OLD optimistic logic (indexed=true before
// the HNSW write), such a record would never be selected by `WHERE indexed=0`
// and recovery would be impossible, so this test guards the regression.
#[tokio::test]
async fn background_reindex_recovers_unindexed_row() {
    let source = build_deterministic_state();
    let stored = store(&source, "recover").await;
    let memory_id = stored["id"].as_str().expect("id").to_string();
    let mut embedded_row = {
        let database = lock_helpers::lock_db(&source);
        database.get_memory(&memory_id).expect("embedded row")
    };
    assert!(
        embedded_row.embedding_context.is_some(),
        "deterministic provider yields embeddings"
    );

    // Fresh state: empty HNSW index, SQLite row inserted with indexed=false.
    // This models a memory whose SQLite write succeeded but whose HNSW write
    // never completed.
    let recovered = build_deterministic_state();
    embedded_row.indexed = false;
    {
        let database = lock_helpers::lock_db(&recovered);
        database
            .insert_memory(&embedded_row)
            .expect("insert unindexed row");
    }

    // The row was inserted into SQLite only — it is genuinely absent from the
    // HNSW (vector) index until reindex runs. Assert HNSW non-membership directly:
    // the widened FTS recall (OR-of-prefix) can surface the row via the sparse path
    // on shared literal tokens, so blended search results no longer isolate the
    // vector path this test guards.
    assert!(
        !index_contains(&recovered, &memory_id),
        "row absent from HNSW must not be in the vector index before reindex"
    );

    reindex_unindexed_memories(&recovered);

    assert!(
        index_contains(&recovered, &memory_id),
        "reindex must insert the recovered row into the HNSW index"
    );

    let row_indexed = {
        let database = lock_helpers::lock_db(&recovered);
        database.get_memory(&memory_id).expect("row exists").indexed
    };
    assert!(row_indexed, "reindex must mark the recovered row indexed=1");

    let after = search(&recovered, "recover").await;
    assert!(
        results_contain(&after, &memory_id),
        "recovered memory must be searchable after reindex"
    );
}

// Transient HNSW failure: the SQLite write succeeds but the HNSW write fails
// with a non-collision error. The store must NOT fail and must NOT roll back —
// the response reports `indexed: false` and the row stays indexed=0 so the
// background reindex can recover it later. This is the precise behavioral
// divergence from the old optimistic logic, which reported a constant
// `indexed: true` and never wrote the truthful flag — making such records
// invisible to `WHERE indexed=0` and recovery impossible.
#[tokio::test]
async fn transient_hnsw_failure_leaves_row_unindexed_without_failing_store() {
    let state = build_dimension_mismatch_state();

    // Self-validation: the test only exercises the transient-failure path if the
    // embedding provider really emits vectors of a different dimension than the
    // HNSW index expects. Assert the mismatch is actually wired up so the test
    // can never silently degrade into a happy-path store.
    let embedding_dimension = state.embedding_provider.dimension();
    let hnsw_dimension = state.config.hnsw.dimension;
    assert_eq!(
        embedding_dimension, 16,
        "embedding provider must emit 16-dim vectors for the mismatch setup"
    );
    assert_eq!(
        hnsw_dimension, 32,
        "HNSW index must expect 32-dim vectors for the mismatch setup"
    );
    assert_ne!(
        embedding_dimension, hnsw_dimension,
        "embedding and HNSW dimensions must differ to trigger the HNSW DimensionMismatch"
    );

    let stored = dispatch::route("memory_store", &state, make_store_params("transient"))
        .await
        .expect("store must not fail on transient HNSW error");
    let memory_id = stored["id"].as_str().expect("id").to_string();

    assert_eq!(
        stored["indexed"],
        json!(false),
        "transient HNSW failure must report indexed=false, not a constant true"
    );

    let database = lock_helpers::lock_db(&state);
    let row = database.get_memory(&memory_id).expect("row persisted");
    assert!(
        !row.indexed,
        "row must remain indexed=0 for background reindex after transient failure"
    );
    assert!(
        row.embedding_context.is_some(),
        "embeddings must be persisted so background reindex can recover the row"
    );
}
