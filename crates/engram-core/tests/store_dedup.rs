use std::sync::{Arc, Mutex, RwLock};

use serde_json::{Value, json};

use engram_core::config::Config;
use engram_core::dispatch;
use engram_core::indexes::IndexSet;
use engram_core::lock_helpers;
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

// Builds a state whose embedding provider emits vectors of one dimension while
// the HNSW index expects another. The dedup search inside `find_duplicate`
// queries the HNSW graphs and fails with a `DimensionMismatch`, which the
// handler must swallow (logging a degraded-dedup warning) and proceed to a
// normal insert reporting `deduplicated: false`.
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

// The deterministic provider embeds each field's text independently and yields
// byte-identical (hence cosine-similarity 1.0) vectors for identical text. Two
// stores built from the same marker therefore produce three pairwise-equal
// field embeddings, clearing the all-three-fields dedup policy at any threshold
// in (0.0, 1.0].
fn make_store_params(marker: &str) -> Value {
    json!({
        "memory_type": "decision",
        "context": format!("dedup context {marker}"),
        "action": format!("dedup action {marker}"),
        "result": format!("dedup result {marker}"),
    })
}

// A memory whose every field differs from `make_store_params`. The deterministic
// provider sums each byte into dimension `index % dimension`, so it concentrates
// signal in the low-index positions and overlaps short, similar-length strings
// heavily (a different trailing word still scores ~0.999 — well above 0.95). To
// build genuinely distinct embeddings under that fixture, each field here is a
// long, fully unrelated sentence: the large length and byte difference spread
// the vector across many positions, dropping every pairwise cosine similarity
// far below the 0.95 threshold so none of the three fields qualifies.
fn make_distinct_store_params() -> Value {
    json!({
        "memory_type": "decision",
        "context": "an entirely unrelated situation describing prolonged offline batch ingestion of \
                     historical climate telemetry gathered from remote arctic weather stations",
        "action": "the operator reconfigured the distributed message broker partition assignment and \
                    rebalanced consumer groups to drain a multi-day processing backlog gradually",
        "result": "throughput recovered to nominal levels and downstream dashboards reflected the \
                    corrected aggregates after the lengthy catch-up window finally completed cleanly",
    })
}

// Shares `make_store_params(marker)`'s CONTEXT text verbatim, but pairs it with
// action/result drawn from the fully-distinct fixture above. Under the
// all-three-fields dedup policy only the context field clears the threshold, so
// the store must NOT merge: a context match alone is insufficient.
fn make_shared_context_store_params(marker: &str) -> Value {
    json!({
        "memory_type": "decision",
        "context": format!("dedup context {marker}"),
        "action": "the operator reconfigured the distributed message broker partition assignment and \
                    rebalanced consumer groups to drain a multi-day processing backlog gradually",
        "result": "throughput recovered to nominal levels and downstream dashboards reflected the \
                    corrected aggregates after the lengthy catch-up window finally completed cleanly",
    })
}

async fn store(state: &Arc<ServerState>, params: Value) -> Value {
    dispatch::route("memory_store", state, params)
        .await
        .expect("memory_store")
}

fn row_count(state: &Arc<ServerState>) -> usize {
    lock_helpers::lock_db(state)
        .list_all_memories()
        .expect("list memories")
        .len()
}

// Happy path: storing a memory with field text identical to an already-stored
// memory does NOT create a second row. The second store reports
// `deduplicated: true`, folds into the first id, and bumps that row's
// used_count via touch_memory. This is the precise regression guard against the
// old always-insert behavior: under that logic the second store would insert a
// new row, leaving the DB with TWO rows and the first row's used_count at 0.
#[tokio::test]
async fn identical_memory_deduplicates_into_existing_row() {
    let state = build_deterministic_state();

    let first = store(&state, make_store_params("twin")).await;
    let first_id = first["id"].as_str().expect("first id").to_string();
    assert_eq!(
        first["deduplicated"],
        json!(false),
        "the first store of a marker is never a duplicate"
    );
    assert_eq!(row_count(&state), 1, "first store inserts exactly one row");

    let second = store(&state, make_store_params("twin")).await;
    assert_eq!(
        second["deduplicated"],
        json!(true),
        "a byte-identical second store must deduplicate"
    );
    assert_eq!(
        second["merged_into"].as_str(),
        Some(first_id.as_str()),
        "the duplicate must merge into the original id"
    );
    assert_eq!(
        second["id"].as_str(),
        Some(first_id.as_str()),
        "the returned id must be the surviving original"
    );

    assert_eq!(
        row_count(&state),
        1,
        "deduplication must leave exactly one row (old always-insert path leaves two)"
    );
    let survivor = lock_helpers::lock_db(&state)
        .get_memory(&first_id)
        .expect("surviving row");
    assert_eq!(
        survivor.used_count, 1,
        "touch_memory must increment the survivor's used_count to 1"
    );
    assert!(
        survivor.last_used_at.is_some(),
        "touch_memory must stamp last_used_at on the survivor"
    );
}

// Below threshold: a store whose context, action AND result text all differ
// from the existing memory clears none of the all-three fields, so it is a
// genuinely new memory — `deduplicated: false`, a fresh id, and a second row.
#[tokio::test]
async fn distinct_memory_is_not_deduplicated() {
    let state = build_deterministic_state();

    let first = store(&state, make_store_params("alpha")).await;
    let first_id = first["id"].as_str().expect("first id").to_string();

    let second = store(&state, make_distinct_store_params()).await;
    let second_id = second["id"].as_str().expect("second id").to_string();
    assert_eq!(
        second["deduplicated"],
        json!(false),
        "a fully distinct memory must not deduplicate"
    );
    assert!(
        second.get("merged_into").is_none(),
        "merged_into must be omitted when deduplicated is false"
    );
    assert_ne!(
        first_id, second_id,
        "distinct memories must receive distinct ids"
    );
    assert_eq!(
        row_count(&state),
        2,
        "two distinct memories must produce two rows"
    );
}

// Degraded dedup: when the dedup search errors internally (here, an embedding /
// HNSW dimension mismatch makes `find_duplicate` fail), the store must not panic
// and must fall through to a normal insert reporting `deduplicated: false`.
#[tokio::test]
async fn dedup_search_error_degrades_to_insert() {
    let state = build_dimension_mismatch_state();

    let embedding_dimension = state.embedding_provider.dimension();
    let hnsw_dimension = state.config.hnsw.dimension;
    assert_ne!(
        embedding_dimension, hnsw_dimension,
        "embedding and HNSW dimensions must differ to force a dedup search error"
    );

    let stored = store(&state, make_store_params("degraded")).await;
    assert_eq!(
        stored["deduplicated"],
        json!(false),
        "a failed dedup search must degrade to a normal non-dedup insert"
    );
    assert!(
        stored["id"].as_str().is_some(),
        "the store must still return a fresh memory id"
    );
    assert_eq!(
        row_count(&state),
        1,
        "the degraded store must still persist the row"
    );
}

// Context-only match: memory B reuses A's CONTEXT text verbatim while its action
// and result are fully distinct. The all-three-fields policy requires every
// field to clear the threshold, so a single matching field must NOT trigger a
// merge — B is a genuinely new memory.
#[tokio::test]
async fn shared_context_alone_does_not_deduplicate() {
    let state = build_deterministic_state();

    let first = store(&state, make_store_params("shared")).await;
    let first_id = first["id"].as_str().expect("first id").to_string();
    assert_eq!(row_count(&state), 1, "first store inserts exactly one row");

    let second = store(&state, make_shared_context_store_params("shared")).await;
    let second_id = second["id"].as_str().expect("second id").to_string();
    assert_eq!(
        second["deduplicated"],
        json!(false),
        "a context-only match must not deduplicate under the all-three-fields policy"
    );
    assert!(
        second.get("merged_into").is_none(),
        "merged_into must be omitted when deduplicated is false"
    );
    assert_ne!(
        first_id, second_id,
        "the non-merged memory must receive a distinct id"
    );
    assert_eq!(
        row_count(&state),
        2,
        "matching context alone must still produce two rows"
    );
}
