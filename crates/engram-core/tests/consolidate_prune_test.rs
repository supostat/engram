use std::sync::{Arc, Mutex, RwLock};

use serde_json::json;

use engram_core::config::Config;
use engram_core::dispatch;
use engram_core::indexes::IndexSet;
use engram_core::lock_helpers;
use engram_core::persistence::hash_string_to_u64;
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

fn index_contains(state: &Arc<ServerState>, id: &str) -> bool {
    let indexes = lock_helpers::read_indexes(state);
    indexes.contains(hash_string_to_u64(id))
}

// Two stores sharing only their context (action/result distinct) clear the FTS
// duplicate pass without merging at store time under the all-three dedup policy, so
// both are indexed into the HNSW set. Driving memory_consolidate_apply merges the
// heuristic loser into the survivor and the handler must prune the merged target's
// node out of the HNSW index — a stale node left behind would resurface a superseded
// memory on vector search.
#[tokio::test]
async fn consolidate_apply_prunes_merged_target_from_hnsw() {
    let state = build_deterministic_state();

    let shared_context = "consolidation prune shared context tokens";
    let first = dispatch::route(
        "memory_store",
        &state,
        json!({
            "memory_type": "decision",
            "context": shared_context,
            "action": "first distinct action describing one specific operational procedure",
            "result": "first distinct result capturing a particular measured production outcome",
        }),
    )
    .await
    .expect("first store");
    let first_id = first["id"].as_str().expect("first id").to_string();

    let second = dispatch::route(
        "memory_store",
        &state,
        json!({
            "memory_type": "decision",
            "context": shared_context,
            "action": "second distinct action describing an entirely different remediation workflow",
            "result": "second distinct result documenting an unrelated downstream metric recovery",
        }),
    )
    .await
    .expect("second store");
    let second_id = second["id"].as_str().expect("second id").to_string();

    assert_ne!(
        first_id, second_id,
        "context-only overlap must not dedup at store time"
    );
    assert!(
        index_contains(&state, &first_id) && index_contains(&state, &second_id),
        "both rows must be indexed into the HNSW set before consolidation"
    );

    dispatch::route("memory_consolidate_apply", &state, json!({}))
        .await
        .expect("consolidate apply succeeds");

    // Exactly one of the two becomes the survivor; the merged target must be pruned.
    let survivor_id = {
        let database = lock_helpers::lock_db(&state);
        let first_row = database.get_memory(&first_id).expect("first row");
        let second_row = database.get_memory(&second_id).expect("second row");
        let merged_count = usize::from(first_row.superseded_by.is_some())
            + usize::from(second_row.superseded_by.is_some());
        assert_eq!(merged_count, 1, "exactly one row must be superseded");
        if first_row.superseded_by.is_some() {
            second_id.clone()
        } else {
            first_id.clone()
        }
    };
    let merged_id = if survivor_id == first_id {
        second_id.clone()
    } else {
        first_id.clone()
    };

    assert!(
        !index_contains(&state, &merged_id),
        "the merged target's node must be pruned from the HNSW index"
    );
    assert!(
        index_contains(&state, &survivor_id),
        "the surviving node must remain in the HNSW index"
    );
}
