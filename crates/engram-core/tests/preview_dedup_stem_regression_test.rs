use std::sync::{Arc, Mutex, RwLock};

use serde_json::json;

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

async fn store(state: &Arc<ServerState>, context: &str, action: &str, result: &str) -> String {
    let stored = dispatch::route(
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
    .expect("store");
    stored["id"].as_str().expect("id").to_string()
}

// The OR-of-prefix FTS rewrite widens the dedup *candidate* probe, but two memories
// that share only a common character prefix ("pay") — with no shared whole token and
// no shared porter stem — must NOT be pulled into the same duplicate group and must
// NOT be silently auto-merged by `memory_consolidate_apply`. In the deterministic
// (no-LLM) path the heuristic would merge anything that groups, so this proves the
// probe itself does not over-group on a bare prefix overlap.
#[tokio::test]
async fn stem_prefix_overlap_does_not_auto_merge() {
    let state = build_deterministic_state();

    let payment_id = store(
        &state,
        "payment processing gateway",
        "charge the customer card through the provider",
        "transaction settled and receipt issued",
    )
    .await;
    let payload_id = store(
        &state,
        "payload encoding pipeline",
        "serialize the message body before transport",
        "bytes framed and queued for delivery",
    )
    .await;

    assert_ne!(
        payment_id, payload_id,
        "the two rows must not dedup at store time"
    );

    dispatch::route("memory_consolidate_apply", &state, json!({}))
        .await
        .expect("consolidate apply succeeds");

    {
        let database = lock_helpers::lock_db(&state);
        let payment = database.get_memory(&payment_id).expect("payment row");
        let payload = database.get_memory(&payload_id).expect("payload row");
        assert!(
            payment.superseded_by.is_none(),
            "the payment row must survive — a shared 'pay' prefix is not a duplicate"
        );
        assert!(
            payload.superseded_by.is_none(),
            "the payload row must survive — a shared 'pay' prefix is not a duplicate"
        );
    }

    // No merge happened, so the audit log stays empty.
    let log = dispatch::route("memory_consolidate_log", &state, json!({}))
        .await
        .expect("log should succeed");
    assert_eq!(
        log["count"], 0,
        "a prefix-only overlap must not produce a merge entry"
    );
}
