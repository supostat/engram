use std::sync::{Arc, Mutex, RwLock};

use serde_json::json;

use engram_core::config::Config;
use engram_core::dispatch;
use engram_core::indexes::IndexSet;
use engram_core::server::ServerState;
use engram_embeddings::Embedder;
use engram_llm_client::{ApiError, EmbeddingProvider, TextGenerator};
use engram_router::Router;
use engram_storage::{Database, Memory};

struct UnavailableEmbeddingProvider;

impl EmbeddingProvider for UnavailableEmbeddingProvider {
    fn embed(&self, _text: &str, _input_type: Option<&str>) -> Result<Vec<f32>, ApiError> {
        Err(ApiError::EmbeddingApiUnavailable(
            "provider down for test".into(),
        ))
    }

    fn dimension(&self) -> usize {
        1024
    }

    fn model_name(&self) -> &str {
        "unavailable-test-provider"
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

fn build_degraded_state() -> Arc<ServerState> {
    let database = Database::in_memory().expect("in-memory database");
    let mut config = Config::default();
    config.embedding.provider = "deterministic".into();
    let indexes = IndexSet::new(|| config.build_hnsw_params()).expect("index set");
    let embedder = Embedder::new(0);
    let router = Router::new(0.1, 0.15);
    let embedding_provider: Arc<dyn EmbeddingProvider + Send + Sync> =
        Arc::new(UnavailableEmbeddingProvider);
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

async fn store_memory(state: &Arc<ServerState>, context: &str) -> String {
    let stored = dispatch::route(
        "memory_store",
        state,
        json!({
            "memory_type": "decision",
            "context": context,
            "action": "act",
            "result": "res",
        }),
    )
    .await
    .expect("store memory");
    stored["id"].as_str().expect("id").to_string()
}

async fn search(state: &Arc<ServerState>, query: &str) -> serde_json::Value {
    dispatch::route(
        "memory_search",
        state,
        json!({ "query": query, "limit": 5 }),
    )
    .await
    .expect("search")
}

fn ids_of(response: &serde_json::Value) -> Vec<String> {
    response["results"]
        .as_array()
        .expect("results array")
        .iter()
        .map(|entry| entry["id"].as_str().expect("id").to_string())
        .collect()
}

#[tokio::test]
async fn search_writes_exactly_one_routing_log_row() {
    let state = build_deterministic_state();
    store_memory(&state, "routing alpha context").await;
    store_memory(&state, "routing beta context").await;

    let response = search(&state, "routing context").await;
    assert!(response["results"].is_array());
    assert_eq!(response["degraded"], json!(false));

    let database = state.database.lock().unwrap();
    let row_count: i64 = database
        .connection()
        .query_row("SELECT COUNT(*) FROM routing_log", [], |row| row.get(0))
        .expect("count routing_log");
    assert_eq!(row_count, 1, "exactly one routing_log row per search");

    let (mode, search_strategy, llm_selection, contextualization, proactivity, top_k): (
        String,
        String,
        String,
        String,
        String,
        i64,
    ) = database
        .connection()
        .query_row(
            "SELECT mode, search_strategy, llm_selection, contextualization, proactivity, top_k \
             FROM routing_log",
            [],
            |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                ))
            },
        )
        .expect("read routing_log row");

    // The query "routing context" carries no mode keyword, so detection falls
    // through to Routine. With Router::new(0.1, 0.15) and the served rng 0.5
    // (>= epsilon 0.15) the choice is exploit; the Q-tables are empty so every
    // level resolves to Routine's static ModeDefaults. These values are fully
    // deterministic — a field-swap or default-drift regression is caught here.
    assert_eq!(mode, "routine", "query resolves to Routine mode");
    assert_eq!(search_strategy, "high_threshold");
    assert_eq!(llm_selection, "cheap");
    assert_eq!(contextualization, "raw");
    assert_eq!(proactivity, "passive");
    assert!(top_k > 0, "served top_k must be recorded");
}

#[tokio::test]
async fn feedback_tracking_rows_carry_the_search_query_id() {
    let state = build_deterministic_state();
    store_memory(&state, "query id stamp context").await;

    search(&state, "query id stamp").await;

    let database = state.database.lock().unwrap();
    let query_id: String = database
        .connection()
        .query_row("SELECT query_id FROM routing_log", [], |row| row.get(0))
        .expect("routing_log query_id");

    let mut statement = database
        .connection()
        .prepare("SELECT query_id FROM feedback_tracking")
        .expect("prepare feedback_tracking");
    let stamped: Vec<Option<String>> = statement
        .query_map([], |row| row.get::<_, Option<String>>(0))
        .expect("query feedback_tracking")
        .map(|row| row.expect("row"))
        .collect();
    assert!(
        !stamped.is_empty(),
        "the search must have tracked at least one shown memory"
    );
    for tracked in &stamped {
        assert_eq!(
            tracked.as_deref(),
            Some(query_id.as_str()),
            "every feedback_tracking row from this search must carry the search's query_id"
        );
    }
}

#[tokio::test]
async fn shadow_rewards_json_is_monotonic_non_decreasing_in_k() {
    let state = build_deterministic_state();
    for index in 0..6 {
        store_memory(&state, &format!("shadow context number {index}")).await;
    }

    search(&state, "shadow context number").await;

    let database = state.database.lock().unwrap();
    let shadow_json: String = database
        .connection()
        .query_row("SELECT shadow_rewards FROM routing_log", [], |row| {
            row.get(0)
        })
        .expect("shadow_rewards column");

    let parsed: Vec<serde_json::Value> =
        serde_json::from_str(&shadow_json).expect("shadow_rewards must be valid JSON");
    assert!(!parsed.is_empty(), "shadow rewards must be populated");

    let mut previous = f64::NEG_INFINITY;
    for entry in &parsed {
        let k = entry["k"].as_u64().expect("k is an integer");
        let reward = entry["reward"].as_f64().expect("reward is a float");
        assert!(k > 0, "k must be positive");
        assert!(
            (0.0..=1.0 + 1e-9).contains(&reward),
            "reward must be a coverage fraction in [0,1]: {reward}"
        );
        assert!(
            reward + 1e-9 >= previous,
            "shadow reward must be monotonic non-decreasing in k: {reward} < {previous}"
        );
        previous = reward;
    }
}

#[tokio::test]
async fn serving_is_identical_across_repeated_search() {
    let state = build_deterministic_state();
    let stored_ids = [
        store_memory(&state, "serving identity alpha").await,
        store_memory(&state, "serving identity beta").await,
        store_memory(&state, "serving identity gamma").await,
    ];

    let first = search(&state, "serving identity").await;
    let second = search(&state, "serving identity").await;

    let first_ids = ids_of(&first);
    let second_ids = ids_of(&second);
    assert_eq!(
        first_ids, second_ids,
        "instrumentation must not perturb the served result order"
    );
    assert_eq!(first["degraded"], second["degraded"]);

    // The served set must be a duplicate-free, non-empty subset of what was
    // stored — a regression that injected a phantom id, dropped every result,
    // or duplicated a row would be caught here. The exact id list is also
    // pinned below so an ordering/count change is caught deterministically.
    assert!(!first_ids.is_empty(), "search must return results");
    let mut deduplicated = first_ids.clone();
    deduplicated.sort();
    deduplicated.dedup();
    assert_eq!(
        deduplicated.len(),
        first_ids.len(),
        "served results must not contain duplicate ids"
    );
    for served in &first_ids {
        assert!(
            stored_ids.contains(served),
            "served id {served} must be one of the stored memories"
        );
    }
    assert_eq!(
        first_ids,
        vec![stored_ids[0].clone()],
        "the deterministic embedder + fixed store order pins the served ids; \
         an ordering or count regression would change this list"
    );

    // Scores must be sorted descending — the fused ranking is intact.
    let scores: Vec<f64> = first["results"]
        .as_array()
        .expect("results array")
        .iter()
        .map(|entry| entry["score"].as_f64().expect("score"))
        .collect();
    for window in scores.windows(2) {
        assert!(
            window[0] >= window[1],
            "served results must stay sorted by score desc: {window:?}"
        );
    }
}

#[tokio::test]
async fn empty_result_search_still_logs_one_row_without_shadow_rewards() {
    let state = build_deterministic_state();

    let response = search(&state, "nothing was ever stored here").await;
    assert!(
        response["results"]
            .as_array()
            .expect("results array")
            .is_empty(),
        "empty corpus yields no results"
    );

    let database = state.database.lock().unwrap();
    let row_count: i64 = database
        .connection()
        .query_row("SELECT COUNT(*) FROM routing_log", [], |row| row.get(0))
        .expect("count routing_log");
    assert_eq!(
        row_count, 1,
        "an empty-result search still writes exactly one routing_log row"
    );

    let shadow_rewards: Option<String> = database
        .connection()
        .query_row("SELECT shadow_rewards FROM routing_log", [], |row| {
            row.get(0)
        })
        .expect("shadow_rewards column");
    let per_k_entries = shadow_rewards.map_or_else(Vec::new, |json| {
        serde_json::from_str::<Vec<serde_json::Value>>(&json)
            .expect("shadow_rewards must be valid JSON when present")
    });
    assert!(
        per_k_entries.is_empty(),
        "an empty merged result set carries no per-k shadow rewards: {per_k_entries:?}"
    );
}

fn seed_memory_directly(state: &Arc<ServerState>, id: &str, context: &str) {
    let memory = Memory {
        id: id.to_string(),
        memory_type: "decision".to_string(),
        context: context.to_string(),
        action: "act".to_string(),
        result: "res".to_string(),
        score: 0.0,
        embedding_context: None,
        embedding_action: None,
        embedding_result: None,
        indexed: false,
        tags: None,
        project: None,
        parent_id: None,
        source_ids: None,
        insight_type: None,
        created_at: "2026-05-01T00:00:00Z".to_string(),
        updated_at: "2026-05-01T00:00:00Z".to_string(),
        used_count: 0,
        last_used_at: None,
        superseded_by: None,
    };
    let database = state.database.lock().unwrap();
    database.insert_memory(&memory).expect("seed memory");
}

#[tokio::test]
async fn degraded_search_still_logs_exactly_one_row() {
    // The store route needs a working embedder, so seed the row directly; the
    // failing embedder forces the search down the degraded FTS fallback path.
    let state = build_degraded_state();
    seed_memory_directly(&state, "degraded-1", "degraded path context");

    let response = search(&state, "degraded path").await;
    assert_eq!(
        response["degraded"],
        json!(true),
        "an unavailable embedder degrades the search"
    );

    let database = state.database.lock().unwrap();
    let row_count: i64 = database
        .connection()
        .query_row("SELECT COUNT(*) FROM routing_log", [], |row| row.get(0))
        .expect("count routing_log");
    assert_eq!(
        row_count, 1,
        "the degraded shadow path still writes exactly one routing_log row"
    );
}
