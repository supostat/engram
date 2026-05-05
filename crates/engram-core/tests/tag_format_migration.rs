use std::sync::{Arc, Mutex, RwLock};

use rusqlite::params;
use serde_json::json;

use engram_core::config::Config;
use engram_core::dispatch;
use engram_core::indexes::IndexSet;
use engram_core::migrations::{self, TAGS_FORMAT_KEY, TAGS_FORMAT_TARGET_VALUE, tags_format_v1};
use engram_core::server::ServerState;
use engram_embeddings::Embedder;
use engram_llm_client::{EmbeddingProvider, TextGenerator};
use engram_router::Router;
use engram_storage::Database;

fn seed_raw_tag(database: &Database, id: &str, tags: Option<&str>) {
    let timestamp = "2026-05-01T00:00:00Z";
    database
        .connection()
        .execute(
            "INSERT INTO memories \
             (id, memory_type, context, action, result, score, indexed, tags, created_at, updated_at) \
             VALUES (?1, 'decision', 'ctx', 'act', 'res', 0.0, 0, ?2, ?3, ?3)",
            params![id, tags, timestamp],
        )
        .expect("seed memory");
}

fn read_tag(database: &Database, id: &str) -> Option<String> {
    database
        .connection()
        .query_row("SELECT tags FROM memories WHERE id = ?1", [id], |row| {
            row.get::<_, Option<String>>(0)
        })
        .expect("query tag")
}

fn read_meta(database: &Database, key: &str) -> Option<String> {
    use rusqlite::OptionalExtension;
    database
        .connection()
        .query_row(
            "SELECT value FROM schema_meta WHERE key = ?1",
            [key],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .expect("query meta")
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

#[test]
fn csv_stored_tag_converts_to_json() {
    let database = Database::in_memory().expect("in-memory db");
    seed_raw_tag(&database, "m1", Some("a,b,c"));
    let report = migrations::run_pending(&database).expect("migration runs");
    let stats = report.tags_format_v1.expect("ran first time");
    assert_eq!(stats.rewritten_csv, 1);
    assert_eq!(stats.rewritten_naked, 0);
    assert_eq!(
        read_tag(&database, "m1").as_deref(),
        Some(r#"["a","b","c"]"#)
    );
}

#[test]
fn naked_stored_tag_converts_to_json() {
    let database = Database::in_memory().expect("in-memory db");
    seed_raw_tag(&database, "m1", Some("rust"));
    let report = migrations::run_pending(&database).expect("migration runs");
    let stats = report.tags_format_v1.expect("ran first time");
    assert_eq!(stats.rewritten_naked, 1);
    assert_eq!(stats.rewritten_csv, 0);
    assert_eq!(read_tag(&database, "m1").as_deref(), Some(r#"["rust"]"#));
}

#[test]
fn json_stored_tag_is_noop() {
    let database = Database::in_memory().expect("in-memory db");
    seed_raw_tag(&database, "m1", Some(r#"["rust","bugfix"]"#));
    let report = migrations::run_pending(&database).expect("migration runs");
    let stats = report.tags_format_v1.expect("ran first time");
    assert_eq!(stats.already_json, 1);
    assert_eq!(stats.rewritten_csv, 0);
    assert_eq!(stats.rewritten_naked, 0);
    assert_eq!(
        read_tag(&database, "m1").as_deref(),
        Some(r#"["rust","bugfix"]"#)
    );
}

#[test]
fn mixed_formats_all_processed() {
    let database = Database::in_memory().expect("in-memory db");
    seed_raw_tag(&database, "json-row", Some(r#"["rust"]"#));
    seed_raw_tag(&database, "csv-row", Some("python,feature"));
    seed_raw_tag(&database, "naked-row", Some("docs"));
    let report = migrations::run_pending(&database).expect("migration runs");
    let stats = report.tags_format_v1.expect("ran first time");
    assert_eq!(stats.scanned, 3);
    assert_eq!(stats.already_json, 1);
    assert_eq!(stats.rewritten_csv, 1);
    assert_eq!(stats.rewritten_naked, 1);
    assert_eq!(
        read_tag(&database, "json-row").as_deref(),
        Some(r#"["rust"]"#)
    );
    assert_eq!(
        read_tag(&database, "csv-row").as_deref(),
        Some(r#"["python","feature"]"#)
    );
    assert_eq!(
        read_tag(&database, "naked-row").as_deref(),
        Some(r#"["docs"]"#)
    );
}

#[test]
fn idempotency_second_run_is_noop() {
    let database = Database::in_memory().expect("in-memory db");
    seed_raw_tag(&database, "m1", Some("rust,bugfix"));
    let first = migrations::run_pending(&database).expect("first migration");
    assert!(first.tags_format_v1.is_some(), "first run actually ran");

    let second = migrations::run_pending(&database).expect("second migration");
    assert!(
        second.tags_format_v1.is_none(),
        "second run must be no-op (returns None)"
    );
    assert_eq!(
        read_meta(&database, TAGS_FORMAT_KEY).as_deref(),
        Some(TAGS_FORMAT_TARGET_VALUE)
    );
}

#[test]
fn audit_strict_mode_returns_err_without_writes() {
    let database = Database::in_memory().expect("in-memory db");
    seed_raw_tag(&database, "comma-row", Some("rust,bugfix"));
    let result = tags_format_v1::run(&database, false, true);
    assert!(
        result.is_err(),
        "strict mode must reject comma-suspect rows"
    );
    assert_eq!(
        read_tag(&database, "comma-row").as_deref(),
        Some("rust,bugfix"),
        "no rows must be rewritten on strict abort"
    );
    assert!(
        read_meta(&database, TAGS_FORMAT_KEY).is_none(),
        "schema_meta must not be set on strict abort"
    );
}

#[test]
fn comma_in_tag_strict_mode_blocks() {
    let database = Database::in_memory().expect("in-memory db");
    seed_raw_tag(&database, "naked", Some("naked-tag"));
    seed_raw_tag(&database, "csv", Some("a,b"));
    let result = tags_format_v1::run(&database, false, true);
    let error = result.expect_err("strict must error");
    assert!(
        error.to_string().contains("comma"),
        "error message must mention comma audit: {error}"
    );
    assert_eq!(read_tag(&database, "naked").as_deref(), Some("naked-tag"));
    assert_eq!(read_tag(&database, "csv").as_deref(), Some("a,b"));
}

#[test]
fn dry_run_emits_diff_no_writes() {
    let database = Database::in_memory().expect("in-memory db");
    seed_raw_tag(&database, "m1", Some("rust,bugfix"));
    seed_raw_tag(&database, "m2", Some("naked"));
    let stats = tags_format_v1::run(&database, true, false)
        .expect("dry-run must succeed")
        .expect("returns stats on first run");
    assert!(stats.dry_run);
    assert_eq!(stats.rewritten_csv, 1);
    assert_eq!(stats.rewritten_naked, 1);
    assert_eq!(read_tag(&database, "m1").as_deref(), Some("rust,bugfix"));
    assert_eq!(read_tag(&database, "m2").as_deref(), Some("naked"));
    assert!(
        read_meta(&database, TAGS_FORMAT_KEY).is_none(),
        "dry-run must not write schema_meta"
    );
}

#[tokio::test]
async fn multi_tag_and_works_post_migration() {
    let state = build_deterministic_state();
    {
        let database = state.database.lock().unwrap();
        seed_raw_tag(&database, "rust-bug", Some("rust,bugfix"));
        seed_raw_tag(&database, "py-feat", Some("python,feature"));
        migrations::run_pending(&database).expect("migration");
    }

    let results = dispatch::route(
        "memory_search",
        &state,
        json!({
            "query": "ctx",
            "limit": 10,
            "tags": ["rust", "bugfix"],
        }),
    )
    .await
    .expect("search post-migration");
    let ids: Vec<String> = results
        .as_array()
        .expect("array")
        .iter()
        .map(|entry| entry["id"].as_str().expect("id").to_string())
        .collect();
    assert!(ids.contains(&"rust-bug".to_string()));
    assert!(!ids.contains(&"py-feat".to_string()));
}

#[tokio::test]
async fn empty_string_tags_treated_as_null() {
    let state = build_deterministic_state();
    let stored = dispatch::route(
        "memory_store",
        &state,
        json!({
            "memory_type": "decision",
            "context": "empty tag context",
            "action": "act",
            "result": "res",
            "tags": "",
        }),
    )
    .await
    .expect("store with empty tags");
    let id = stored["id"].as_str().expect("id");
    let database = state.database.lock().unwrap();
    let memory = database.get_memory(id).expect("get memory");
    assert!(
        memory.tags.is_none(),
        "empty-string tags must store as NULL, got {:?}",
        memory.tags
    );
}

#[tokio::test]
async fn wire_contract_array_path() {
    let state = build_deterministic_state();
    let stored = dispatch::route(
        "memory_store",
        &state,
        json!({
            "memory_type": "decision",
            "context": "array wire",
            "action": "act",
            "result": "res",
            "tags": ["rust", "bugfix"],
        }),
    )
    .await
    .expect("store with array tags");
    let id = stored["id"].as_str().expect("id");
    let database = state.database.lock().unwrap();
    let memory = database.get_memory(id).expect("get memory");
    assert_eq!(memory.tags.as_deref(), Some(r#"["rust","bugfix"]"#));
}

#[tokio::test]
async fn wire_contract_encoded_string_path() {
    let state = build_deterministic_state();
    let stored = dispatch::route(
        "memory_store",
        &state,
        json!({
            "memory_type": "decision",
            "context": "encoded wire",
            "action": "act",
            "result": "res",
            "tags": "rust,bugfix",
        }),
    )
    .await
    .expect("store with encoded csv tags");
    let id = stored["id"].as_str().expect("id");
    let database = state.database.lock().unwrap();
    let memory = database.get_memory(id).expect("get memory");
    assert_eq!(memory.tags.as_deref(), Some(r#"["rust","bugfix"]"#));
}

#[tokio::test]
async fn empty_array_input_stores_null() {
    let state = build_deterministic_state();
    let stored = dispatch::route(
        "memory_store",
        &state,
        json!({
            "memory_type": "decision",
            "context": "empty array wire",
            "action": "act",
            "result": "res",
            "tags": [],
        }),
    )
    .await
    .expect("store with empty array");
    let id = stored["id"].as_str().expect("id");
    let database = state.database.lock().unwrap();
    let memory = database.get_memory(id).expect("get memory");
    assert!(
        memory.tags.is_none(),
        "empty array must store as NULL, got {:?}",
        memory.tags
    );
}
