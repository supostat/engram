use std::sync::{Arc, Mutex, RwLock};

use serde_json::{Value, json};

use engram_core::cli;
use engram_core::config::Config;
use engram_core::dispatch;
use engram_core::error::CoreError;
use engram_core::indexes::IndexSet;
use engram_core::output::{OutputFormat, format_output};
use engram_core::server::ServerState;
use engram_embeddings::Embedder;
use engram_llm_client::{EmbeddingProvider, TextGenerator};
use engram_router::Router;
use engram_storage::Database;

// Serializes tests that mutate process-global state (cwd, HOME).
static ENV_LOCK: Mutex<()> = Mutex::new(());

fn build_deterministic_state() -> Arc<ServerState> {
    let database = Database::in_memory().expect("in-memory database");
    let mut config = Config::default();
    config.embedding.provider = "deterministic".into();
    let indexes = IndexSet::new(|| config.build_hnsw_params()).expect("index set");
    let embedder = Embedder::new();
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
fn output_json_format_is_pretty_printed() {
    let value = json!({"key": "value", "count": 42});
    let formatted = format_output(&value, &OutputFormat::Json);
    assert!(formatted.contains('\n'));
    assert!(formatted.contains("\"key\""));
    assert!(formatted.contains("\"value\""));
    let reparsed: Value = serde_json::from_str(&formatted).expect("valid json");
    assert_eq!(reparsed["key"], "value");
    assert_eq!(reparsed["count"], 42);
}

#[test]
fn output_text_format_renders_key_value_lines() {
    let value = json!({"memory_count": 5, "indexed_count": 3});
    let formatted = format_output(&value, &OutputFormat::Text);
    assert!(formatted.contains("memory_count: 5"));
    assert!(formatted.contains("indexed_count: 3"));
}

#[test]
fn output_text_format_renders_array_with_separators() {
    let value = json!([
        {"id": "a", "score": 0.9},
        {"id": "b", "score": 0.5}
    ]);
    let formatted = format_output(&value, &OutputFormat::Text);
    assert!(formatted.contains("id: a"));
    assert!(formatted.contains("id: b"));
    assert!(formatted.contains("---"));
}

#[test]
fn output_jsonl_format_renders_one_line_per_array_item() {
    let value = json!([{"id": "a"}, {"id": "b"}]);
    let formatted = format_output(&value, &OutputFormat::Jsonl);
    let lines: Vec<&str> = formatted.lines().collect();
    assert_eq!(lines.len(), 2);
    let first: Value = serde_json::from_str(lines[0]).expect("valid jsonl line");
    assert_eq!(first["id"], "a");
    let second: Value = serde_json::from_str(lines[1]).expect("valid jsonl line");
    assert_eq!(second["id"], "b");
}

#[test]
fn output_jsonl_format_renders_object_as_single_line() {
    let value = json!({"status": "ok"});
    let formatted = format_output(&value, &OutputFormat::Jsonl);
    assert_eq!(formatted.lines().count(), 1);
    let reparsed: Value = serde_json::from_str(&formatted).expect("valid jsonl");
    assert_eq!(reparsed["status"], "ok");
}

#[tokio::test]
async fn cli_store_and_search_full_cycle() {
    let state = build_deterministic_state();
    let store_params = json!({
        "memory_type": "decision",
        "context": "configured database connection pooling for high throughput",
        "action": "set maximum connections to fifty per instance",
        "result": "database connections are stable under load testing",
    });
    let store_result = dispatch::route("memory_store", &state, store_params).await;
    let stored = store_result.expect("store should succeed");
    let stored_id = stored["id"].as_str().expect("stored id");
    assert!(!stored_id.is_empty());

    let search_params = json!({
        "query": "database connection pooling configured high throughput",
        "limit": 10,
    });
    let search_result = dispatch::route("memory_search", &state, search_params).await;
    let results = search_result.expect("search should succeed");
    let results_array = results.as_array().expect("results array");
    let found = results_array.iter().any(|entry| entry["id"] == stored_id);
    assert!(found, "stored memory must appear in search results");
}

#[tokio::test]
async fn cli_status_shows_counts() {
    let state = build_deterministic_state();
    let result = dispatch::route("memory_status", &state, json!({})).await;
    let data = result.expect("status should succeed");
    assert_eq!(data["memory_count"], 0);
    assert_eq!(data["indexed_count"], 0);
    assert_eq!(data["index_size"], 0);

    let store_params = json!({
        "memory_type": "decision",
        "context": "test context for status verification",
        "action": "test action for status verification",
        "result": "test result for status verification",
    });
    dispatch::route("memory_store", &state, store_params)
        .await
        .expect("store should succeed");

    let result = dispatch::route("memory_status", &state, json!({})).await;
    let data = result.expect("status should succeed after insert");
    assert_eq!(data["memory_count"], 1);
    assert_eq!(data["indexed_count"], 1);
    assert_eq!(data["index_size"], 1);
}

#[test]
fn cli_version_output_contains_version() {
    let version = env!("CARGO_PKG_VERSION");
    let value = json!({ "version": version });
    let json_output = format_output(&value, &OutputFormat::Json);
    assert!(json_output.contains(version));
    let text_output = format_output(&value, &OutputFormat::Text);
    assert!(text_output.contains(version));
}

#[tokio::test]
async fn cli_build_state_creates_valid_state() {
    let mut config = Config::default();
    config.embedding.provider = "deterministic".into();
    let project = tempfile::tempdir().expect("project");
    let home = tempfile::tempdir().expect("home");
    std::fs::create_dir_all(project.path().join(".engram")).expect("create .engram");
    let state =
        cli::build_state_with_dirs(&config, project.path(), home.path()).expect("build state");
    let result = dispatch::route("memory_status", &state, json!({})).await;
    let data = result.expect("status should succeed");
    assert_eq!(data["memory_count"], 0);
}

#[test]
fn output_text_format_handles_scalar_string() {
    let value = json!("hello world");
    let formatted = format_output(&value, &OutputFormat::Text);
    assert_eq!(formatted, "hello world");
}

#[test]
fn output_text_format_handles_null() {
    let value = json!(null);
    let formatted = format_output(&value, &OutputFormat::Text);
    assert!(formatted.is_empty());
}

fn seed_project_database(project_dir: &std::path::Path) {
    let engram_dir = project_dir.join(".engram");
    std::fs::create_dir_all(&engram_dir).expect("create .engram");
    let database_path = engram_dir.join("engram.db");
    Database::open(database_path.to_str().expect("valid utf-8")).expect("open project database");
}

#[test]
fn cli_build_state_walks_up_from_nested_dir() {
    let _lock = ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    let project = tempfile::tempdir().expect("project");
    seed_project_database(project.path());
    let nested = project.path().join("sub").join("deep");
    std::fs::create_dir_all(&nested).expect("create nested dirs");
    let home = tempfile::tempdir().expect("home");

    let mut config = Config::default();
    config.embedding.provider = "deterministic".into();

    let original_cwd = std::env::current_dir().expect("cwd");
    let original_home = std::env::var("HOME").ok();
    // SAFETY: serialized via ENV_LOCK; restored before returning.
    std::env::set_current_dir(&nested).expect("chdir into nested");
    unsafe {
        std::env::set_var("HOME", home.path());
    }
    let result = cli::build_state(&config);
    std::env::set_current_dir(&original_cwd).expect("restore cwd");
    unsafe {
        match &original_home {
            Some(value) => std::env::set_var("HOME", value),
            None => std::env::remove_var("HOME"),
        }
    }

    let state = result.expect("walk-up from nested should resolve project");
    let actual_db = std::path::Path::new(&state.database_path)
        .canonicalize()
        .expect("canonicalize resolved db path");
    let expected_db = project
        .path()
        .join(".engram")
        .join("engram.db")
        .canonicalize()
        .expect("canonicalize expected db path");
    assert_eq!(actual_db, expected_db);
}

#[test]
fn cli_build_state_legacy_database_error() {
    let _lock = ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    // Project has .engram/ but NO engram.db; home has legacy engram.db → triggers [6017].
    let project = tempfile::tempdir().expect("project");
    std::fs::create_dir_all(project.path().join(".engram")).expect("create project .engram");

    let home = tempfile::tempdir().expect("home");
    let home_engram = home.path().join(".engram");
    std::fs::create_dir_all(&home_engram).expect("create home .engram");
    let legacy_db = home_engram.join("engram.db");
    Database::open(legacy_db.to_str().expect("valid utf-8")).expect("create legacy db");

    let mut config = Config::default();
    config.embedding.provider = "deterministic".into();

    let original_cwd = std::env::current_dir().expect("cwd");
    let original_home = std::env::var("HOME").ok();
    std::env::set_current_dir(project.path()).expect("chdir to project");
    unsafe {
        std::env::set_var("HOME", home.path());
    }
    let result = cli::build_state(&config);
    std::env::set_current_dir(&original_cwd).expect("restore cwd");
    unsafe {
        match &original_home {
            Some(value) => std::env::set_var("HOME", value),
            None => std::env::remove_var("HOME"),
        }
    }

    let error = match result {
        Ok(_) => panic!("legacy db must be rejected"),
        Err(err) => err,
    };
    assert!(
        matches!(error, CoreError::LegacyDatabaseDetected { .. }),
        "expected LegacyDatabaseDetected, got: {error}"
    );
    assert!(error.to_string().contains("[6017]"));
}
