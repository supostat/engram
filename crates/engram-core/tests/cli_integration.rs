use std::sync::{Arc, Mutex};

use serde_json::{Value, json};

use engram_core::cli;
use engram_core::config::Config;
use engram_core::dispatch;
use engram_core::indexes::IndexSet;
use engram_core::output::{OutputFormat, format_output};
use engram_core::server::ServerState;
use engram_embeddings::Embedder;
use engram_router::Router;
use engram_storage::Database;

fn build_deterministic_state() -> Arc<ServerState> {
    let database = Database::in_memory().expect("in-memory database");
    let mut config = Config::default();
    config.embedding.provider = "deterministic".into();
    let indexes = IndexSet::new(|| config.build_hnsw_params()).expect("index set");
    let embedder = Embedder::new();
    let router = Router::new(0.1, 0.15);
    Arc::new(ServerState {
        database: Mutex::new(database),
        indexes: Mutex::new(indexes),
        embedder: Mutex::new(embedder),
        router: Mutex::new(router),
        config,
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
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let database_path = temp_dir
        .path()
        .join("test.db")
        .to_str()
        .expect("path")
        .to_string();
    config.database.path = database_path;
    let state = cli::build_state(&config).expect("build state");
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
