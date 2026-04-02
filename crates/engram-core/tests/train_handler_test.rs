use std::sync::{Arc, Mutex};

use serde_json::json;

use engram_core::config::Config;
use engram_core::dispatch;
use engram_core::indexes::IndexSet;
use engram_core::server::ServerState;
use engram_core::train_handler::{TrainerMessage, parse_trainer_output};
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

fn insert_test_memory(state: &Arc<ServerState>, id: &str, memory_type: &str) {
    let memory = engram_storage::Memory {
        id: id.to_string(),
        memory_type: memory_type.to_string(),
        context: "test context".to_string(),
        action: "test action".to_string(),
        result: "test result".to_string(),
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
        created_at: "2025-01-01T00:00:00Z".to_string(),
        updated_at: "2025-01-01T00:00:00Z".to_string(),
        used_count: 0,
        last_used_at: None,
        superseded_by: None,
    };
    let database = state.database.lock().unwrap();
    database.insert_memory(&memory).expect("insert memory");
}

#[tokio::test]
async fn train_list_empty() {
    let state = build_deterministic_state();
    let result = dispatch::route("memory_train_list", &state, json!({})).await;
    let data = result.expect("list should succeed");
    assert_eq!(data["count"], 0);
    assert!(data["insights"].as_array().expect("array").is_empty());
}

#[tokio::test]
async fn train_list_returns_insights_only() {
    let state = build_deterministic_state();
    insert_test_memory(&state, "decision-001", "decision");
    insert_test_memory(&state, "insight-001", "insight");
    insert_test_memory(&state, "pattern-001", "pattern");

    let result = dispatch::route("memory_train_list", &state, json!({})).await;
    let data = result.expect("list should succeed");
    assert_eq!(data["count"], 1);
    let insights = data["insights"].as_array().expect("array");
    assert_eq!(insights.len(), 1);
    assert_eq!(insights[0]["id"], "insight-001");
}

#[tokio::test]
async fn train_delete_nonexistent() {
    let state = build_deterministic_state();
    let result = dispatch::route(
        "memory_train_delete",
        &state,
        json!({"id": "nonexistent-id"}),
    )
    .await;
    let error = result.expect_err("delete nonexistent should fail");
    assert!(error.to_string().contains("not found"));
}

#[tokio::test]
async fn train_delete_non_insight() {
    let state = build_deterministic_state();
    insert_test_memory(&state, "decision-001", "decision");

    let result =
        dispatch::route("memory_train_delete", &state, json!({"id": "decision-001"})).await;
    let error = result.expect_err("delete non-insight should fail");
    assert!(error.to_string().contains("not an insight"));
}

#[tokio::test]
async fn train_delete_success() {
    let state = build_deterministic_state();
    insert_test_memory(&state, "insight-del-001", "insight");

    let result = dispatch::route(
        "memory_train_delete",
        &state,
        json!({"id": "insight-del-001"}),
    )
    .await;
    let data = result.expect("delete should succeed");
    assert_eq!(data["deleted"], "insight-del-001");

    let database = state.database.lock().unwrap();
    let get_result = database.get_memory("insight-del-001");
    assert!(get_result.is_err(), "memory should be gone after delete");
}

#[tokio::test]
async fn train_generate_missing_binary() {
    let state = build_deterministic_state();
    let result = dispatch::route("memory_train_generate", &state, json!({})).await;
    let error = result.expect_err("generate with missing binary should fail");
    assert!(error.to_string().contains("[6013] trainer failed:"));
}

#[test]
fn parse_trainer_message_insight() {
    let line = r#"{"type":"insight","id":"ins-001","context":"pattern found","action":"use caching","result":"improved latency","insight_type":"optimization","tags":"perf,cache","source_ids":"mem-1,mem-2"}"#;
    let message: TrainerMessage = serde_json::from_str(line).expect("parse insight");
    match message {
        TrainerMessage::Insight {
            id,
            context,
            action,
            result,
            insight_type,
            tags,
            source_ids,
        } => {
            assert_eq!(id, "ins-001");
            assert_eq!(context, "pattern found");
            assert_eq!(action, "use caching");
            assert_eq!(result, "improved latency");
            assert_eq!(insight_type, "optimization");
            assert_eq!(tags.as_deref(), Some("perf,cache"));
            assert_eq!(source_ids.as_deref(), Some("mem-1,mem-2"));
        }
        other => panic!("expected Insight, got: {other:?}"),
    }
}

#[test]
fn parse_trainer_message_complete() {
    let line = r#"{"type":"complete","insights_generated":5,"duration_secs":12.5}"#;
    let message: TrainerMessage = serde_json::from_str(line).expect("parse complete");
    match message {
        TrainerMessage::Complete {
            insights_generated,
            duration_secs,
        } => {
            assert_eq!(insights_generated, 5);
            assert!((duration_secs - 12.5).abs() < 0.01);
        }
        other => panic!("expected Complete, got: {other:?}"),
    }
}

#[test]
fn parse_trainer_message_malformed() {
    let line = r#"{"not_valid_json_for_protocol": true}"#;
    let result: Result<TrainerMessage, _> = serde_json::from_str(line);
    assert!(result.is_err(), "malformed message should fail to parse");
}

#[test]
fn parse_trainer_message_progress() {
    let line = r#"{"type":"progress","stage":"analyzing","percent":42.5}"#;
    let message: TrainerMessage = serde_json::from_str(line).expect("parse progress");
    match message {
        TrainerMessage::Progress { stage, percent } => {
            assert_eq!(stage, "analyzing");
            assert!((percent - 42.5).abs() < 0.01);
        }
        other => panic!("expected Progress, got: {other:?}"),
    }
}

#[test]
fn parse_trainer_message_recommendation() {
    let line = r#"{"type":"recommendation","target_id":"mem-1","action":"archive","reasoning":"low usage"}"#;
    let message: TrainerMessage = serde_json::from_str(line).expect("parse recommendation");
    match message {
        TrainerMessage::Recommendation {
            target_id,
            action,
            reasoning,
        } => {
            assert_eq!(target_id, "mem-1");
            assert_eq!(action, "archive");
            assert_eq!(reasoning, "low usage");
        }
        other => panic!("expected Recommendation, got: {other:?}"),
    }
}

#[test]
fn parse_trainer_message_metric() {
    let line = r#"{"type":"metric","name":"memory_coverage","value":0.85}"#;
    let message: TrainerMessage = serde_json::from_str(line).expect("parse metric");
    match message {
        TrainerMessage::Metric { name, value } => {
            assert_eq!(name, "memory_coverage");
            assert!((value - 0.85).abs() < 0.01);
        }
        other => panic!("expected Metric, got: {other:?}"),
    }
}

#[test]
fn parse_trainer_message_artifact() {
    let line = r#"{"type":"artifact","path":"/models/v1.bin","size_bytes":1024}"#;
    let message: TrainerMessage = serde_json::from_str(line).expect("parse artifact");
    match message {
        TrainerMessage::Artifact { path, size_bytes } => {
            assert_eq!(path, "/models/v1.bin");
            assert_eq!(size_bytes, 1024);
        }
        other => panic!("expected Artifact, got: {other:?}"),
    }
}

#[tokio::test]
async fn train_delete_missing_params() {
    let state = build_deterministic_state();
    let result = dispatch::route("memory_train_delete", &state, json!({})).await;
    let error = result.expect_err("delete with missing params should fail");
    assert!(error.to_string().contains("dispatch error"));
}

#[test]
fn parse_trainer_message_insight_minimal() {
    let line = r#"{"type":"insight","id":"ins-002","context":"ctx","action":"act","result":"res","insight_type":"general"}"#;
    let message: TrainerMessage = serde_json::from_str(line).expect("parse minimal insight");
    match message {
        TrainerMessage::Insight {
            id,
            tags,
            source_ids,
            ..
        } => {
            assert_eq!(id, "ins-002");
            assert!(tags.is_none());
            assert!(source_ids.is_none());
        }
        other => panic!("expected Insight, got: {other:?}"),
    }
}

#[test]
fn parse_trainer_output_empty() {
    let messages = parse_trainer_output("").expect("empty input should succeed");
    assert!(messages.is_empty());
}

#[test]
fn parse_trainer_output_malformed_line() {
    let input = concat!(
        r#"{"type":"progress","stage":"start","percent":0.0}"#,
        "\n",
        r#"{"broken json"#,
        "\n",
        r#"{"type":"complete","insights_generated":1,"duration_secs":1.0}"#,
    );
    let error = parse_trainer_output(input).expect_err("malformed line should fail");
    assert!(error.to_string().contains("[6015]"));
    assert!(error.to_string().contains("line 2"));
}

#[test]
fn parse_trainer_output_multiple_messages() {
    let input = concat!(
        r#"{"type":"progress","stage":"analyzing","percent":50.0}"#,
        "\n",
        r#"{"type":"insight","id":"ins-100","context":"c","action":"a","result":"r","insight_type":"pattern"}"#,
        "\n",
        r#"{"type":"complete","insights_generated":1,"duration_secs":2.5}"#,
    );
    let messages = parse_trainer_output(input).expect("valid multi-line should parse");
    assert_eq!(messages.len(), 3);
    assert!(matches!(&messages[0], TrainerMessage::Progress { .. }));
    assert!(matches!(&messages[1], TrainerMessage::Insight { .. }));
    assert!(matches!(&messages[2], TrainerMessage::Complete { .. }));
}
