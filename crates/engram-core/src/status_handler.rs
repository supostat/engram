use std::path::Path;
use std::sync::Arc;

use serde_json::{Value, json};

use crate::config::{Config, expand_tilde};
use crate::error::CoreError;
use crate::server::ServerState;

pub async fn handle(state: &Arc<ServerState>, _params: Value) -> Result<Value, CoreError> {
    let state_clone = Arc::clone(state);
    tokio::task::spawn_blocking(move || {
        let database = state_clone.database.lock().unwrap();
        let memory_count = count_memories(&database)?;
        let indexed_count = count_indexed_memories(&database)?;
        let pending_judgments = database.get_pending_judgments(usize::MAX)?.len();
        let index_size = state_clone.indexes.lock().unwrap().len();
        let hints = build_hints(&state_clone.config, memory_count, pending_judgments);
        Ok::<Value, CoreError>(json!({
            "memory_count": memory_count,
            "indexed_count": indexed_count,
            "pending_judgments": pending_judgments,
            "index_size": index_size,
            "hints": hints,
        }))
    })
    .await
    .map_err(|error| CoreError::SocketError(error.to_string()))?
}

fn build_hints(config: &Config, memory_count: usize, pending_judgments: usize) -> Vec<String> {
    let mut hints = Vec::new();
    let models_path = expand_tilde(&config.trainer.models_path);
    let has_onnx = directory_contains_onnx(&models_path);

    if memory_count >= 20 && !has_onnx {
        hints.push(format!(
            "You have {memory_count} memories. Install trainer for self-learning: \
             pip install engram-trainer && engram train"
        ));
    }

    if memory_count >= 20 && has_onnx && onnx_models_older_than_database(config, &models_path) {
        hints.push("Models may be outdated. Re-run: engram train".to_string());
    }

    if pending_judgments > 10 {
        hints.push(format!(
            "{pending_judgments} memories pending judgment. \
             Use memory_judge to improve search quality"
        ));
    }

    let llm_key_set = config
        .llm
        .api_key
        .as_deref()
        .is_some_and(|key| !key.is_empty());
    if !llm_key_set {
        hints.push(
            "Set ENGRAM_OPENAI_API_KEY for LLM-powered judge, HyDE, and consolidation".to_string(),
        );
    }

    hints
}

fn directory_contains_onnx(path: &str) -> bool {
    let Ok(entries) = std::fs::read_dir(path) else {
        return false;
    };
    entries
        .flatten()
        .any(|entry| entry.path().extension().is_some_and(|ext| ext == "onnx"))
}

fn onnx_models_older_than_database(config: &Config, models_path: &str) -> bool {
    let database_path = expand_tilde(&config.database.path);
    let database_modified = Path::new(&database_path)
        .metadata()
        .ok()
        .and_then(|meta| meta.modified().ok());

    let newest_onnx = std::fs::read_dir(models_path)
        .ok()
        .into_iter()
        .flatten()
        .flatten()
        .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "onnx"))
        .filter_map(|entry| entry.metadata().ok()?.modified().ok())
        .max();

    match (database_modified, newest_onnx) {
        (Some(db_time), Some(model_time)) => model_time < db_time,
        _ => false,
    }
}

fn count_memories(database: &engram_storage::Database) -> Result<usize, CoreError> {
    let count: i64 = database
        .connection()
        .query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))
        .map_err(|error| CoreError::Storage(engram_storage::StorageError::Sqlite(error)))?;
    Ok(count as usize)
}

fn count_indexed_memories(database: &engram_storage::Database) -> Result<usize, CoreError> {
    let count: i64 = database
        .connection()
        .query_row(
            "SELECT COUNT(*) FROM memories WHERE indexed = TRUE",
            [],
            |row| row.get(0),
        )
        .map_err(|error| CoreError::Storage(engram_storage::StorageError::Sqlite(error)))?;
    Ok(count as usize)
}
