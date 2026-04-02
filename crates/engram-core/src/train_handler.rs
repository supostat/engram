use std::sync::Arc;

use serde::Deserialize;
use serde_json::{Value, json};

use engram_storage::Memory;

use crate::config::expand_tilde;
use crate::error::CoreError;
use crate::export_handler::memory_to_portable_json;
use crate::server::ServerState;
use crate::timestamp::current_utc_timestamp;

const INSIGHT_MEMORY_TYPE: &str = "insight";

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TrainerMessage {
    Progress {
        stage: String,
        percent: f64,
    },
    Insight {
        id: String,
        context: String,
        action: String,
        result: String,
        insight_type: String,
        tags: Option<String>,
        source_ids: Option<String>,
    },
    Recommendation {
        target_id: String,
        action: String,
        reasoning: String,
    },
    Metric {
        name: String,
        value: f64,
    },
    Artifact {
        path: String,
        size_bytes: u64,
    },
    Complete {
        insights_generated: u64,
        duration_secs: f64,
    },
}

#[derive(Deserialize)]
struct DeleteParams {
    id: String,
}

pub async fn handle_generate(
    state: &Arc<ServerState>,
    _params: Value,
) -> Result<Value, CoreError> {
    let trainer_binary = resolve_trainer_binary(&state.config.trainer.trainer_binary);
    validate_trainer_exists(&trainer_binary)?;
    let timeout_secs = state.config.trainer.trainer_timeout_secs;
    let database_path = state.config.resolve_database_path();
    let models_path = expand_tilde(&state.config.trainer.models_path);
    let output = spawn_trainer(&trainer_binary, &database_path, &models_path, timeout_secs).await?;
    let insights = parse_trainer_output(&output)?;
    let inserted_count = insert_generated_insights(state, &insights).await?;
    Ok(json!({
        "generated": inserted_count,
        "messages_parsed": insights.len(),
    }))
}

pub async fn handle_list(
    state: &Arc<ServerState>,
    _params: Value,
) -> Result<Value, CoreError> {
    let state_clone = Arc::clone(state);
    tokio::task::spawn_blocking(move || {
        let database = state_clone.database.lock().unwrap();
        let insights = query_active_insights(&database)?;
        let count = insights.len();
        let serialized: Vec<Value> = insights.iter().map(memory_to_portable_json).collect();
        Ok::<Value, CoreError>(json!({ "insights": serialized, "count": count }))
    })
    .await
    .map_err(|error| CoreError::SocketError(error.to_string()))?
}

pub async fn handle_delete(
    state: &Arc<ServerState>,
    params: Value,
) -> Result<Value, CoreError> {
    let parsed: DeleteParams = serde_json::from_value(params)
        .map_err(|error| CoreError::DispatchError(error.to_string()))?;
    let insight_id = parsed.id;
    let state_clone = Arc::clone(state);
    tokio::task::spawn_blocking(move || {
        let database = state_clone.database.lock().unwrap();
        let memory = database.get_memory(&insight_id)?;
        if memory.memory_type != INSIGHT_MEMORY_TYPE {
            return Err(CoreError::DispatchError(format!(
                "memory {insight_id} is not an insight"
            )));
        }
        database.delete_memory(&insight_id)?;
        Ok::<Value, CoreError>(json!({ "deleted": insight_id }))
    })
    .await
    .map_err(|error| CoreError::SocketError(error.to_string()))?
}

fn resolve_trainer_binary(configured_path: &str) -> String {
    expand_tilde(configured_path)
}

fn validate_trainer_exists(binary_path: &str) -> Result<(), CoreError> {
    if which_binary(binary_path).is_none() {
        return Err(CoreError::TrainerFailed(binary_path.to_string()));
    }
    Ok(())
}

fn which_binary(binary_path: &str) -> Option<std::path::PathBuf> {
    let path = std::path::Path::new(binary_path);
    if path.is_absolute() {
        return if path.exists() { Some(path.to_path_buf()) } else { None };
    }
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths)
            .map(|directory| directory.join(binary_path))
            .find(|candidate| candidate.exists())
    })
}

async fn spawn_trainer(
    binary_path: &str,
    database_path: &str,
    models_path: &str,
    timeout_secs: u64,
) -> Result<String, CoreError> {
    let child = tokio::process::Command::new(binary_path)
        .arg("--database")
        .arg(database_path)
        .arg("--models-path")
        .arg(models_path)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|error| CoreError::TrainerFailed(format!(
            "{binary_path}: {error}"
        )))?;

    let timeout_duration = std::time::Duration::from_secs(timeout_secs);
    let wait_result = tokio::time::timeout(
        timeout_duration,
        child.wait_with_output(),
    )
    .await;

    match wait_result {
        Ok(Ok(output)) => {
            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(CoreError::TrainerFailed(format!(
                    "{binary_path} exited with {}: {stderr}",
                    output.status
                )));
            }
            String::from_utf8(output.stdout).map_err(|error| {
                CoreError::TrainerMalformedOutput(format!(
                    "non-utf8 stdout: {error}"
                ))
            })
        }
        Ok(Err(error)) => Err(CoreError::TrainerFailed(format!(
            "{binary_path}: {error}"
        ))),
        Err(_) => Err(CoreError::TrainerTimeout),
    }
}

pub fn parse_trainer_output(output: &str) -> Result<Vec<TrainerMessage>, CoreError> {
    let mut messages = Vec::new();
    for (line_number, line) in output.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let message: TrainerMessage = serde_json::from_str(trimmed).map_err(|error| {
            CoreError::TrainerMalformedOutput(format!(
                "line {}: {error}",
                line_number + 1
            ))
        })?;
        messages.push(message);
    }
    Ok(messages)
}

async fn insert_generated_insights(
    state: &Arc<ServerState>,
    messages: &[TrainerMessage],
) -> Result<u64, CoreError> {
    let insights: Vec<Memory> = messages
        .iter()
        .filter_map(build_insight_memory)
        .collect();
    let count = insights.len() as u64;
    if insights.is_empty() {
        return Ok(0);
    }
    let state_clone = Arc::clone(state);
    tokio::task::spawn_blocking(move || {
        let database = state_clone.database.lock().unwrap();
        database.bulk_insert_memories(&insights)?;
        Ok::<u64, CoreError>(count)
    })
    .await
    .map_err(|error| CoreError::SocketError(error.to_string()))?
}

fn build_insight_memory(message: &TrainerMessage) -> Option<Memory> {
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
            let timestamp = current_utc_timestamp();
            Some(Memory {
                id: id.clone(),
                memory_type: INSIGHT_MEMORY_TYPE.to_string(),
                context: context.clone(),
                action: action.clone(),
                result: result.clone(),
                score: 0.0,
                embedding_context: None,
                embedding_action: None,
                embedding_result: None,
                indexed: false,
                tags: tags.clone(),
                project: None,
                parent_id: None,
                source_ids: source_ids.clone(),
                insight_type: Some(insight_type.clone()),
                created_at: timestamp.clone(),
                updated_at: timestamp,
                used_count: 0,
                last_used_at: None,
                superseded_by: None,
            })
        }
        _ => None,
    }
}

fn query_active_insights(
    database: &engram_storage::Database,
) -> Result<Vec<Memory>, CoreError> {
    let mut statement = database
        .connection()
        .prepare(
            "SELECT * FROM memories WHERE memory_type = ?1 \
             AND superseded_by IS NULL",
        )
        .map_err(|error| CoreError::Storage(engram_storage::StorageError::Sqlite(error)))?;
    let rows = statement
        .query_map([INSIGHT_MEMORY_TYPE], engram_storage::row_to_memory)
        .map_err(|error| CoreError::Storage(engram_storage::StorageError::Sqlite(error)))?;
    let mut insights = Vec::new();
    for row in rows {
        insights.push(
            row.map_err(|error| {
                CoreError::Storage(engram_storage::StorageError::Sqlite(error))
            })?,
        );
    }
    Ok(insights)
}
