use std::sync::Arc;

use serde::Deserialize;
use serde_json::{Value, json};

use engram_storage::Memory;

use crate::error::CoreError;
use crate::export_handler::memory_to_portable_json;
use crate::server::ServerState;

#[derive(Deserialize)]
struct InsightsParams {
    action: String,
    id: Option<String>,
}

pub async fn handle(
    state: &Arc<ServerState>,
    params: Value,
) -> Result<Value, CoreError> {
    let parsed: InsightsParams = serde_json::from_value(params)
        .map_err(|error| CoreError::DispatchError(error.to_string()))?;
    match parsed.action.as_str() {
        "list" => handle_list(state).await,
        "generate" => Err(CoreError::DispatchError(
            "insight generation requires engram-trainer (Phase 5)".into(),
        )),
        "delete" => handle_delete(state, parsed.id).await,
        other => Err(CoreError::DispatchError(format!(
            "invalid insights action: {other}"
        ))),
    }
}

async fn handle_list(state: &Arc<ServerState>) -> Result<Value, CoreError> {
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

async fn handle_delete(
    state: &Arc<ServerState>,
    id: Option<String>,
) -> Result<Value, CoreError> {
    let insight_id = id.ok_or_else(|| {
        CoreError::DispatchError("delete requires 'id' parameter".into())
    })?;
    let state_clone = Arc::clone(state);
    tokio::task::spawn_blocking(move || {
        let database = state_clone.database.lock().unwrap();
        let memory = database.get_memory(&insight_id)?;
        if memory.memory_type != "insight" {
            return Err(CoreError::DispatchError(format!(
                "memory {} is not an insight", insight_id
            )));
        }
        database.delete_memory(&insight_id)?;
        Ok::<Value, CoreError>(json!({ "deleted": insight_id }))
    })
    .await
    .map_err(|error| CoreError::SocketError(error.to_string()))?
}

fn query_active_insights(
    database: &engram_storage::Database,
) -> Result<Vec<Memory>, CoreError> {
    let mut statement = database
        .connection()
        .prepare(
            "SELECT * FROM memories WHERE memory_type = 'insight' \
             AND superseded_by IS NULL",
        )
        .map_err(|error| CoreError::Storage(engram_storage::StorageError::Sqlite(error)))?;
    let rows = statement
        .query_map([], engram_storage::row_to_memory)
        .map_err(|error| CoreError::Storage(engram_storage::StorageError::Sqlite(error)))?;
    let mut insights = Vec::new();
    for row in rows {
        insights.push(
            row.map_err(|error| CoreError::Storage(engram_storage::StorageError::Sqlite(error)))?,
        );
    }
    Ok(insights)
}

