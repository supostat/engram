use std::sync::Arc;

use serde::Deserialize;
use serde_json::{Value, json};

use engram_storage::ConsolidationLogEntry;

use crate::error::CoreError;
use crate::lock_helpers;
use crate::server::ServerState;

const DEFAULT_LIMIT: usize = 50;

#[derive(Deserialize)]
struct ConsolidationLogParams {
    limit: Option<usize>,
}

pub async fn handle(state: &Arc<ServerState>, params: Value) -> Result<Value, CoreError> {
    let parsed: ConsolidationLogParams = serde_json::from_value(params)
        .map_err(|error| CoreError::DispatchError(error.to_string()))?;
    let limit = parsed.limit.unwrap_or(DEFAULT_LIMIT);

    let state_clone = Arc::clone(state);
    tokio::task::spawn_blocking(move || {
        let database = lock_helpers::lock_db(&state_clone);
        let entries = database.list_consolidation_log(limit)?;
        let count = entries.len();
        let serialized: Vec<Value> = entries.iter().map(entry_to_json).collect();
        Ok::<Value, CoreError>(json!({ "log": serialized, "count": count }))
    })
    .await
    .map_err(|error| CoreError::SocketError(error.to_string()))?
}

fn entry_to_json(entry: &ConsolidationLogEntry) -> Value {
    let memory_ids = serde_json::from_str::<Value>(&entry.memory_ids_json)
        .unwrap_or_else(|_| Value::String(entry.memory_ids_json.clone()));
    json!({
        "id": entry.id,
        "action": entry.action,
        "memory_ids": memory_ids,
        "reason": entry.reason,
        "performed_at": entry.performed_at,
        "performed_by": entry.performed_by,
    })
}
