use std::sync::Arc;

use serde_json::{Value, json};

use engram_storage::Memory;

use crate::error::CoreError;
use crate::server::ServerState;
use crate::timestamp::current_utc_timestamp;

pub async fn handle(state: &Arc<ServerState>, _params: Value) -> Result<Value, CoreError> {
    let state_clone = Arc::clone(state);
    tokio::task::spawn_blocking(move || {
        let database = state_clone.database.lock().unwrap();
        let memories = query_active_memories(&database)?;
        let exported_at = current_utc_timestamp();
        let count = memories.len();
        let serialized: Vec<Value> = memories.iter().map(memory_to_portable_json).collect();
        Ok::<Value, CoreError>(json!({
            "version": 1,
            "memories": serialized,
            "exported_at": exported_at,
            "count": count,
        }))
    })
    .await
    .map_err(|error| CoreError::SocketError(error.to_string()))?
}

fn query_active_memories(database: &engram_storage::Database) -> Result<Vec<Memory>, CoreError> {
    let mut statement = database
        .connection()
        .prepare("SELECT * FROM memories WHERE superseded_by IS NULL")
        .map_err(|error| CoreError::ExportFailed(error.to_string()))?;
    let rows = statement
        .query_map([], engram_storage::row_to_memory)
        .map_err(|error| CoreError::ExportFailed(error.to_string()))?;
    let mut memories = Vec::new();
    for row in rows {
        memories.push(row.map_err(|error| CoreError::ExportFailed(error.to_string()))?);
    }
    Ok(memories)
}

pub fn memory_to_portable_json(memory: &Memory) -> Value {
    json!({
        "id": memory.id,
        "memory_type": memory.memory_type,
        "context": memory.context,
        "action": memory.action,
        "result": memory.result,
        "score": memory.score,
        "tags": memory.tags,
        "project": memory.project,
        "parent_id": memory.parent_id,
        "source_ids": memory.source_ids,
        "insight_type": memory.insight_type,
        "created_at": memory.created_at,
        "updated_at": memory.updated_at,
        "used_count": memory.used_count,
        "last_used_at": memory.last_used_at,
    })
}
