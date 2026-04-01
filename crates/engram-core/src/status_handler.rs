use std::sync::Arc;

use serde_json::{Value, json};

use crate::error::CoreError;
use crate::server::ServerState;

pub async fn handle(
    state: &Arc<ServerState>,
    _params: Value,
) -> Result<Value, CoreError> {
    let state_clone = Arc::clone(state);
    tokio::task::spawn_blocking(move || {
        let database = state_clone.database.lock().unwrap();
        let memory_count = count_memories(&database)?;
        let indexed_count = count_indexed_memories(&database)?;
        let pending_judgments = database.get_pending_judgments(usize::MAX)?.len();
        let index_size = state_clone.indexes.lock().unwrap().len();
        Ok::<Value, CoreError>(json!({
            "memory_count": memory_count,
            "indexed_count": indexed_count,
            "pending_judgments": pending_judgments,
            "index_size": index_size,
        }))
    })
    .await
    .map_err(|error| CoreError::SocketError(error.to_string()))?
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
