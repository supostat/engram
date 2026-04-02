use std::sync::Arc;

use serde::Deserialize;
use serde_json::{Value, json};

use engram_storage::Memory;

use crate::error::CoreError;
use crate::server::ServerState;

#[derive(Deserialize)]
struct ImportParams {
    version: u64,
    memories: Vec<ImportedMemory>,
}

#[derive(Deserialize)]
struct ImportedMemory {
    id: String,
    memory_type: String,
    context: String,
    action: String,
    result: String,
    score: f32,
    tags: Option<String>,
    project: Option<String>,
    parent_id: Option<String>,
    source_ids: Option<String>,
    insight_type: Option<String>,
    created_at: String,
    updated_at: String,
    used_count: i64,
    last_used_at: Option<String>,
}

pub async fn handle(state: &Arc<ServerState>, params: Value) -> Result<Value, CoreError> {
    let parsed: ImportParams = serde_json::from_value(params)
        .map_err(|error| CoreError::DispatchError(error.to_string()))?;
    if parsed.version != 1 {
        return Err(CoreError::ImportVersionMismatch(parsed.version));
    }
    let state_clone = Arc::clone(state);
    tokio::task::spawn_blocking(move || {
        let database = state_clone.database.lock().unwrap();
        let (imported, skipped) = insert_non_duplicate_memories(&database, &parsed.memories)?;
        Ok::<Value, CoreError>(json!({ "imported": imported, "skipped": skipped }))
    })
    .await
    .map_err(|error| CoreError::SocketError(error.to_string()))?
}

fn insert_non_duplicate_memories(
    database: &engram_storage::Database,
    memories: &[ImportedMemory],
) -> Result<(usize, usize), CoreError> {
    let mut imported = 0;
    let mut skipped = 0;
    for entry in memories {
        if memory_exists(database, &entry.id)? {
            skipped += 1;
            continue;
        }
        let memory = imported_memory_to_storage(entry);
        database
            .insert_memory(&memory)
            .map_err(|error| CoreError::ImportFailed(error.to_string()))?;
        imported += 1;
    }
    Ok((imported, skipped))
}

fn memory_exists(database: &engram_storage::Database, id: &str) -> Result<bool, CoreError> {
    match database.get_memory(id) {
        Ok(_) => Ok(true),
        Err(engram_storage::StorageError::NotFound(_)) => Ok(false),
        Err(error) => Err(CoreError::ImportFailed(error.to_string())),
    }
}

fn imported_memory_to_storage(entry: &ImportedMemory) -> Memory {
    Memory {
        id: entry.id.clone(),
        memory_type: entry.memory_type.clone(),
        context: entry.context.clone(),
        action: entry.action.clone(),
        result: entry.result.clone(),
        score: entry.score,
        embedding_context: None,
        embedding_action: None,
        embedding_result: None,
        indexed: false,
        tags: entry.tags.clone(),
        project: entry.project.clone(),
        parent_id: entry.parent_id.clone(),
        source_ids: entry.source_ids.clone(),
        insight_type: entry.insight_type.clone(),
        created_at: entry.created_at.clone(),
        updated_at: entry.updated_at.clone(),
        used_count: entry.used_count,
        last_used_at: entry.last_used_at.clone(),
        superseded_by: None,
    }
}
