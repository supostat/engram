use std::sync::Arc;

use serde::Deserialize;
use serde_json::{Value, json};

use crate::error::CoreError;
use crate::lock_helpers;
use crate::persistence::hash_string_to_u64;
use crate::server::ServerState;

#[derive(Deserialize)]
struct DeleteParams {
    id: String,
}

pub async fn handle(state: &Arc<ServerState>, params: Value) -> Result<Value, CoreError> {
    let parsed: DeleteParams = serde_json::from_value(params)
        .map_err(|error| CoreError::DispatchError(error.to_string()))?;
    let id = parsed.id;
    let state_clone = Arc::clone(state);
    tokio::task::spawn_blocking(move || {
        {
            let database = lock_helpers::lock_db(&state_clone);
            database.delete_memory(&id)?;
        }
        let hashed = hash_string_to_u64(&id);
        let mut indexes = lock_helpers::write_indexes(&state_clone);
        if indexes.contains(hashed) {
            indexes.delete(hashed).map_err(CoreError::Hnsw)?;
        }
        Ok::<Value, CoreError>(json!({ "deleted": id }))
    })
    .await
    .map_err(|error| CoreError::SocketError(error.to_string()))?
}
