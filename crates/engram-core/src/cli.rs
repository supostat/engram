use std::sync::Arc;

use serde_json::Value;

use crate::dispatch;
use crate::error::CoreError;
use crate::output::{OutputFormat, format_output};
use crate::persistence;
use crate::server::{ServerState, initialize_state, resolve_index_directory};

pub async fn execute(
    state: Arc<ServerState>,
    method: &str,
    params: Value,
    format: &OutputFormat,
) -> Result<(), CoreError> {
    let result = dispatch::route(method, &state, params).await?;
    let formatted = format_output(&result, format);
    println!("{formatted}");
    save_indexes_if_mutating(method, &state)?;
    Ok(())
}

pub fn build_state(
    config: &crate::config::Config,
) -> Result<Arc<ServerState>, CoreError> {
    let state = initialize_state(config)?;
    Ok(Arc::new(state))
}

fn save_indexes_if_mutating(
    method: &str,
    state: &Arc<ServerState>,
) -> Result<(), CoreError> {
    let mutating = matches!(
        method,
        "memory_store" | "memory_consolidate_apply"
    );
    if !mutating {
        return Ok(());
    }
    let database_path = state.config.resolve_database_path();
    let index_directory = resolve_index_directory(&database_path);
    let indexes = state.indexes.lock().unwrap();
    persistence::save_to_disk(&index_directory, &indexes)
}
