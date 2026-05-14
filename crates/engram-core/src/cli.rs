use std::path::PathBuf;
use std::sync::Arc;

use serde_json::Value;

use crate::config;
use crate::dispatch;
use crate::error::CoreError;
use crate::output::{OutputFormat, format_output};
use crate::persistence;
use crate::server::{
    ServerState, check_legacy_database, initialize_state, resolve_index_directory,
};

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

pub fn build_state(config: &crate::config::Config) -> Result<Arc<ServerState>, CoreError> {
    let cwd = std::env::current_dir()
        .map_err(|error| CoreError::InitFailed(format!("cwd unavailable: {error}")))?;
    let home = home_dir_or_error()?;
    build_state_with_dirs(config, &cwd, &home)
}

pub fn build_state_with_dirs(
    config: &crate::config::Config,
    cwd: &std::path::Path,
    home_dir: &std::path::Path,
) -> Result<Arc<ServerState>, CoreError> {
    let project_dir = match config::resolve_project_dir(cwd, None) {
        Ok(path) => path,
        Err(CoreError::ProjectDirNotFound) => {
            return Err(CoreError::InitFailed(
                "no .engram/ directory found in cwd or ancestors — run 'engram init' in your project root".into(),
            ));
        }
        Err(other) => return Err(other),
    };
    check_legacy_database(&project_dir, home_dir)?;
    let state = initialize_state(config, &project_dir, home_dir)?;
    Ok(Arc::new(state))
}

fn home_dir_or_error() -> Result<PathBuf, CoreError> {
    match std::env::var("HOME") {
        Ok(value) if !value.is_empty() => Ok(PathBuf::from(value)),
        _ => Err(CoreError::InitFailed(
            "HOME environment variable not set".into(),
        )),
    }
}

fn save_indexes_if_mutating(method: &str, state: &Arc<ServerState>) -> Result<(), CoreError> {
    let mutating = matches!(
        method,
        "memory_store"
            | "memory_consolidate_apply"
            | "memory_import"
            | "memory_insights"
            | "memory_migrate_apply"
            | "memory_train_generate"
            | "memory_train_delete"
            | "memory_reembed"
    );
    if !mutating {
        return Ok(());
    }
    let index_directory = resolve_index_directory(&state.database_path);
    let indexes = state.indexes.read().unwrap();
    persistence::save_to_disk(&index_directory, &indexes)
}
