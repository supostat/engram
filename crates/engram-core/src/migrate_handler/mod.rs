//! Migration orchestration: resolves legacy/project DB paths, opens databases, dispatches work.
//!
//! Pure migration logic (filtering, duplicate-skip, bulk insert) lives in [`logic`].

mod logic;

use std::path::Path;
use std::sync::Arc;

use serde::Deserialize;
use serde_json::Value;

use engram_storage::Database;

use crate::config::home_directory;
use crate::error::CoreError;
use crate::output::{OutputFormat, format_output};
use crate::server::ServerState;

pub use logic::{MigrationStats, perform_migration_impl};

const ENGRAM_DIRECTORY: &str = ".engram";
const DATABASE_FILENAME: &str = "engram.db";
const CONCURRENT_SERVER_WARNING: &str = "warning: if an engram server is running for this project, migration writes may contend with server writes (SQLite WAL serializes access)";

#[derive(Deserialize)]
struct MigrateParams {
    #[serde(default)]
    all: bool,
    #[serde(default)]
    dry_run: bool,
}

pub fn execute(all: bool, dry_run: bool, format: &OutputFormat) -> Result<(), CoreError> {
    eprintln!("{CONCURRENT_SERVER_WARNING}");
    let source_path = resolve_legacy_db_path()?;
    let dest_path = resolve_project_db_path()?;
    let project_hint = if all { None } else { extract_project_hint() };

    let source = Database::open_read_only(&source_path)?;
    let dest = Database::open(&dest_path)?;
    let stats = perform_migration_impl(&source, &dest, project_hint.as_deref(), all, dry_run)?;

    let output = format_output(
        &stats.to_json(dry_run, all, project_hint.as_deref()),
        format,
    );
    println!("{output}");
    Ok(())
}

pub async fn handle_preview(state: &Arc<ServerState>, params: Value) -> Result<Value, CoreError> {
    let parsed: MigrateParams = serde_json::from_value(params)
        .map_err(|error| CoreError::DispatchError(error.to_string()))?;
    run_dispatch(state, parsed.all, true).await
}

pub async fn handle_apply(state: &Arc<ServerState>, params: Value) -> Result<Value, CoreError> {
    let parsed: MigrateParams = serde_json::from_value(params)
        .map_err(|error| CoreError::DispatchError(error.to_string()))?;
    run_dispatch(state, parsed.all, parsed.dry_run).await
}

async fn run_dispatch(
    state: &Arc<ServerState>,
    all: bool,
    dry_run: bool,
) -> Result<Value, CoreError> {
    let source_path = resolve_legacy_db_path()?;
    let project_hint = if all { None } else { extract_project_hint() };
    let state_clone = Arc::clone(state);
    tokio::task::spawn_blocking(move || {
        let source = Database::open_read_only(&source_path)?;
        let dest = state_clone.database.lock().unwrap();
        let stats = perform_migration_impl(&source, &dest, project_hint.as_deref(), all, dry_run)?;
        Ok::<Value, CoreError>(stats.to_json(dry_run, all, project_hint.as_deref()))
    })
    .await
    .map_err(|error| CoreError::SocketError(error.to_string()))?
}

fn resolve_legacy_db_path() -> Result<String, CoreError> {
    let home = home_directory()
        .ok_or_else(|| CoreError::MigrationFailed("HOME environment variable not set".into()))?;
    let legacy_path = Path::new(&home)
        .join(ENGRAM_DIRECTORY)
        .join(DATABASE_FILENAME);
    if !legacy_path.exists() {
        return Err(CoreError::MigrationSourceNotFound);
    }
    legacy_path
        .to_str()
        .map(|value| value.to_string())
        .ok_or_else(|| CoreError::MigrationFailed("legacy path is not valid utf-8".into()))
}

fn resolve_project_db_path() -> Result<String, CoreError> {
    let cwd = std::env::current_dir()
        .map_err(|error| CoreError::MigrationFailed(format!("cwd unavailable: {error}")))?;
    let engram_dir = cwd.join(ENGRAM_DIRECTORY);
    if !engram_dir.is_dir() {
        return Err(CoreError::InitFailed(format!(
            "no .engram/ directory at {} — run 'engram init' first",
            cwd.display()
        )));
    }
    let dest_path = engram_dir.join(DATABASE_FILENAME);
    dest_path
        .to_str()
        .map(|value| value.to_string())
        .ok_or_else(|| CoreError::MigrationFailed("destination path is not valid utf-8".into()))
}

fn extract_project_hint() -> Option<String> {
    let cwd = std::env::current_dir().ok()?;
    cwd_basename(&cwd)
}

fn cwd_basename(cwd: &Path) -> Option<String> {
    cwd.file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn cwd_basename_extracts_last_path_segment() {
        let path = PathBuf::from("/tmp/some/project-root");
        assert_eq!(cwd_basename(&path).as_deref(), Some("project-root"));
    }
}
