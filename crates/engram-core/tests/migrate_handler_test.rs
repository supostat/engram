use std::sync::{Arc, Mutex};

use serde_json::json;
use tempfile::tempdir;

use engram_core::config::Config;
use engram_core::dispatch;
use engram_core::indexes::IndexSet;
use engram_core::migrate_handler;
use engram_core::server::ServerState;
use engram_embeddings::Embedder;
use engram_router::Router;
use engram_storage::{Database, Memory};

// Serializes tests that mutate process-global env vars (HOME) or cwd.
static ENV_LOCK: Mutex<()> = Mutex::new(());

fn build_deterministic_state() -> Arc<ServerState> {
    let database = Database::in_memory().expect("in-memory database");
    let mut config = Config::default();
    config.embedding.provider = "deterministic".into();
    let indexes = IndexSet::new(|| config.build_hnsw_params()).expect("index set");
    let embedder = Embedder::new();
    let router = Router::new(0.1, 0.15);
    Arc::new(ServerState {
        database: Mutex::new(database),
        indexes: Mutex::new(indexes),
        embedder: Mutex::new(embedder),
        router: Mutex::new(router),
        config,
        database_path: String::new(),
    })
}

fn sample_memory(id: &str, project: Option<&str>) -> Memory {
    Memory {
        id: id.to_string(),
        memory_type: "decision".into(),
        context: "context".into(),
        action: "action".into(),
        result: "result".into(),
        score: 0.5,
        embedding_context: None,
        embedding_action: None,
        embedding_result: None,
        indexed: false,
        tags: None,
        project: project.map(|value| value.to_string()),
        parent_id: None,
        source_ids: None,
        insight_type: None,
        created_at: "2025-01-01T00:00:00Z".into(),
        updated_at: "2025-01-01T00:00:00Z".into(),
        used_count: 0,
        last_used_at: None,
        superseded_by: None,
    }
}

fn seed_legacy_db_in_home(home_dir: &std::path::Path, memories: &[Memory]) -> std::path::PathBuf {
    let home_engram = home_dir.join(".engram");
    std::fs::create_dir_all(&home_engram).expect("create home .engram");
    let legacy_path = home_engram.join("engram.db");
    let legacy_path_str = legacy_path.to_str().expect("utf-8").to_string();
    {
        let database = Database::open(&legacy_path_str).expect("open legacy db");
        for memory in memories {
            database.insert_memory(memory).expect("seed legacy row");
        }
    }
    legacy_path
}

struct ScopedHome {
    original: Option<String>,
}

impl ScopedHome {
    fn set(home_dir: &std::path::Path) -> Self {
        let original = std::env::var("HOME").ok();
        // SAFETY: callers hold ENV_LOCK.
        unsafe {
            std::env::set_var("HOME", home_dir);
        }
        Self { original }
    }
}

impl Drop for ScopedHome {
    fn drop(&mut self) {
        // SAFETY: callers hold ENV_LOCK.
        unsafe {
            match self.original.take() {
                Some(value) => std::env::set_var("HOME", value),
                None => std::env::remove_var("HOME"),
            }
        }
    }
}

#[tokio::test]
#[allow(clippy::await_holding_lock)]
async fn dispatch_migrate_preview_counts_without_writing() {
    let _lock = ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    let home = tempdir().expect("home");
    seed_legacy_db_in_home(
        home.path(),
        &[
            sample_memory("preview-1", None),
            sample_memory("preview-2", Some("any-project")),
        ],
    );

    let state = build_deterministic_state();
    let _home_guard = ScopedHome::set(home.path());
    let data = dispatch::route("memory_migrate_preview", &state, json!({ "all": true }))
        .await
        .expect("preview should succeed");

    assert_eq!(data["read"], 2);
    assert_eq!(data["matched"], 2);
    assert_eq!(data["migrated"], 0);
    assert_eq!(data["dry_run"], true);

    let database = state.database.lock().expect("lock dest");
    assert!(database.get_memory("preview-1").is_err());
    assert!(database.get_memory("preview-2").is_err());
}

#[tokio::test]
#[allow(clippy::await_holding_lock)]
async fn dispatch_migrate_apply_persists_through_state() {
    let _lock = ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    let home = tempdir().expect("home");
    seed_legacy_db_in_home(
        home.path(),
        &[
            sample_memory("apply-1", Some("irrelevant")),
            sample_memory("apply-2", Some("irrelevant")),
        ],
    );

    let state = build_deterministic_state();
    let _home_guard = ScopedHome::set(home.path());
    let data = dispatch::route("memory_migrate_apply", &state, json!({ "all": true }))
        .await
        .expect("apply should succeed");

    assert_eq!(data["read"], 2);
    assert_eq!(data["matched"], 2);
    assert_eq!(data["migrated"], 2);
    assert_eq!(data["dry_run"], false);

    let database = state.database.lock().expect("lock dest");
    for id in ["apply-1", "apply-2"] {
        database.get_memory(id).expect("row persisted");
    }
}

#[test]
fn execute_reports_missing_source_with_6018() {
    let _lock = ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    let home = tempdir().expect("home");
    let project = tempdir().expect("project");
    std::fs::create_dir_all(project.path().join(".engram")).expect("create project .engram");

    let original_cwd = std::env::current_dir().expect("cwd");
    let original_home = std::env::var("HOME").ok();
    std::env::set_current_dir(project.path()).expect("chdir project");
    unsafe {
        std::env::set_var("HOME", home.path());
    }
    let result = migrate_handler::execute(false, true, &engram_core::output::OutputFormat::Json);
    std::env::set_current_dir(&original_cwd).expect("restore cwd");
    unsafe {
        match original_home {
            Some(value) => std::env::set_var("HOME", value),
            None => std::env::remove_var("HOME"),
        }
    }

    let error = result.expect_err("missing source must fail");
    assert!(
        error.to_string().contains("[6018]"),
        "expected 6018, got: {error}"
    );
}

#[test]
fn execute_fails_when_project_dir_missing() {
    let _lock = ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    let home = tempdir().expect("home");
    seed_legacy_db_in_home(home.path(), &[sample_memory("x", None)]);
    let project = tempdir().expect("project");
    // project dir has NO .engram/ directory — execute must refuse to proceed.

    let original_cwd = std::env::current_dir().expect("cwd");
    let original_home = std::env::var("HOME").ok();
    std::env::set_current_dir(project.path()).expect("chdir project");
    unsafe {
        std::env::set_var("HOME", home.path());
    }
    let result = migrate_handler::execute(false, true, &engram_core::output::OutputFormat::Json);
    std::env::set_current_dir(&original_cwd).expect("restore cwd");
    unsafe {
        match original_home {
            Some(value) => std::env::set_var("HOME", value),
            None => std::env::remove_var("HOME"),
        }
    }

    let error = result.expect_err("missing .engram/ must fail");
    let message = error.to_string();
    assert!(
        message.contains("[6012]") && message.contains(".engram/"),
        "expected 6012 init failure, got: {message}"
    );
}

#[test]
fn execute_migrates_rows_into_project_database() {
    let _lock = ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    let home = tempdir().expect("home");
    let project = tempdir().expect("project");
    let project_engram = project.path().join(".engram");
    std::fs::create_dir_all(&project_engram).expect("create project .engram");

    let matching_project = project
        .path()
        .file_name()
        .and_then(|name| name.to_str())
        .expect("project basename")
        .to_string();
    seed_legacy_db_in_home(
        home.path(),
        &[
            sample_memory("match", Some(&matching_project)),
            sample_memory("mismatch", Some("other-project")),
        ],
    );

    let original_cwd = std::env::current_dir().expect("cwd");
    let original_home = std::env::var("HOME").ok();
    std::env::set_current_dir(project.path()).expect("chdir project");
    unsafe {
        std::env::set_var("HOME", home.path());
    }
    let result = migrate_handler::execute(false, false, &engram_core::output::OutputFormat::Json);
    std::env::set_current_dir(&original_cwd).expect("restore cwd");
    unsafe {
        match original_home {
            Some(value) => std::env::set_var("HOME", value),
            None => std::env::remove_var("HOME"),
        }
    }

    assert!(result.is_ok(), "migrate should succeed: {result:?}");
    let dest_path = project_engram.join("engram.db");
    let destination = Database::open(dest_path.to_str().expect("utf-8")).expect("open dest");
    destination
        .get_memory("match")
        .expect("matching project row migrated");
    assert!(
        destination.get_memory("mismatch").is_err(),
        "non-matching project row must be skipped"
    );
}
