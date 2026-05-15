use std::path::{Path, PathBuf};
use std::sync::Mutex;

use tempfile::tempdir;

use engram_core::config;
use engram_core::error::CoreError;

// Serializes every test in this file. `resolve_project_dir` reads the
// process-global `ENGRAM_PROJECT_DIR` env var, so any test that does
// NOT explicitly override that env can be poisoned by the one test
// that sets it (`resolve_project_dir_respects_env_override`) running
// concurrently. Holding the lock for the whole test body keeps the
// env state consistent for each assertion.
static ENV_LOCK: Mutex<()> = Mutex::new(());

fn lock_env() -> std::sync::MutexGuard<'static, ()> {
    ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[test]
fn resolve_project_dir_finds_engram_in_cwd() {
    let _lock = lock_env();
    let temp = tempdir().expect("temp dir");
    let engram_dir = temp.path().join(".engram");
    std::fs::create_dir_all(&engram_dir).expect("create .engram");
    let resolved = config::resolve_project_dir(temp.path(), None).expect("should resolve");
    assert_eq!(resolved, temp.path());
}

#[test]
fn resolve_project_dir_walks_up() {
    let _lock = lock_env();
    let temp = tempdir().expect("temp dir");
    let engram_dir = temp.path().join(".engram");
    std::fs::create_dir_all(&engram_dir).expect("create .engram");
    let nested = temp.path().join("a").join("b").join("c");
    std::fs::create_dir_all(&nested).expect("create nested");
    let resolved = config::resolve_project_dir(&nested, None).expect("should walk up");
    assert_eq!(resolved, temp.path());
}

#[test]
fn resolve_project_dir_respects_explicit_override() {
    let _lock = lock_env();
    let temp = tempdir().expect("temp dir");
    let other = tempdir().expect("other");
    let engram_dir = other.path().join(".engram");
    std::fs::create_dir_all(&engram_dir).expect("create .engram");
    let resolved =
        config::resolve_project_dir(temp.path(), Some(other.path())).expect("override wins");
    assert_eq!(resolved, other.path());
}

#[test]
fn resolve_project_dir_not_found() {
    let _lock = lock_env();
    let temp = tempdir().expect("temp dir");
    let nested = temp.path().join("sub");
    std::fs::create_dir_all(&nested).expect("create nested");
    let result = config::resolve_project_dir(&nested, None);
    let error = result.expect_err("should fail without .engram");
    assert!(matches!(error, CoreError::ProjectDirNotFound));
    assert!(error.to_string().contains("[6016]"));
}

#[test]
fn resolve_project_dir_respects_env_override() {
    let _lock = lock_env();

    // Project that the env var points to: create a real .engram/ so it is a valid project.
    let env_project = tempdir().expect("env project");
    std::fs::create_dir_all(env_project.path().join(".engram")).expect("create .engram");

    // Unrelated start directory with no marker — walk-up would fail from here.
    let other_start = tempdir().expect("other start");

    let original = std::env::var("ENGRAM_PROJECT_DIR").ok();
    // SAFETY: serialized via ENV_LOCK above.
    unsafe {
        std::env::set_var("ENGRAM_PROJECT_DIR", env_project.path());
    }
    let result = config::resolve_project_dir(other_start.path(), None);
    unsafe {
        match &original {
            Some(value) => std::env::set_var("ENGRAM_PROJECT_DIR", value),
            None => std::env::remove_var("ENGRAM_PROJECT_DIR"),
        }
    }

    let resolved = result.expect("env override should win");
    assert_eq!(resolved, env_project.path());
}

#[test]
fn resolve_project_dir_at_filesystem_root() {
    let _lock = lock_env();
    // PathBuf::pop() returns false at filesystem root, loop terminates with ProjectDirNotFound.
    // No panic even though "/" exists and likely lacks a .engram/ marker.
    let result = config::resolve_project_dir(Path::new("/"), None);
    let error = result.expect_err("filesystem root lacks .engram/");
    assert!(matches!(error, CoreError::ProjectDirNotFound));
}

#[test]
fn resolve_project_dir_with_nonexistent_start() {
    let _lock = lock_env();
    // Walk-up from a path that does not exist on disk: PathBuf::pop works regardless
    // of existence, and is_dir() returns false without panicking. Should cleanly
    // terminate with ProjectDirNotFound at filesystem root.
    let nonexistent: PathBuf = PathBuf::from("/this/path/does/not/exist/nowhere");
    let result = config::resolve_project_dir(&nonexistent, None);
    let error = result.expect_err("nonexistent start has no .engram/ ancestor");
    assert!(matches!(error, CoreError::ProjectDirNotFound));
}
