//! Path resolution for engram-tui under ADR 2026-04-22 per-project layout.
//!
//! Walk-up logic mirrors `engram_core::config::resolve_project_dir`, but TUI
//! returns `Option` (orphan view fallback) instead of `Result` (server/CLI fail-fast):
//! TUI may legitimately display global `~/.engram/engram.db` data when no project
//! is found in cwd ancestors. Server/CLI cannot operate without project context.
//!
//! No dependency on `engram-core` to avoid pulling in tokio/HNSW/embedding deps.

use std::path::{Path, PathBuf};

const PROJECT_DIR_MARKER: &str = ".engram";
const PROJECT_DB_RELATIVE: &str = ".engram/engram.db";
const PROJECT_SOCKET_RELATIVE: &str = ".engram/engram.sock";
const PROJECT_HNSW_RELATIVE: &str = ".engram/indexes.hnsw";
const GLOBAL_DB: &str = "~/.engram/engram.db";
const GLOBAL_SOCKET: &str = "~/.engram/engram.sock";
const GLOBAL_HNSW: &str = "~/.engram/indexes.hnsw";
const GLOBAL_MODELS: &str = "~/.engram/models";

pub fn resolve_project_dir(start: &Path, explicit_override: Option<&Path>) -> Option<PathBuf> {
    if let Some(path) = explicit_override {
        return Some(path.to_path_buf());
    }
    if let Ok(env_path) = std::env::var("ENGRAM_PROJECT_DIR") {
        let candidate = PathBuf::from(env_path);
        if candidate.is_absolute() {
            return Some(candidate);
        }
    }
    let mut current = start.to_path_buf();
    loop {
        if current.join(PROJECT_DIR_MARKER).is_dir() {
            return Some(current);
        }
        if !current.pop() {
            return None;
        }
    }
}

pub fn resolve_database_path(cwd: &Path, explicit: Option<String>) -> String {
    if let Some(path) = explicit {
        return expand_tilde(&path);
    }
    if let Ok(env_path) = std::env::var("ENGRAM_DB_PATH") {
        return env_path;
    }
    if let Some(project_dir) = resolve_project_dir(cwd, None) {
        return project_dir
            .join(PROJECT_DB_RELATIVE)
            .to_string_lossy()
            .into_owned();
    }
    expand_tilde(GLOBAL_DB)
}

pub fn resolve_socket_path(cwd: &Path, explicit: Option<String>) -> String {
    if let Some(path) = explicit {
        return expand_tilde(&path);
    }
    if let Ok(env_path) = std::env::var("ENGRAM_SOCKET_PATH") {
        return env_path;
    }
    if let Some(project_dir) = resolve_project_dir(cwd, None) {
        return project_dir
            .join(PROJECT_SOCKET_RELATIVE)
            .to_string_lossy()
            .into_owned();
    }
    expand_tilde(GLOBAL_SOCKET)
}

pub fn resolve_hnsw_path(cwd: &Path) -> String {
    if let Some(project_dir) = resolve_project_dir(cwd, None) {
        return project_dir
            .join(PROJECT_HNSW_RELATIVE)
            .to_string_lossy()
            .into_owned();
    }
    expand_tilde(GLOBAL_HNSW)
}

pub fn resolve_models_path(explicit: Option<String>) -> String {
    if let Some(path) = explicit {
        return expand_tilde(&path);
    }
    expand_tilde(GLOBAL_MODELS)
}

pub fn expand_tilde(path: &str) -> String {
    if let Some(rest) = path.strip_prefix("~/")
        && let Some(home) = dirs::home_dir()
    {
        return home.join(rest).to_string_lossy().into_owned();
    }
    path.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use std::fs;
    use tempfile::TempDir;

    const ENV_VARS: &[&str] = &["ENGRAM_PROJECT_DIR", "ENGRAM_DB_PATH", "ENGRAM_SOCKET_PATH"];

    /// SAFETY: tests touching env vars are serialized via `#[serial]`, so no
    /// concurrent reader/writer races the process-global env table.
    fn clear_env_vars() {
        for var in ENV_VARS {
            unsafe { std::env::remove_var(var) };
        }
    }

    fn set_env(key: &str, value: impl AsRef<std::ffi::OsStr>) {
        unsafe { std::env::set_var(key, value) };
    }

    fn make_project(temp: &TempDir) -> &Path {
        fs::create_dir_all(temp.path().join(".engram")).unwrap();
        temp.path()
    }

    fn make_subdir(temp: &TempDir, name: &str) -> PathBuf {
        let path = temp.path().join(name);
        fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    #[serial]
    fn resolve_project_dir_walks_up_to_marker() {
        clear_env_vars();
        let temp = TempDir::new().unwrap();
        let root = make_project(&temp);
        let nested = root.join("a/b/c");
        fs::create_dir_all(&nested).unwrap();
        assert_eq!(resolve_project_dir(&nested, None), Some(root.to_path_buf()));
    }

    #[test]
    #[serial]
    fn resolve_project_dir_returns_none_when_no_marker() {
        clear_env_vars();
        let temp = TempDir::new().unwrap();
        let nested = make_subdir(&temp, "empty/subdir");
        assert!(resolve_project_dir(&nested, None).is_none());
    }

    #[test]
    #[serial]
    fn resolve_project_dir_explicit_override_wins() {
        clear_env_vars();
        let temp = TempDir::new().unwrap();
        let override_path = make_subdir(&temp, "override");
        let found = resolve_project_dir(Path::new("/does/not/matter"), Some(&override_path));
        assert_eq!(found, Some(override_path));
    }

    #[test]
    #[serial]
    fn resolve_project_dir_uses_absolute_env_var() {
        clear_env_vars();
        let temp = TempDir::new().unwrap();
        let env_target = make_subdir(&temp, "env-project");
        set_env("ENGRAM_PROJECT_DIR", &env_target);
        let nested = make_subdir(&temp, "unrelated");
        assert_eq!(resolve_project_dir(&nested, None), Some(env_target));
        clear_env_vars();
    }

    #[test]
    #[serial]
    fn resolve_project_dir_ignores_relative_env_var() {
        clear_env_vars();
        set_env("ENGRAM_PROJECT_DIR", "relative/path");
        let temp = TempDir::new().unwrap();
        let nested = make_subdir(&temp, "empty/inner");
        assert!(resolve_project_dir(&nested, None).is_none());
        clear_env_vars();
    }

    #[test]
    #[serial]
    fn resolve_project_dir_skips_marker_when_file_not_dir() {
        clear_env_vars();
        let temp = TempDir::new().unwrap();
        // Create .engram as a regular file at temp/sub
        let sub = temp.path().join("sub");
        fs::create_dir_all(&sub).unwrap();
        fs::write(sub.join(".engram"), b"this is a file, not a dir").unwrap();
        // Walk-up from sub/inner should skip 'sub' (since .engram is a file there)
        // and continue up. If no .engram dir exists in any ancestor, returns None.
        let inner = sub.join("inner");
        fs::create_dir_all(&inner).unwrap();
        assert!(resolve_project_dir(&inner, None).is_none());
    }

    #[test]
    #[serial]
    fn resolve_database_path_explicit_takes_precedence() {
        clear_env_vars();
        let temp = TempDir::new().unwrap();
        let project = make_project(&temp);
        assert_eq!(
            resolve_database_path(project, Some("/custom/db.sqlite".into())),
            "/custom/db.sqlite"
        );
    }

    #[test]
    #[serial]
    fn resolve_database_path_uses_project_when_found() {
        clear_env_vars();
        let temp = TempDir::new().unwrap();
        let project = make_project(&temp);
        let expected = project
            .join(".engram/engram.db")
            .to_string_lossy()
            .into_owned();
        assert_eq!(resolve_database_path(project, None), expected);
    }

    #[test]
    #[serial]
    fn resolve_database_path_falls_back_to_global() {
        clear_env_vars();
        let temp = TempDir::new().unwrap();
        let nested = make_subdir(&temp, "empty");
        let result = resolve_database_path(&nested, None);
        assert_eq!(result, expand_tilde(GLOBAL_DB));
        assert!(!result.contains('~'), "tilde should be expanded: {result}");
    }

    #[test]
    #[serial]
    fn resolve_database_path_env_var_returns_verbatim() {
        clear_env_vars();
        set_env("ENGRAM_DB_PATH", "/env/db.sqlite");
        let temp = TempDir::new().unwrap();
        assert_eq!(resolve_database_path(temp.path(), None), "/env/db.sqlite");
        clear_env_vars();
    }

    #[test]
    #[serial]
    fn resolve_database_path_explicit_with_tilde_is_expanded() {
        clear_env_vars();
        let temp = TempDir::new().unwrap();
        let result = resolve_database_path(temp.path(), Some("~/custom/db".into()));
        let expected = dirs::home_dir()
            .unwrap()
            .join("custom/db")
            .to_string_lossy()
            .into_owned();
        assert_eq!(result, expected);
        assert!(!result.contains('~'), "tilde should be expanded: {result}");
    }

    #[test]
    #[serial]
    fn resolve_socket_path_uses_project_when_found() {
        clear_env_vars();
        let temp = TempDir::new().unwrap();
        let project = make_project(&temp);
        let expected = project
            .join(".engram/engram.sock")
            .to_string_lossy()
            .into_owned();
        assert_eq!(resolve_socket_path(project, None), expected);
    }

    #[test]
    #[serial]
    fn resolve_socket_path_falls_back_to_global() {
        clear_env_vars();
        let temp = TempDir::new().unwrap();
        let nested = make_subdir(&temp, "empty");
        assert_eq!(
            resolve_socket_path(&nested, None),
            expand_tilde(GLOBAL_SOCKET)
        );
    }

    #[test]
    #[serial]
    fn resolve_socket_path_explicit_takes_precedence() {
        clear_env_vars();
        let temp = TempDir::new().unwrap();
        let project = make_project(&temp);
        assert_eq!(
            resolve_socket_path(project, Some("/custom/engram.sock".into())),
            "/custom/engram.sock"
        );
    }

    #[test]
    #[serial]
    fn resolve_socket_path_explicit_with_tilde_is_expanded() {
        clear_env_vars();
        let temp = TempDir::new().unwrap();
        let result = resolve_socket_path(temp.path(), Some("~/custom/engram.sock".into()));
        let expected = dirs::home_dir()
            .unwrap()
            .join("custom/engram.sock")
            .to_string_lossy()
            .into_owned();
        assert_eq!(result, expected);
        assert!(!result.contains('~'), "tilde should be expanded: {result}");
    }

    #[test]
    #[serial]
    fn resolve_socket_path_env_var_returns_verbatim() {
        clear_env_vars();
        set_env("ENGRAM_SOCKET_PATH", "/env/engram.sock");
        let temp = TempDir::new().unwrap();
        assert_eq!(resolve_socket_path(temp.path(), None), "/env/engram.sock");
        clear_env_vars();
    }

    #[test]
    #[serial]
    fn resolve_hnsw_path_uses_project_when_found() {
        clear_env_vars();
        let temp = TempDir::new().unwrap();
        let project = make_project(&temp);
        let expected = project
            .join(".engram/indexes.hnsw")
            .to_string_lossy()
            .into_owned();
        assert_eq!(resolve_hnsw_path(project), expected);
    }

    #[test]
    #[serial]
    fn resolve_hnsw_path_falls_back_to_global() {
        clear_env_vars();
        let temp = TempDir::new().unwrap();
        let nested = make_subdir(&temp, "empty");
        assert_eq!(resolve_hnsw_path(&nested), expand_tilde(GLOBAL_HNSW));
    }

    #[test]
    fn resolve_models_path_explicit_takes_precedence() {
        assert_eq!(
            resolve_models_path(Some("/custom/models".into())),
            "/custom/models"
        );
    }

    #[test]
    fn resolve_models_path_falls_back_to_global() {
        let result = resolve_models_path(None);
        assert_eq!(result, expand_tilde(GLOBAL_MODELS));
        assert!(!result.contains('~'), "tilde should be expanded: {result}");
    }

    #[test]
    fn expand_tilde_expands_home_prefix() {
        let home = dirs::home_dir().expect("home dir required for test");
        let expected = home.join("foo/bar").to_string_lossy().into_owned();
        assert_eq!(expand_tilde("~/foo/bar"), expected);
    }

    #[test]
    fn expand_tilde_leaves_absolute_unchanged() {
        assert_eq!(expand_tilde("/absolute/path"), "/absolute/path");
    }

    #[test]
    fn expand_tilde_leaves_relative_unchanged() {
        assert_eq!(expand_tilde("relative/path"), "relative/path");
    }

    #[test]
    fn expand_tilde_leaves_bare_tilde_unchanged() {
        assert_eq!(expand_tilde("~"), "~");
    }
}
