use std::path::{Path, PathBuf};
use std::sync::Mutex;

use tempfile::tempdir;

use engram_core::Config;
use engram_core::config;
use engram_core::error::CoreError;

const PROVIDER_ENV_VARS: [&str; 7] = [
    "ENGRAM_EMBEDDING_MODEL",
    "ENGRAM_LLM_MODEL",
    "ENGRAM_TRAINER_BINARY",
    "ENGRAM_TRAINER_TIMEOUT",
    "ENGRAM_MODELS_PATH",
    "ENGRAM_VOYAGE_API_KEY",
    "ENGRAM_OPENAI_API_KEY",
];

/// Snapshots `HOME`, the current working directory, `ENGRAM_PROJECT_DIR`, and
/// every provider `ENGRAM_*` override that `Config::load()` consults, then
/// restores all of them on drop. Tests that exercise `Config::load()` mutate
/// process-global state; without this guard a leaked env var or cwd would
/// poison every other test in the file.
struct ConfigLoadEnvironment {
    home: Option<String>,
    current_directory: PathBuf,
    project_directory: Option<String>,
    provider_overrides: Vec<(&'static str, Option<String>)>,
}

impl ConfigLoadEnvironment {
    fn capture() -> Self {
        let provider_overrides = PROVIDER_ENV_VARS
            .iter()
            .map(|name| (*name, std::env::var(name).ok()))
            .collect();
        Self {
            home: std::env::var("HOME").ok(),
            current_directory: std::env::current_dir().expect("current dir"),
            project_directory: std::env::var("ENGRAM_PROJECT_DIR").ok(),
            provider_overrides,
        }
    }

    /// Points `Config::load()` at `home` as the global config root and `cwd`
    /// as the project-discovery starting directory, with all provider env
    /// overrides cleared so only the file layers are exercised.
    fn redirect(home: &Path, current_directory: &Path) {
        // SAFETY: every caller holds `lock_env()`, serializing env mutation.
        unsafe {
            std::env::set_var("HOME", home);
            std::env::remove_var("ENGRAM_PROJECT_DIR");
            for name in PROVIDER_ENV_VARS {
                std::env::remove_var(name);
            }
        }
        std::env::set_current_dir(current_directory).expect("set cwd");
    }
}

impl Drop for ConfigLoadEnvironment {
    fn drop(&mut self) {
        std::env::set_current_dir(&self.current_directory).expect("restore cwd");
        // SAFETY: every caller holds `lock_env()`, serializing env mutation.
        unsafe {
            match &self.home {
                Some(value) => std::env::set_var("HOME", value),
                None => std::env::remove_var("HOME"),
            }
            match &self.project_directory {
                Some(value) => std::env::set_var("ENGRAM_PROJECT_DIR", value),
                None => std::env::remove_var("ENGRAM_PROJECT_DIR"),
            }
            for (name, value) in &self.provider_overrides {
                match value {
                    Some(value) => std::env::set_var(name, value),
                    None => std::env::remove_var(name),
                }
            }
        }
    }
}

fn write_global_config(home: &Path, contents: &str) {
    let engram_dir = home.join(".engram");
    std::fs::create_dir_all(&engram_dir).expect("create global .engram");
    std::fs::write(engram_dir.join("engram.toml"), contents).expect("write global config");
}

fn write_project_local_config(project_dir: &Path, contents: &str) {
    let engram_dir = project_dir.join(".engram");
    std::fs::create_dir_all(&engram_dir).expect("create project .engram");
    std::fs::write(engram_dir.join("engram.toml"), contents).expect("write project config");
}

const FULL_GLOBAL_CONFIG: &str = r#"
[database]

[embedding]
provider = "voyage"
api_key = "global-voyage-key"
model = "voyage-4"
dimension = 1024
hyde_threshold = 5

[llm]
provider = "openai"
api_key = "global-openai-key"
model = "gpt-4o-mini"

[server]
reindex_interval_secs = 3600

[hnsw]
max_connections = 16
ef_construction = 200
ef_search = 40
dimension = 1024

[consolidation]
stale_days = 90
min_score = 0.3
"#;

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

#[test]
fn load_uses_global_only_when_no_project_local() {
    let _lock = lock_env();
    let _environment = ConfigLoadEnvironment::capture();
    let home = tempdir().expect("home");
    let project = tempdir().expect("project");
    // Project has a `.engram/` marker but no project-local engram.toml.
    std::fs::create_dir_all(project.path().join(".engram")).expect("create marker");
    write_global_config(home.path(), FULL_GLOBAL_CONFIG);
    ConfigLoadEnvironment::redirect(home.path(), project.path());

    let config = Config::load().expect("global-only load succeeds");

    assert_eq!(config.embedding.hyde_threshold, 5);
    assert_eq!(config.embedding.model.as_deref(), Some("voyage-4"));
    assert_eq!(
        config.embedding.api_key.as_deref(),
        Some("global-voyage-key")
    );
}

#[test]
fn load_merges_project_local_over_global() {
    let _lock = lock_env();
    let _environment = ConfigLoadEnvironment::capture();
    let home = tempdir().expect("home");
    let project = tempdir().expect("project");
    write_global_config(home.path(), FULL_GLOBAL_CONFIG);
    write_project_local_config(
        project.path(),
        r#"
[embedding]
model = "voyage-code-3"
hyde_threshold = 25

[hnsw]
ef_search = 80
"#,
    );
    ConfigLoadEnvironment::redirect(home.path(), project.path());

    let config = Config::load().expect("layered load succeeds");

    // Project-local scalars override global field-by-field.
    assert_eq!(config.embedding.model.as_deref(), Some("voyage-code-3"));
    assert_eq!(config.embedding.hyde_threshold, 25);
    assert_eq!(config.hnsw.ef_search, 80);
    // Untouched global values survive the merge.
    assert_eq!(config.hnsw.max_connections, 16);
    assert_eq!(config.embedding.dimension, Some(1024));
}

#[test]
fn load_project_local_partial_inherits_global() {
    let _lock = lock_env();
    let _environment = ConfigLoadEnvironment::capture();
    let home = tempdir().expect("home");
    let project = tempdir().expect("project");
    write_global_config(home.path(), FULL_GLOBAL_CONFIG);
    // A single-field project-local config — every other field comes from global.
    write_project_local_config(
        project.path(),
        r#"
[embedding]
hyde_threshold = 12
"#,
    );
    ConfigLoadEnvironment::redirect(home.path(), project.path());

    let config = Config::load().expect("partial layered load succeeds");

    assert_eq!(config.embedding.hyde_threshold, 12);
    assert_eq!(config.embedding.provider, "voyage");
    assert_eq!(config.llm.provider, "openai");
    assert_eq!(config.server.reindex_interval_secs, 3600);
    assert_eq!(config.hnsw.dimension, 1024);
}

#[test]
fn load_project_local_api_key_is_ignored() {
    let _lock = lock_env();
    let _environment = ConfigLoadEnvironment::capture();
    let home = tempdir().expect("home");
    let project = tempdir().expect("project");
    write_global_config(home.path(), FULL_GLOBAL_CONFIG);
    // Project-local tries to override the secret — the invariant must reject it.
    write_project_local_config(
        project.path(),
        r#"
[embedding]
api_key = "project-local-voyage-key"

[llm]
api_key = "project-local-openai-key"
"#,
    );
    ConfigLoadEnvironment::redirect(home.path(), project.path());

    let config = Config::load().expect("layered load succeeds");

    assert_eq!(
        config.embedding.api_key.as_deref(),
        Some("global-voyage-key")
    );
    assert_eq!(config.llm.api_key.as_deref(), Some("global-openai-key"));
}

#[test]
fn load_project_local_api_key_ignored_when_global_has_none() {
    let _lock = lock_env();
    let _environment = ConfigLoadEnvironment::capture();
    let home = tempdir().expect("home");
    let project = tempdir().expect("project");
    // Global config without any api_key fields.
    write_global_config(
        home.path(),
        r#"
[database]

[embedding]
provider = "voyage"
model = "voyage-4"
dimension = 1024

[llm]
provider = "openai"
model = "gpt-4o-mini"

[server]
reindex_interval_secs = 3600

[hnsw]
max_connections = 16
ef_construction = 200
ef_search = 40
dimension = 1024
"#,
    );
    write_project_local_config(
        project.path(),
        r#"
[embedding]
api_key = "project-local-voyage-key"

[llm]
api_key = "project-local-openai-key"
"#,
    );
    ConfigLoadEnvironment::redirect(home.path(), project.path());

    let config = Config::load().expect("layered load succeeds");

    // Global has no key, so the merged tree's project-local key is removed.
    assert_eq!(config.embedding.api_key, None);
    assert_eq!(config.llm.api_key, None);
}

#[test]
fn load_returns_default_when_no_config_files() {
    let _lock = lock_env();
    let _environment = ConfigLoadEnvironment::capture();
    let home = tempdir().expect("home");
    let project = tempdir().expect("project");
    // Neither a global config nor a project `.engram/` marker exists.
    ConfigLoadEnvironment::redirect(home.path(), project.path());

    let config = Config::load().expect("default load succeeds");

    let default_config = Config::default();
    assert_eq!(config.embedding.provider, default_config.embedding.provider);
    assert_eq!(
        config.embedding.hyde_threshold,
        default_config.embedding.hyde_threshold
    );
    assert_eq!(config.hnsw.dimension, default_config.hnsw.dimension);
    assert_eq!(
        config.server.reindex_interval_secs,
        default_config.server.reindex_interval_secs
    );
}

#[test]
fn load_invalid_project_local_returns_parse_error() {
    let _lock = lock_env();
    let _environment = ConfigLoadEnvironment::capture();
    let home = tempdir().expect("home");
    let project = tempdir().expect("project");
    write_global_config(home.path(), FULL_GLOBAL_CONFIG);
    write_project_local_config(project.path(), "this is not valid toml [[[");
    ConfigLoadEnvironment::redirect(home.path(), project.path());

    let result = Config::load();

    assert!(matches!(result, Err(CoreError::ConfigParseError(_))));
}

#[test]
fn load_uses_global_only_when_no_project_marker() {
    let _lock = lock_env();
    let _environment = ConfigLoadEnvironment::capture();
    let home = tempdir().expect("home");
    // Start directory with no `.engram/` marker anywhere up to filesystem root
    // is impossible to guarantee, so point cwd at a tempdir whose ancestors
    // lack the marker — `resolve_project_dir` returns ProjectDirNotFound which
    // `Config::load()` must swallow, yielding the global-only config.
    let cwd = tempdir().expect("cwd without marker");
    write_global_config(home.path(), FULL_GLOBAL_CONFIG);
    ConfigLoadEnvironment::redirect(home.path(), cwd.path());

    let config = Config::load().expect("global-only load succeeds");

    assert_eq!(config.embedding.hyde_threshold, 5);
    assert_eq!(
        config.embedding.api_key.as_deref(),
        Some("global-voyage-key")
    );
}

#[test]
fn load_project_only_merges_over_defaults() {
    let _lock = lock_env();
    let _environment = ConfigLoadEnvironment::capture();
    let home = tempdir().expect("home");
    let project = tempdir().expect("project");
    // No global config file is written — the only file layer is project-local,
    // so the merge base is `toml::Value::try_from(Config::default())`.
    write_project_local_config(
        project.path(),
        r#"
[embedding]
hyde_threshold = 7
"#,
    );
    ConfigLoadEnvironment::redirect(home.path(), project.path());

    let config = Config::load().expect("project-only load succeeds");

    let default_config = Config::default();
    // Project-local overlay applied over the built-in defaults.
    assert_eq!(config.embedding.hyde_threshold, 7);
    // Every untouched field falls through to `Config::default()`.
    assert_eq!(config.embedding.provider, default_config.embedding.provider);
    assert_eq!(config.embedding.model, default_config.embedding.model);
    assert_eq!(config.embedding.api_key, None);
    assert_eq!(config.hnsw.dimension, default_config.hnsw.dimension);
    assert_eq!(
        config.server.reindex_interval_secs,
        default_config.server.reindex_interval_secs
    );
}

#[test]
fn load_global_secret_survives_partial_project_section() {
    let _lock = lock_env();
    let _environment = ConfigLoadEnvironment::capture();
    let home = tempdir().expect("home");
    let project = tempdir().expect("project");
    write_global_config(home.path(), FULL_GLOBAL_CONFIG);
    // Project-local touches `[embedding]` and `[llm]` without an `api_key` —
    // the global secret must survive the partial-section merge.
    write_project_local_config(
        project.path(),
        r#"
[embedding]
model = "voyage-code-3"

[llm]
model = "gpt-4o"
"#,
    );
    ConfigLoadEnvironment::redirect(home.path(), project.path());

    let config = Config::load().expect("layered load succeeds");

    assert_eq!(config.embedding.model.as_deref(), Some("voyage-code-3"));
    assert_eq!(config.llm.model.as_deref(), Some("gpt-4o"));
    assert_eq!(
        config.embedding.api_key.as_deref(),
        Some("global-voyage-key")
    );
    assert_eq!(config.llm.api_key.as_deref(), Some("global-openai-key"));
}

#[test]
fn load_empty_project_local_inherits_global_fully() {
    let _lock = lock_env();
    let _environment = ConfigLoadEnvironment::capture();
    let home = tempdir().expect("home");
    let project = tempdir().expect("project");
    write_global_config(home.path(), FULL_GLOBAL_CONFIG);
    // A comment-only project-local config — exactly what `init` ships via the
    // fully commented `PROJECT_LOCAL_CONFIG_TEMPLATE`. It parses to an empty
    // table, so every field must come from the global layer.
    write_project_local_config(
        project.path(),
        "# engram project-local config\n# all settings inherited from global\n",
    );
    ConfigLoadEnvironment::redirect(home.path(), project.path());

    let config = Config::load().expect("empty project-local load succeeds");

    assert_eq!(config.embedding.provider, "voyage");
    assert_eq!(config.embedding.hyde_threshold, 5);
    assert_eq!(config.hnsw.dimension, 1024);
    assert_eq!(
        config.embedding.api_key.as_deref(),
        Some("global-voyage-key")
    );
}

#[test]
fn load_invalid_global_returns_parse_error() {
    let _lock = lock_env();
    let _environment = ConfigLoadEnvironment::capture();
    let home = tempdir().expect("home");
    let project = tempdir().expect("project");
    // Malformed global config, no project-local layer at all.
    write_global_config(home.path(), "not toml [[[");
    ConfigLoadEnvironment::redirect(home.path(), project.path());

    let result = Config::load();

    assert!(matches!(result, Err(CoreError::ConfigParseError(_))));
}

#[test]
fn load_deduplication_threshold_round_trips() {
    let _lock = lock_env();
    let _environment = ConfigLoadEnvironment::capture();
    let home = tempdir().expect("home");
    let project = tempdir().expect("project");
    write_global_config(
        home.path(),
        &format!("{FULL_GLOBAL_CONFIG}\n[deduplication]\nthreshold = 0.88\n"),
    );
    ConfigLoadEnvironment::redirect(home.path(), project.path());

    let config = Config::load().expect("config with [deduplication] loads");

    assert_eq!(config.deduplication.threshold, 0.88);
}

#[test]
fn load_without_deduplication_section_uses_default() {
    let _lock = lock_env();
    let _environment = ConfigLoadEnvironment::capture();
    let home = tempdir().expect("home");
    let project = tempdir().expect("project");
    // `FULL_GLOBAL_CONFIG` has no `[deduplication]` section — regression guard
    // for the `#[serde(default)]` requirement. Existing on-disk configs written
    // before this section existed must still deserialize.
    write_global_config(home.path(), FULL_GLOBAL_CONFIG);
    ConfigLoadEnvironment::redirect(home.path(), project.path());

    let config = Config::load().expect("config without [deduplication] still loads");

    assert_eq!(config.deduplication.threshold, 0.95);
}
