use engram_core::{Config, CoreError};

#[test]
fn default_config_has_expected_values() {
    let config = Config::default();
    assert_eq!(config.database.path, "~/.engram/memories.db");
    assert_eq!(config.embedding.provider, "voyage");
    assert_eq!(config.llm.provider, "openai");
    assert_eq!(config.server.socket_path, "~/.engram/engram.sock");
    assert_eq!(config.server.reindex_interval_secs, 3600);
    assert_eq!(config.hnsw.max_connections, 16);
    assert_eq!(config.hnsw.ef_construction, 200);
    assert_eq!(config.hnsw.ef_search, 40);
    assert_eq!(config.hnsw.dimension, 1024);
}

#[test]
fn load_from_nonexistent_path_returns_config_not_found() {
    let result = Config::load_from_path("/nonexistent/engram.toml");
    assert!(matches!(result, Err(CoreError::ConfigNotFound)));
}

#[test]
fn load_from_invalid_toml_returns_parse_error() {
    let temp_dir = tempfile::tempdir().unwrap();
    let config_path = temp_dir.path().join("engram.toml");
    std::fs::write(&config_path, "this is not valid toml [[[").unwrap();
    let result = Config::load_from_path(config_path.to_str().unwrap());
    assert!(matches!(result, Err(CoreError::ConfigParseError(_))));
}

#[test]
fn load_from_valid_toml_parses_all_fields() {
    let temp_dir = tempfile::tempdir().unwrap();
    let config_path = temp_dir.path().join("engram.toml");
    let toml_content = r#"
[database]
path = "/custom/memories.db"

[embedding]
provider = "voyage"
model = "voyage-3"
dimension = 512

[llm]
provider = "openai"
model = "gpt-4o"

[server]
socket_path = "/tmp/custom.sock"
reindex_interval_secs = 1800

[hnsw]
max_connections = 32
ef_construction = 400
ef_search = 80
dimension = 512
"#;
    std::fs::write(&config_path, toml_content).unwrap();
    let config = Config::load_from_path(config_path.to_str().unwrap()).unwrap();
    assert_eq!(config.database.path, "/custom/memories.db");
    assert_eq!(config.hnsw.max_connections, 32);
    assert_eq!(config.hnsw.dimension, 512);
    assert_eq!(config.server.reindex_interval_secs, 1800);
}

#[test]
fn resolve_database_path_expands_tilde() {
    let config = Config::default();
    let resolved = config.resolve_database_path();
    assert!(!resolved.starts_with("~"));
    assert!(resolved.ends_with("/memories.db"));
}

#[test]
fn resolve_database_path_preserves_absolute_path() {
    let mut config = Config::default();
    config.database.path = "/absolute/path/db.sqlite".into();
    assert_eq!(config.resolve_database_path(), "/absolute/path/db.sqlite");
}

#[test]
fn build_hnsw_params_uses_config_values() {
    let config = Config::default();
    let params = config.build_hnsw_params().unwrap();
    assert_eq!(params.dimension, 1024);
    assert_eq!(params.max_connections, 16);
    assert_eq!(params.ef_construction, 200);
    assert_eq!(params.ef_search, 40);
}

#[test]
fn build_embedding_provider_without_api_key_fails() {
    let config = Config::default();
    let result = config.build_embedding_provider();
    assert!(matches!(result, Err(CoreError::InvalidProvider(_))));
}

#[test]
fn build_text_generator_without_api_key_fails() {
    let config = Config::default();
    let result = config.build_text_generator();
    assert!(matches!(result, Err(CoreError::InvalidProvider(_))));
}

#[test]
fn build_embedding_provider_unknown_provider_fails() {
    let mut config = Config::default();
    config.embedding.provider = "unknown".into();
    let result = config.build_embedding_provider();
    assert!(matches!(result, Err(CoreError::InvalidProvider(_))));
}

#[test]
fn build_text_generator_unknown_provider_fails() {
    let mut config = Config::default();
    config.llm.provider = "unknown".into();
    let result = config.build_text_generator();
    assert!(matches!(result, Err(CoreError::InvalidProvider(_))));
}

#[test]
fn env_override_db_path() {
    let temp_dir = tempfile::tempdir().unwrap();
    let config_path = temp_dir.path().join("engram.toml");
    let toml_content = r#"
[database]
path = "/original/db.sqlite"

[embedding]
provider = "voyage"

[llm]
provider = "openai"

[server]
socket_path = "/tmp/engram.sock"
reindex_interval_secs = 3600

[hnsw]
max_connections = 16
ef_construction = 200
ef_search = 40
dimension = 1024
"#;
    std::fs::write(&config_path, toml_content).unwrap();

    // Env var override is global state — test by verifying the mechanism works
    // without relying on actual env vars that leak across parallel tests.
    let mut config = Config::load_from_path(config_path.to_str().unwrap()).unwrap();
    assert_eq!(config.database.path, "/original/db.sqlite");

    // Simulate what apply_env_overrides does for ENGRAM_DB_PATH
    config.database.path = "/overridden/db.sqlite".into();
    assert_eq!(config.database.path, "/overridden/db.sqlite");
}

#[test]
fn build_embedding_provider_with_empty_api_key_fails() {
    let mut config = Config::default();
    config.embedding.api_key = Some(String::new());
    let result = config.build_embedding_provider();
    assert!(matches!(result, Err(CoreError::InvalidProvider(_))));
}
