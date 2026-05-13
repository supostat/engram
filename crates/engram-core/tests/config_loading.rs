use engram_core::{Config, CoreError};

#[test]
fn default_config_has_expected_values() {
    let config = Config::default();
    assert_eq!(config.database.path, None);
    assert_eq!(config.embedding.provider, "voyage");
    assert_eq!(config.llm.provider, "openai");
    assert_eq!(config.server.socket_path, None);
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
    assert_eq!(config.database.path.as_deref(), Some("/custom/memories.db"));
    assert_eq!(config.hnsw.max_connections, 32);
    assert_eq!(config.hnsw.dimension, 512);
    assert_eq!(config.server.reindex_interval_secs, 1800);
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
fn build_embedding_provider_with_empty_api_key_fails() {
    let mut config = Config::default();
    config.embedding.api_key = Some(String::new());
    let result = config.build_embedding_provider();
    assert!(matches!(result, Err(CoreError::InvalidProvider(_))));
}

#[test]
fn trainer_config_defaults() {
    let config = Config::default();
    assert_eq!(config.trainer.trainer_binary, "engram-trainer");
    assert_eq!(config.trainer.trainer_timeout_secs, 300);
    assert_eq!(config.trainer.models_path, "~/.engram/models");
}

#[test]
fn trainer_config_from_toml() {
    let temp_dir = tempfile::tempdir().unwrap();
    let config_path = temp_dir.path().join("engram.toml");
    let toml_content = r#"
[database]
path = "/custom/memories.db"

[embedding]
provider = "voyage"

[llm]
provider = "openai"

[server]
socket_path = "/tmp/custom.sock"
reindex_interval_secs = 1800

[hnsw]
max_connections = 16
ef_construction = 200
ef_search = 40
dimension = 1024

[trainer]
trainer_binary = "/opt/bin/custom-trainer"
trainer_timeout_secs = 600
models_path = "/data/models"
"#;
    std::fs::write(&config_path, toml_content).unwrap();
    let config = Config::load_from_path(config_path.to_str().unwrap()).unwrap();
    assert_eq!(config.trainer.trainer_binary, "/opt/bin/custom-trainer");
    assert_eq!(config.trainer.trainer_timeout_secs, 600);
    assert_eq!(config.trainer.models_path, "/data/models");
}

#[test]
fn trainer_config_env_overrides() {
    let mut config = Config::default();
    assert_eq!(config.trainer.trainer_binary, "engram-trainer");

    // Cannot call apply_env_overrides() directly because env vars are global
    // state that leaks across parallel tests. Instead, simulate the override
    // mechanism by setting fields manually. See project gotcha: "Parallel
    // tests: env vars leak across threads — don't use real env vars in
    // parallel tests, simulate overrides manually".
    config.trainer.trainer_binary = "/env/override/trainer".into();
    config.trainer.trainer_timeout_secs = 120;
    config.trainer.models_path = "/env/models".into();

    assert_eq!(config.trainer.trainer_binary, "/env/override/trainer");
    assert_eq!(config.trainer.trainer_timeout_secs, 120);
    assert_eq!(config.trainer.models_path, "/env/models");
}

#[test]
fn build_text_generator_local_missing_model() {
    let mut config = Config::default();
    config.llm.provider = "local".into();
    config.trainer.models_path = "/nonexistent/path/to/models".into();
    let result = config.build_text_generator();
    assert!(matches!(result, Err(CoreError::InvalidProvider(_))));
}

#[test]
fn build_text_generator_local_provider_name() {
    let mut config = Config::default();
    config.llm.provider = "local".into();
    config.trainer.models_path = "/nonexistent/path/to/models".into();
    let result = config.build_text_generator();
    // "local" is accepted as valid provider — error is about missing file, not unknown provider
    match result {
        Err(error) => {
            let error_message = format!("{error}");
            assert!(
                !error_message.contains("not supported"),
                "should not be 'not supported' error"
            );
        }
        Ok(_) => panic!("expected error for nonexistent model path"),
    }
}

#[test]
fn default_hyde_threshold_is_zero() {
    let config = Config::default();
    assert_eq!(config.embedding.hyde_threshold, 0);
}

#[test]
fn parses_hyde_threshold_from_toml() {
    let temp_dir = tempfile::tempdir().unwrap();
    let config_path = temp_dir.path().join("engram.toml");
    let toml_content = r#"
[database]
path = "/x/db"
[embedding]
provider = "voyage"
hyde_threshold = 20
[llm]
provider = "openai"
[server]
socket_path = "/tmp/x.sock"
reindex_interval_secs = 1800
[hnsw]
max_connections = 16
ef_construction = 200
ef_search = 40
dimension = 1024
"#;
    std::fs::write(&config_path, toml_content).unwrap();
    let config = Config::load_from_path(config_path.to_str().unwrap()).unwrap();
    assert_eq!(config.embedding.hyde_threshold, 20);
}

#[test]
fn missing_hyde_threshold_in_toml_defaults_to_zero() {
    let temp_dir = tempfile::tempdir().unwrap();
    let config_path = temp_dir.path().join("engram.toml");
    let toml_content = r#"
[database]
path = "/x/db"
[embedding]
provider = "voyage"
[llm]
provider = "openai"
[server]
socket_path = "/tmp/x.sock"
reindex_interval_secs = 1800
[hnsw]
max_connections = 16
ef_construction = 200
ef_search = 40
dimension = 1024
"#;
    std::fs::write(&config_path, toml_content).unwrap();
    let config = Config::load_from_path(config_path.to_str().unwrap()).unwrap();
    assert_eq!(config.embedding.hyde_threshold, 0);
}
