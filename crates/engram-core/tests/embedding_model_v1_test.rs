use engram_core::error::CoreError;
use engram_core::migrations::embedding_model_v1;
use engram_storage::Database;

#[test]
fn bootstrap_writes_model_on_empty_schema_meta() {
    let database = Database::in_memory().expect("in-memory database");

    let result = embedding_model_v1::check(&database, "voyage-code-3");

    assert!(result.is_ok(), "bootstrap on empty meta should succeed");
    // Subsequent check with the same model must pass (proves the write happened).
    assert!(embedding_model_v1::check(&database, "voyage-code-3").is_ok());
}

#[test]
fn check_passes_when_stored_matches_configured() {
    let database = Database::in_memory().expect("in-memory database");
    embedding_model_v1::record(&database, "voyage-code-3").unwrap();

    assert!(embedding_model_v1::check(&database, "voyage-code-3").is_ok());
}

#[test]
fn check_fails_with_6020_when_mismatch() {
    let database = Database::in_memory().expect("in-memory database");
    embedding_model_v1::record(&database, "voyage-code-3").unwrap();

    let error = embedding_model_v1::check(&database, "voyage-4").unwrap_err();

    assert!(
        matches!(
            &error,
            CoreError::EmbeddingModelMismatch { stored, configured }
                if stored == "voyage-code-3" && configured == "voyage-4"
        ),
        "expected EmbeddingModelMismatch, got {error:?}"
    );
    let message = format!("{error}");
    assert!(
        message.contains("[6020]"),
        "message missing code: {message}"
    );
    assert!(
        message.contains("engram reembed"),
        "message must instruct user: {message}"
    );
}

#[test]
fn record_overwrites_existing_value() {
    let database = Database::in_memory().expect("in-memory database");
    embedding_model_v1::record(&database, "voyage-code-3").unwrap();
    embedding_model_v1::record(&database, "voyage-4").unwrap();

    // After record-overwrite, check against the new model must pass.
    assert!(embedding_model_v1::check(&database, "voyage-4").is_ok());
    // And check against the old model must now fail.
    assert!(embedding_model_v1::check(&database, "voyage-code-3").is_err());
}
