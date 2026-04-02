use engram_consolidate::ConsolidateError;
use engram_core::CoreError;
use engram_hnsw::HnswError;
use engram_llm_client::ApiError;
use engram_storage::StorageError;

#[test]
fn config_not_found_display() {
    let error = CoreError::ConfigNotFound;
    assert_eq!(error.to_string(), "[6001] config not found");
}

#[test]
fn config_parse_error_display() {
    let error = CoreError::ConfigParseError("bad toml".into());
    assert_eq!(error.to_string(), "[6002] config parse error: bad toml");
}

#[test]
fn invalid_provider_display() {
    let error = CoreError::InvalidProvider("unknown".into());
    assert_eq!(error.to_string(), "[6003] invalid provider: unknown");
}

#[test]
fn index_corrupted_display() {
    let error = CoreError::IndexCorrupted("bad magic".into());
    assert_eq!(error.to_string(), "[6004] index corrupted: bad magic");
}

#[test]
fn rebuild_failed_display() {
    let error = CoreError::RebuildFailed("disk full".into());
    assert_eq!(error.to_string(), "[6005] rebuild failed: disk full");
}

#[test]
fn socket_error_display() {
    let error = CoreError::SocketError("connection refused".into());
    assert_eq!(error.to_string(), "[6006] socket error: connection refused");
}

#[test]
fn dispatch_error_display() {
    let error = CoreError::DispatchError("unknown method".into());
    assert_eq!(error.to_string(), "[6007] dispatch error: unknown method");
}

#[test]
fn from_storage_error() {
    let storage = StorageError::NotFound("memory id=42".into());
    let core: CoreError = storage.into();
    assert!(core.to_string().contains("[1002]"));
}

#[test]
fn from_hnsw_error() {
    let hnsw = HnswError::RebuildRequired;
    let core: CoreError = hnsw.into();
    assert!(core.to_string().contains("[3003]"));
}

#[test]
fn from_api_error() {
    let api = ApiError::InvalidApiKey("bad key".into());
    let core: CoreError = api.into();
    assert!(core.to_string().contains("[2004]"));
}

#[test]
fn from_consolidate_error() {
    let consolidate = ConsolidateError::NoCandidates;
    let core: CoreError = consolidate.into();
    assert!(core.to_string().contains("[5001]"));
}

#[test]
fn error_source_delegates_for_storage() {
    use std::error::Error;
    let storage = StorageError::NotFound("test".into());
    let core = CoreError::Storage(storage);
    assert!(core.source().is_some());
}

#[test]
fn error_source_none_for_own_variants() {
    use std::error::Error;
    let core = CoreError::ConfigNotFound;
    assert!(core.source().is_none());
}

#[test]
fn trainer_failed_display() {
    let error = CoreError::TrainerFailed("/usr/bin/engram-trainer".into());
    assert_eq!(
        error.to_string(),
        "[6013] trainer failed: /usr/bin/engram-trainer"
    );
}

#[test]
fn trainer_timeout_display() {
    let error = CoreError::TrainerTimeout;
    assert_eq!(error.to_string(), "[6014] trainer timeout");
}

#[test]
fn trainer_malformed_output_display() {
    let error = CoreError::TrainerMalformedOutput("invalid json at line 3".into());
    assert_eq!(
        error.to_string(),
        "[6015] trainer malformed output: invalid json at line 3"
    );
}
