#![cfg(feature = "local")]

use engram_llm_client::error::ApiError;

// ── Error display ───────────────────────────────────────────────────────

#[test]
fn display_local_model_load_failed() {
    let error = ApiError::LocalModelLoadFailed("file not found".into());
    assert_eq!(
        error.to_string(),
        "[2006] local model load failed: file not found"
    );
}

#[test]
fn local_model_load_failed_is_not_retryable() {
    let error = ApiError::LocalModelLoadFailed("broken".into());
    assert!(!error.is_retryable());
}

#[test]
fn display_local_inference_failed() {
    let error = ApiError::LocalInferenceFailed("session poisoned".into());
    assert_eq!(
        error.to_string(),
        "[2007] local inference failed: session poisoned"
    );
}

#[test]
fn local_inference_failed_is_not_retryable() {
    let error = ApiError::LocalInferenceFailed("broken".into());
    assert!(!error.is_retryable());
}

// ── Constructor error paths ─────────────────────────────────────────────

#[test]
fn local_generator_missing_model_file_returns_error() {
    use engram_llm_client::local::LocalTextGenerator;

    let result = LocalTextGenerator::new(
        "/nonexistent/path/model.onnx",
        "/nonexistent/path/tokenizer.json",
    );

    let Err(error) = result else {
        panic!("expected LocalModelLoadFailed error");
    };
    assert!(matches!(error, ApiError::LocalModelLoadFailed(_)));
    assert!(error.to_string().contains("[2006]"));
}

#[test]
fn local_generator_missing_tokenizer_returns_error() {
    use engram_llm_client::local::LocalTextGenerator;
    use std::io::Write;

    let temp_dir = std::env::temp_dir().join("engram_test_local");
    std::fs::create_dir_all(&temp_dir).unwrap();

    let fake_model = temp_dir.join("fake_model.onnx");
    let mut file = std::fs::File::create(&fake_model).unwrap();
    file.write_all(b"not a real onnx model").unwrap();

    let result = LocalTextGenerator::new(
        fake_model.to_str().unwrap(),
        "/nonexistent/tokenizer.json",
    );

    let Err(error) = result else {
        panic!("expected LocalModelLoadFailed error");
    };
    assert!(matches!(error, ApiError::LocalModelLoadFailed(_)));

    std::fs::remove_dir_all(&temp_dir).ok();
}
