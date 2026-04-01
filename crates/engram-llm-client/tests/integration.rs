use engram_llm_client::error::ApiError;
use engram_llm_client::openai::{
    map_llm_error, parse_chat_response, OpenAITextGenerator, DEFAULT_OPENAI_MODEL,
};
use engram_llm_client::provider::{EmbeddingProvider, TextGenerator};
use engram_llm_client::retry::{compute_backoff, execute_with_retry, RetryConfig};
use engram_llm_client::voyage::{
    map_embedding_error, parse_embedding_response, VoyageEmbeddingProvider, DEFAULT_VOYAGE_DIMENSION,
    DEFAULT_VOYAGE_MODEL,
};
use std::sync::atomic::{AtomicU32, Ordering};

struct MockEmbeddingProvider {
    embedding: Vec<f32>,
    model: String,
}

impl EmbeddingProvider for MockEmbeddingProvider {
    fn embed(&self, _text: &str) -> Result<Vec<f32>, ApiError> {
        Ok(self.embedding.clone())
    }

    fn dimension(&self) -> usize {
        self.embedding.len()
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

struct MockTextGenerator {
    response: String,
    model: String,
}

impl TextGenerator for MockTextGenerator {
    fn generate(&self, _prompt: &str) -> Result<String, ApiError> {
        Ok(self.response.clone())
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

// ── Error display ───────────────────────────────────────────────────────

#[test]
fn display_embedding_api_unavailable() {
    let error = ApiError::EmbeddingApiUnavailable("connection refused".into());
    assert_eq!(
        error.to_string(),
        "[2001] embedding api unavailable: connection refused"
    );
}

#[test]
fn display_llm_api_unavailable() {
    let error = ApiError::LlmApiUnavailable("timeout".into());
    assert_eq!(error.to_string(), "[2002] llm api unavailable: timeout");
}

#[test]
fn display_rate_limit_exceeded() {
    let error = ApiError::RateLimitExceeded("too many requests".into());
    assert_eq!(
        error.to_string(),
        "[2003] rate limit exceeded: too many requests"
    );
}

#[test]
fn display_invalid_api_key() {
    let error = ApiError::InvalidApiKey("empty api key".into());
    assert_eq!(error.to_string(), "[2004] invalid api key: empty api key");
}

#[test]
fn display_hyde_generation_failed() {
    let error = ApiError::HyDeGenerationFailed("model error".into());
    assert_eq!(
        error.to_string(),
        "[2005] hyde generation failed: model error"
    );
}

#[test]
fn error_implements_std_error() {
    let error = ApiError::EmbeddingApiUnavailable("test".into());
    let std_error: &dyn std::error::Error = &error;
    assert!(std_error.source().is_none());
}

// ── Retry ───────────────────────────────────────────────────────────────

#[test]
fn retry_config_default_values() {
    let config = RetryConfig::default();
    assert_eq!(config.max_retries, 3);
    assert_eq!(config.initial_backoff_ms, 100);
    assert_eq!(config.max_backoff_ms, 10_000);
    assert!((config.backoff_multiplier - 2.0).abs() < f64::EPSILON);
}

#[test]
fn retry_succeeds_on_first_try() {
    let config = RetryConfig {
        max_retries: 3,
        initial_backoff_ms: 1,
        max_backoff_ms: 10,
        backoff_multiplier: 2.0,
    };
    let result = execute_with_retry(&config, || Ok::<_, ApiError>(42));
    assert_eq!(result.unwrap(), 42);
}

#[test]
fn retry_fails_then_succeeds() {
    let call_count = AtomicU32::new(0);
    let config = RetryConfig {
        max_retries: 3,
        initial_backoff_ms: 1,
        max_backoff_ms: 10,
        backoff_multiplier: 2.0,
    };

    let result = execute_with_retry(&config, || {
        let count = call_count.fetch_add(1, Ordering::SeqCst);
        if count < 2 {
            Err(ApiError::EmbeddingApiUnavailable("temporary".into()))
        } else {
            Ok(99)
        }
    });

    assert_eq!(result.unwrap(), 99);
    assert_eq!(call_count.load(Ordering::SeqCst), 3);
}

#[test]
fn retry_exhausts_all_retries() {
    let call_count = AtomicU32::new(0);
    let config = RetryConfig {
        max_retries: 2,
        initial_backoff_ms: 1,
        max_backoff_ms: 10,
        backoff_multiplier: 2.0,
    };

    let result: Result<i32, ApiError> = execute_with_retry(&config, || {
        call_count.fetch_add(1, Ordering::SeqCst);
        Err(ApiError::LlmApiUnavailable("always fails".into()))
    });

    assert!(result.is_err());
    let error = result.unwrap_err();
    assert_eq!(
        error.to_string(),
        "[2002] llm api unavailable: always fails"
    );
    assert_eq!(call_count.load(Ordering::SeqCst), 3);
}

#[test]
fn retry_does_not_retry_invalid_api_key() {
    let call_count = AtomicU32::new(0);
    let config = RetryConfig {
        max_retries: 3,
        initial_backoff_ms: 1,
        max_backoff_ms: 10,
        backoff_multiplier: 2.0,
    };

    let result: Result<i32, ApiError> = execute_with_retry(&config, || {
        call_count.fetch_add(1, Ordering::SeqCst);
        Err(ApiError::InvalidApiKey("bad key".into()))
    });

    assert!(result.is_err());
    assert_eq!(call_count.load(Ordering::SeqCst), 1);
}

// ── Constructors: empty key ─────────────────────────────────────────────

#[test]
fn voyage_new_with_empty_key_returns_invalid_api_key() {
    let Err(error) = VoyageEmbeddingProvider::new(String::new()) else {
        panic!("expected InvalidApiKey error");
    };
    assert_eq!(error.to_string(), "[2004] invalid api key: empty api key");
}

#[test]
fn openai_new_with_empty_key_returns_invalid_api_key() {
    let Err(error) = OpenAITextGenerator::new(String::new()) else {
        panic!("expected InvalidApiKey error");
    };
    assert_eq!(error.to_string(), "[2004] invalid api key: empty api key");
}

// ── Mock providers ──────────────────────────────────────────────────────

#[test]
fn mock_embedding_provider_returns_correct_vec() {
    let provider = MockEmbeddingProvider {
        embedding: vec![0.1, 0.2, 0.3],
        model: "test-model".into(),
    };

    let result = provider.embed("test input").unwrap();
    assert_eq!(result, vec![0.1, 0.2, 0.3]);
    assert_eq!(provider.dimension(), 3);
    assert_eq!(provider.model_name(), "test-model");
}

#[test]
fn mock_text_generator_returns_correct_string() {
    let generator = MockTextGenerator {
        response: "generated text".into(),
        model: "test-gen".into(),
    };

    let result = generator.generate("prompt").unwrap();
    assert_eq!(result, "generated text");
    assert_eq!(generator.model_name(), "test-gen");
}

// ── Connection refused with specific error variant ──────────────────────

#[test]
fn voyage_connection_refused_returns_embedding_api_unavailable() {
    let provider = VoyageEmbeddingProvider::with_config(
        "test-key".into(),
        "voyage-code-3".into(),
        1024,
        RetryConfig {
            max_retries: 0,
            initial_backoff_ms: 1,
            max_backoff_ms: 10,
            backoff_multiplier: 2.0,
        },
        "http://127.0.0.1:1".into(),
    )
    .unwrap();

    let result = provider.embed("test");
    assert!(matches!(
        result.unwrap_err(),
        ApiError::EmbeddingApiUnavailable(_)
    ));
}

#[test]
fn openai_connection_refused_returns_llm_api_unavailable() {
    let generator = OpenAITextGenerator::with_config(
        "test-key".into(),
        "gpt-4o-mini".into(),
        RetryConfig {
            max_retries: 0,
            initial_backoff_ms: 1,
            max_backoff_ms: 10,
            backoff_multiplier: 2.0,
        },
        "http://127.0.0.1:1".into(),
    )
    .unwrap();

    let result = generator.generate("test");
    assert!(matches!(
        result.unwrap_err(),
        ApiError::LlmApiUnavailable(_)
    ));
}

#[test]
fn retryable_errors_are_correctly_identified() {
    assert!(ApiError::EmbeddingApiUnavailable("x".into()).is_retryable());
    assert!(ApiError::LlmApiUnavailable("x".into()).is_retryable());
    assert!(ApiError::RateLimitExceeded("x".into()).is_retryable());
    assert!(!ApiError::InvalidApiKey("x".into()).is_retryable());
    assert!(!ApiError::HyDeGenerationFailed("x".into()).is_retryable());
}

// ── Issue 1: Status code mapping — embedding ───────────────────────────

#[test]
fn map_embedding_error_401_returns_invalid_api_key() {
    let error = map_embedding_error(401, "unauthorized".into());
    assert!(matches!(error, ApiError::InvalidApiKey(_)));
}

#[test]
fn map_embedding_error_429_returns_rate_limit_exceeded() {
    let error = map_embedding_error(429, "slow down".into());
    assert!(matches!(error, ApiError::RateLimitExceeded(_)));
}

#[test]
fn map_embedding_error_500_returns_embedding_api_unavailable() {
    let error = map_embedding_error(500, "internal".into());
    assert!(matches!(error, ApiError::EmbeddingApiUnavailable(_)));
}

#[test]
fn map_embedding_error_502_returns_embedding_api_unavailable() {
    let error = map_embedding_error(502, "bad gateway".into());
    assert!(matches!(error, ApiError::EmbeddingApiUnavailable(_)));
}

#[test]
fn map_embedding_error_other_returns_embedding_api_unavailable() {
    let error = map_embedding_error(418, "teapot".into());
    assert!(matches!(error, ApiError::EmbeddingApiUnavailable(_)));
}

// ── Issue 1: Status code mapping — LLM ─────────────────────────────────

#[test]
fn map_llm_error_401_returns_invalid_api_key() {
    let error = map_llm_error(401, "unauthorized".into());
    assert!(matches!(error, ApiError::InvalidApiKey(_)));
}

#[test]
fn map_llm_error_429_returns_rate_limit_exceeded() {
    let error = map_llm_error(429, "slow down".into());
    assert!(matches!(error, ApiError::RateLimitExceeded(_)));
}

#[test]
fn map_llm_error_500_returns_llm_api_unavailable() {
    let error = map_llm_error(500, "internal".into());
    assert!(matches!(error, ApiError::LlmApiUnavailable(_)));
}

// ── Issue 2: Response parsing — embedding ───────────────────────────────

#[test]
fn parse_embedding_response_valid_json() {
    let body = r#"{"data":[{"embedding":[0.1,0.2,0.3]}]}"#;
    let result = parse_embedding_response(body).unwrap();
    assert_eq!(result.len(), 3);
    assert!((result[0] - 0.1).abs() < f32::EPSILON);
    assert!((result[1] - 0.2).abs() < f32::EPSILON);
    assert!((result[2] - 0.3).abs() < f32::EPSILON);
}

#[test]
fn parse_embedding_response_invalid_json() {
    let result = parse_embedding_response("not json");
    assert!(matches!(
        result.unwrap_err(),
        ApiError::EmbeddingApiUnavailable(_)
    ));
}

#[test]
fn parse_embedding_response_missing_data() {
    let body = r#"{"result":"ok"}"#;
    let result = parse_embedding_response(body);
    assert!(matches!(
        result.unwrap_err(),
        ApiError::EmbeddingApiUnavailable(_)
    ));
}

#[test]
fn parse_embedding_response_non_numeric_value() {
    let body = r#"{"data":[{"embedding":["not_a_number"]}]}"#;
    let result = parse_embedding_response(body);
    assert!(matches!(
        result.unwrap_err(),
        ApiError::EmbeddingApiUnavailable(_)
    ));
}

// ── Issue 2: Response parsing — chat ────────────────────────────────────

#[test]
fn parse_chat_response_valid_json() {
    let body = r#"{"choices":[{"message":{"content":"hello world"}}]}"#;
    let result = parse_chat_response(body).unwrap();
    assert_eq!(result, "hello world");
}

#[test]
fn parse_chat_response_invalid_json() {
    let result = parse_chat_response("not json");
    assert!(matches!(
        result.unwrap_err(),
        ApiError::LlmApiUnavailable(_)
    ));
}

#[test]
fn parse_chat_response_missing_content() {
    let body = r#"{"choices":[]}"#;
    let result = parse_chat_response(body);
    assert!(matches!(
        result.unwrap_err(),
        ApiError::LlmApiUnavailable(_)
    ));
}

// ── Issue 3: Empty text edge cases ──────────────────────────────────────

#[test]
fn voyage_embed_empty_text_sends_request() {
    let provider = VoyageEmbeddingProvider::with_config(
        "test-key".into(),
        "voyage-code-3".into(),
        1024,
        RetryConfig {
            max_retries: 0,
            initial_backoff_ms: 1,
            max_backoff_ms: 10,
            backoff_multiplier: 2.0,
        },
        "http://127.0.0.1:1".into(),
    )
    .unwrap();

    let result = provider.embed("");
    assert!(result.is_err());
}

#[test]
fn openai_generate_empty_text_sends_request() {
    let generator = OpenAITextGenerator::with_config(
        "test-key".into(),
        "gpt-4o-mini".into(),
        RetryConfig {
            max_retries: 0,
            initial_backoff_ms: 1,
            max_backoff_ms: 10,
            backoff_multiplier: 2.0,
        },
        "http://127.0.0.1:1".into(),
    )
    .unwrap();

    let result = generator.generate("");
    assert!(result.is_err());
}

// ── Issue 9: compute_backoff tests ──────────────────────────────────────

#[test]
fn compute_backoff_exponential_growth() {
    let config = RetryConfig {
        max_retries: 5,
        initial_backoff_ms: 100,
        max_backoff_ms: 100_000,
        backoff_multiplier: 2.0,
    };
    assert_eq!(compute_backoff(&config, 0), 100);
    assert_eq!(compute_backoff(&config, 1), 200);
    assert_eq!(compute_backoff(&config, 2), 400);
    assert_eq!(compute_backoff(&config, 3), 800);
}

#[test]
fn compute_backoff_caps_at_max() {
    let config = RetryConfig {
        max_retries: 5,
        initial_backoff_ms: 100,
        max_backoff_ms: 500,
        backoff_multiplier: 2.0,
    };
    assert_eq!(compute_backoff(&config, 0), 100);
    assert_eq!(compute_backoff(&config, 3), 500);
    assert_eq!(compute_backoff(&config, 10), 500);
}

#[test]
fn compute_backoff_handles_overflow_to_infinity() {
    let config = RetryConfig {
        max_retries: 5,
        initial_backoff_ms: u64::MAX,
        max_backoff_ms: 10_000,
        backoff_multiplier: f64::MAX,
    };
    assert_eq!(compute_backoff(&config, 1), 10_000);
}

// ── Issue 10: Constructor defaults ──────────────────────────────────────

#[test]
fn voyage_new_sets_default_model_and_dimension() {
    let provider = VoyageEmbeddingProvider::new("test-key".into()).unwrap();
    assert_eq!(provider.model_name(), DEFAULT_VOYAGE_MODEL);
    assert_eq!(provider.dimension(), DEFAULT_VOYAGE_DIMENSION);
}

#[test]
fn openai_new_sets_default_model() {
    let generator = OpenAITextGenerator::new("test-key".into()).unwrap();
    assert_eq!(generator.model_name(), DEFAULT_OPENAI_MODEL);
}
