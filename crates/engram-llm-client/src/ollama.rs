use std::time::Duration;

use crate::error::{ApiError, map_http_status_to_error};
use crate::provider::EmbeddingProvider;
use crate::retry::{RetryConfig, execute_with_retry};

pub const DEFAULT_OLLAMA_MODEL: &str = "qwen3-embedding:0.6b";
const HTTP_REQUEST_TIMEOUT: Duration = Duration::from_secs(10);
const QWEN3_QUERY_INSTRUCTION: &str =
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:";

pub struct OllamaEmbeddingProvider {
    client: reqwest::blocking::Client,
    retry_config: RetryConfig,
    model: String,
    model_name: String,
    dimension: usize,
    host: String,
}

impl OllamaEmbeddingProvider {
    pub fn new(host: String, model: String, dimension: usize) -> Result<Self, ApiError> {
        Self::with_config(host, model, dimension, RetryConfig::default())
    }

    pub fn with_config(
        host: String,
        model: String,
        dimension: usize,
        retry_config: RetryConfig,
    ) -> Result<Self, ApiError> {
        if host.is_empty() {
            return Err(ApiError::EmbeddingApiUnavailable(
                "empty ollama host".into(),
            ));
        }
        let client = reqwest::blocking::Client::builder()
            .timeout(HTTP_REQUEST_TIMEOUT)
            .build()
            .map_err(|error| ApiError::EmbeddingApiUnavailable(error.to_string()))?;
        let model_name = format!("ollama:{model}");
        instrumentation::record_construction();
        Ok(Self {
            client,
            retry_config,
            model,
            model_name,
            dimension,
            host,
        })
    }

    fn build_request_body(&self, text: &str, input_type: Option<&str>) -> serde_json::Value {
        serde_json::json!({
            "model": self.model,
            "input": apply_qwen3_instruction(text, input_type),
        })
    }
}

fn apply_qwen3_instruction(text: &str, input_type: Option<&str>) -> String {
    match input_type {
        Some("query") => format!("{QWEN3_QUERY_INSTRUCTION} {text}"),
        _ => text.to_string(),
    }
}

pub fn map_embedding_error(status_code: u16, message: String) -> ApiError {
    map_http_status_to_error(status_code, message, ApiError::EmbeddingApiUnavailable)
}

/// Parses an Ollama `/api/embed` response. The endpoint returns
/// `{"embeddings": [[...]]}` (array of arrays, one vector per input). We send a
/// single input, so the first inner array is the embedding. The deprecated
/// singular `/api/embeddings` endpoint returns a flat `embedding` key — we do
/// not accept it here to avoid silently reading from the wrong shape.
pub fn parse_embedding_response(body: &str) -> Result<Vec<f32>, ApiError> {
    let parsed: serde_json::Value = serde_json::from_str(body)
        .map_err(|error| ApiError::EmbeddingApiUnavailable(error.to_string()))?;

    let embedding = parsed["embeddings"][0].as_array().ok_or_else(|| {
        ApiError::EmbeddingApiUnavailable("missing embeddings in response".into())
    })?;

    embedding
        .iter()
        .map(|value| {
            value.as_f64().map(|number| number as f32).ok_or_else(|| {
                ApiError::EmbeddingApiUnavailable("invalid number in embedding".into())
            })
        })
        .collect()
}

impl EmbeddingProvider for OllamaEmbeddingProvider {
    fn embed(&self, text: &str, input_type: Option<&str>) -> Result<Vec<f32>, ApiError> {
        let url = format!("{}/api/embed", self.host);
        let body = self.build_request_body(text, input_type);

        execute_with_retry(&self.retry_config, || {
            let response = self
                .client
                .post(&url)
                .json(&body)
                .send()
                .map_err(|error| ApiError::EmbeddingApiUnavailable(error.to_string()))?;

            let status = response.status().as_u16();
            let response_body = response
                .text()
                .map_err(|error| ApiError::EmbeddingApiUnavailable(error.to_string()))?;

            if status != 200 {
                return Err(map_embedding_error(status, response_body));
            }

            parse_embedding_response(&response_body)
        })
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }
}

pub mod instrumentation {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    static CLIENT_CONSTRUCTION_TRACKING_ENABLED: AtomicBool = AtomicBool::new(false);
    static CLIENT_CONSTRUCTION_COUNT: AtomicUsize = AtomicUsize::new(0);

    pub fn enable_client_construction_tracking() {
        CLIENT_CONSTRUCTION_TRACKING_ENABLED.store(true, Ordering::Relaxed);
    }

    pub fn disable_client_construction_tracking() {
        CLIENT_CONSTRUCTION_TRACKING_ENABLED.store(false, Ordering::Relaxed);
    }

    pub fn reset_client_construction_count() {
        CLIENT_CONSTRUCTION_COUNT.store(0, Ordering::Relaxed);
    }

    pub fn client_construction_count() -> usize {
        CLIENT_CONSTRUCTION_COUNT.load(Ordering::Relaxed)
    }

    pub(crate) fn record_construction() {
        if CLIENT_CONSTRUCTION_TRACKING_ENABLED.load(Ordering::Relaxed) {
            CLIENT_CONSTRUCTION_COUNT.fetch_add(1, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provider() -> OllamaEmbeddingProvider {
        OllamaEmbeddingProvider::with_config(
            "http://127.0.0.1:1".into(),
            DEFAULT_OLLAMA_MODEL.into(),
            1024,
            RetryConfig::default(),
        )
        .unwrap()
    }

    #[test]
    fn body_sends_raw_text_when_input_type_unset() {
        let provider = provider();
        let body = provider.build_request_body("hello", None);

        assert_eq!(body["input"], serde_json::json!("hello"));
        assert_eq!(body["model"], serde_json::json!(DEFAULT_OLLAMA_MODEL));
    }

    #[test]
    fn body_sends_raw_text_for_document_input_type() {
        let provider = provider();
        let body = provider.build_request_body("hello", Some("document"));

        assert_eq!(body["input"], serde_json::json!("hello"));
        assert_eq!(body["model"], serde_json::json!(DEFAULT_OLLAMA_MODEL));
    }

    #[test]
    fn body_prefixes_query_instruction_for_query_input_type() {
        let provider = provider();
        let body = provider.build_request_body("hello", Some("query"));

        let input = body["input"].as_str().expect("input must be a string");
        assert!(input.starts_with(QWEN3_QUERY_INSTRUCTION));
        assert!(input.ends_with("hello"));
    }

    #[test]
    fn apply_qwen3_instruction_leaves_document_and_none_verbatim() {
        assert_eq!(apply_qwen3_instruction("text", None), "text");
        assert_eq!(apply_qwen3_instruction("text", Some("document")), "text");
    }

    #[test]
    fn apply_qwen3_instruction_prepends_query_prompt() {
        let prefixed = apply_qwen3_instruction("text", Some("query"));
        assert_eq!(prefixed, format!("{QWEN3_QUERY_INSTRUCTION} text"));
    }

    #[test]
    fn model_name_is_prefixed_with_provider() {
        let provider = provider();
        assert_eq!(provider.model_name(), "ollama:qwen3-embedding:0.6b");
    }

    #[test]
    fn with_config_empty_host_returns_embedding_api_unavailable() {
        let Err(error) = OllamaEmbeddingProvider::with_config(
            String::new(),
            DEFAULT_OLLAMA_MODEL.into(),
            1024,
            RetryConfig::default(),
        ) else {
            panic!("expected EmbeddingApiUnavailable error");
        };
        assert!(matches!(error, ApiError::EmbeddingApiUnavailable(_)));
    }
}
