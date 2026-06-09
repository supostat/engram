use std::time::Duration;

use crate::error::{ApiError, map_http_status_to_error};
use crate::provider::TextGenerator;
use crate::retry::{RetryConfig, execute_with_retry};

pub const DEFAULT_OLLAMA_LLM_MODEL: &str = "qwen3:4b";

// LLM generation is inherently far slower than embedding: qwen3 produces a
// few dozen tokens per second, so a single thinking+answer response can run
// well past the embedding client's 10s budget. A short timeout would surface
// as a spurious request failure and silently drop callers into the heuristic
// fallback, so the text generator allows a full minute per request.
const HTTP_REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

const THINK_BLOCK_CLOSE: &str = "</think>";

pub struct OllamaTextGenerator {
    client: reqwest::blocking::Client,
    retry_config: RetryConfig,
    model: String,
    model_name: String,
    host: String,
}

impl OllamaTextGenerator {
    pub fn new(host: String, model: String) -> Result<Self, ApiError> {
        Self::with_config(host, model, RetryConfig::localhost())
    }

    pub fn with_config(
        host: String,
        model: String,
        retry_config: RetryConfig,
    ) -> Result<Self, ApiError> {
        if host.is_empty() {
            return Err(ApiError::LlmApiUnavailable("empty ollama host".into()));
        }
        let client = reqwest::blocking::Client::builder()
            .timeout(HTTP_REQUEST_TIMEOUT)
            .build()
            .map_err(|error| ApiError::LlmApiUnavailable(error.to_string()))?;
        let model_name = format!("ollama:{model}");
        instrumentation::record_construction();
        Ok(Self {
            client,
            retry_config,
            model,
            model_name,
            host,
        })
    }

    fn build_request_body(&self, prompt: &str) -> serde_json::Value {
        serde_json::json!({
            "model": self.model,
            "prompt": prompt,
            "stream": false,
        })
    }
}

pub fn map_llm_error(status_code: u16, message: String) -> ApiError {
    map_http_status_to_error(status_code, message, ApiError::LlmApiUnavailable)
}

/// Parses an Ollama `/api/generate` response. With `stream: false` the daemon
/// returns a single JSON object whose `response` field holds the full
/// completion. Reasoning models such as qwen3 prepend a `<think>...</think>`
/// block, which `strip_think_block` removes so judge/HyDE/consolidation see
/// only the final answer.
pub fn parse_generate_response(body: &str) -> Result<String, ApiError> {
    let parsed: serde_json::Value = serde_json::from_str(body)
        .map_err(|error| ApiError::LlmApiUnavailable(error.to_string()))?;

    parsed["response"]
        .as_str()
        .map(strip_think_block)
        .ok_or_else(|| ApiError::LlmApiUnavailable("missing response in response".into()))
}

/// Drops a leading qwen3 reasoning block, keeping only the text after the last
/// `</think>`. With no closing tag the text is passed through trimmed — an
/// unclosed block is treated as ambiguous and the answer is never discarded.
fn strip_think_block(text: &str) -> String {
    match text.rfind(THINK_BLOCK_CLOSE) {
        Some(index) => text[index + THINK_BLOCK_CLOSE.len()..].trim().to_string(),
        None => text.trim().to_string(),
    }
}

impl TextGenerator for OllamaTextGenerator {
    fn generate(&self, prompt: &str) -> Result<String, ApiError> {
        let url = format!("{}/api/generate", self.host);
        let body = self.build_request_body(prompt);

        execute_with_retry(&self.retry_config, || {
            let response = self
                .client
                .post(&url)
                .json(&body)
                .send()
                .map_err(|error| ApiError::LlmApiUnavailable(error.to_string()))?;

            let status = response.status().as_u16();
            let response_body = response
                .text()
                .map_err(|error| ApiError::LlmApiUnavailable(error.to_string()))?;

            if status != 200 {
                return Err(map_llm_error(status, response_body));
            }

            parse_generate_response(&response_body)
        })
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

    fn generator() -> OllamaTextGenerator {
        OllamaTextGenerator::with_config(
            "http://127.0.0.1:1".into(),
            DEFAULT_OLLAMA_LLM_MODEL.into(),
            RetryConfig::default(),
        )
        .unwrap()
    }

    #[test]
    fn build_request_body_sets_model_prompt_and_disables_streaming() {
        let generator = generator();
        let body = generator.build_request_body("explain ownership");

        assert_eq!(body["model"], serde_json::json!(DEFAULT_OLLAMA_LLM_MODEL));
        assert_eq!(body["prompt"], serde_json::json!("explain ownership"));
        assert_eq!(body["stream"], serde_json::json!(false));
    }

    #[test]
    fn model_name_is_prefixed_with_provider() {
        let generator = generator();
        assert_eq!(generator.model_name(), "ollama:qwen3:4b");
    }

    #[test]
    fn new_wires_localhost_retry_config() {
        let generator = OllamaTextGenerator::new(
            "http://localhost:11434".into(),
            DEFAULT_OLLAMA_LLM_MODEL.into(),
        )
        .unwrap();

        assert_eq!(generator.retry_config.max_retries, 2);
        assert_eq!(generator.retry_config.initial_backoff_ms, 50);
        assert_eq!(generator.retry_config.max_backoff_ms, 500);
        assert!((generator.retry_config.backoff_multiplier - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn with_config_empty_host_returns_llm_api_unavailable() {
        let Err(error) = OllamaTextGenerator::with_config(
            String::new(),
            DEFAULT_OLLAMA_LLM_MODEL.into(),
            RetryConfig::default(),
        ) else {
            panic!("expected LlmApiUnavailable error");
        };
        assert!(matches!(error, ApiError::LlmApiUnavailable(_)));
    }

    #[test]
    fn strip_think_block_passes_through_text_without_a_block() {
        assert_eq!(strip_think_block("just the answer"), "just the answer");
    }

    #[test]
    fn strip_think_block_removes_leading_reasoning_block() {
        assert_eq!(
            strip_think_block("<think>weigh options</think>final answer"),
            "final answer"
        );
    }

    #[test]
    fn strip_think_block_keeps_text_after_the_last_close_on_multiple_blocks() {
        assert_eq!(
            strip_think_block("<think>a</think>mid<think>b</think>answer"),
            "answer"
        );
    }

    #[test]
    fn strip_think_block_trims_surrounding_whitespace() {
        assert_eq!(
            strip_think_block("<think>reason</think>\n  answer  \n"),
            "answer"
        );
    }

    #[test]
    fn strip_think_block_yields_empty_when_nothing_follows_the_tag() {
        assert_eq!(strip_think_block("<think>reason</think>   "), "");
    }

    #[test]
    fn strip_think_block_passes_through_unclosed_block() {
        assert_eq!(
            strip_think_block("<think>still thinking"),
            "<think>still thinking"
        );
    }

    #[test]
    fn strip_think_block_trims_plain_text() {
        assert_eq!(strip_think_block("  answer  "), "answer");
    }
}
