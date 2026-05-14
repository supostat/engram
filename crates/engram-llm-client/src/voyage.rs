use std::time::Duration;

use crate::error::{ApiError, map_http_status_to_error};
use crate::provider::EmbeddingProvider;
use crate::retry::{RetryConfig, execute_with_retry};

pub const DEFAULT_VOYAGE_MODEL: &str = "voyage-4";
pub const DEFAULT_VOYAGE_DIMENSION: usize = 1024;
const HTTP_REQUEST_TIMEOUT: Duration = Duration::from_secs(10);

pub struct VoyageEmbeddingProvider {
    api_key: String,
    client: reqwest::blocking::Client,
    retry_config: RetryConfig,
    model: String,
    dimension: usize,
    output_dimension: Option<usize>,
    base_url: String,
}

impl VoyageEmbeddingProvider {
    pub fn new(api_key: String) -> Result<Self, ApiError> {
        Self::with_config(
            api_key,
            DEFAULT_VOYAGE_MODEL.into(),
            DEFAULT_VOYAGE_DIMENSION,
            None,
            RetryConfig::default(),
            "https://api.voyageai.com".into(),
        )
    }

    pub fn with_config(
        api_key: String,
        model: String,
        dimension: usize,
        output_dimension: Option<usize>,
        retry_config: RetryConfig,
        base_url: String,
    ) -> Result<Self, ApiError> {
        if api_key.is_empty() {
            return Err(ApiError::InvalidApiKey("empty api key".into()));
        }
        let client = reqwest::blocking::Client::builder()
            .timeout(HTTP_REQUEST_TIMEOUT)
            .build()
            .map_err(|error| ApiError::EmbeddingApiUnavailable(error.to_string()))?;
        instrumentation::record_construction();
        Ok(Self {
            api_key,
            client,
            retry_config,
            model,
            dimension,
            output_dimension,
            base_url,
        })
    }
}

pub fn map_embedding_error(status_code: u16, message: String) -> ApiError {
    map_http_status_to_error(status_code, message, ApiError::EmbeddingApiUnavailable)
}

pub fn parse_embedding_response(body: &str) -> Result<Vec<f32>, ApiError> {
    let parsed: serde_json::Value = serde_json::from_str(body)
        .map_err(|error| ApiError::EmbeddingApiUnavailable(error.to_string()))?;

    let embedding = parsed["data"][0]["embedding"]
        .as_array()
        .ok_or_else(|| ApiError::EmbeddingApiUnavailable("missing embedding in response".into()))?;

    embedding
        .iter()
        .map(|value| {
            value.as_f64().map(|number| number as f32).ok_or_else(|| {
                ApiError::EmbeddingApiUnavailable("invalid number in embedding".into())
            })
        })
        .collect()
}

impl VoyageEmbeddingProvider {
    fn build_request_body(&self, text: &str, input_type: Option<&str>) -> serde_json::Value {
        let mut body = serde_json::json!({
            "input": [text],
            "model": self.model,
        });
        if let Some(out_dim) = self.output_dimension {
            body["output_dimension"] = serde_json::json!(out_dim);
        }
        if let Some(kind) = input_type {
            body["input_type"] = serde_json::json!(kind);
        }
        body
    }
}

impl EmbeddingProvider for VoyageEmbeddingProvider {
    fn embed(&self, text: &str, input_type: Option<&str>) -> Result<Vec<f32>, ApiError> {
        let url = format!("{}/v1/embeddings", self.base_url);
        let body = self.build_request_body(text, input_type);

        execute_with_retry(&self.retry_config, || {
            let response = self
                .client
                .post(&url)
                .bearer_auth(&self.api_key)
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
        &self.model
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

    fn provider_with_output_dimension(output_dimension: Option<usize>) -> VoyageEmbeddingProvider {
        VoyageEmbeddingProvider::with_config(
            "test-key".into(),
            "voyage-4".into(),
            1024,
            output_dimension,
            RetryConfig::default(),
            "http://127.0.0.1:1".into(),
        )
        .unwrap()
    }

    #[test]
    fn body_omits_output_dimension_and_input_type_when_unset() {
        let provider = provider_with_output_dimension(None);
        let body = provider.build_request_body("hello", None);

        assert_eq!(body["input"], serde_json::json!(["hello"]));
        assert_eq!(body["model"], serde_json::json!("voyage-4"));
        assert!(body.get("output_dimension").is_none());
        assert!(body.get("input_type").is_none());
    }

    #[test]
    fn body_includes_output_dimension_when_provider_configured() {
        let provider = provider_with_output_dimension(Some(1024));
        let body = provider.build_request_body("hello", None);

        assert_eq!(body["output_dimension"], serde_json::json!(1024));
    }

    #[test]
    fn body_includes_input_type_when_caller_provides() {
        let provider = provider_with_output_dimension(None);
        let body = provider.build_request_body("hello", Some("query"));

        assert_eq!(body["input_type"], serde_json::json!("query"));
    }

    #[test]
    fn body_combines_output_dimension_and_input_type() {
        let provider = provider_with_output_dimension(Some(512));
        let body = provider.build_request_body("hello", Some("document"));

        assert_eq!(body["output_dimension"], serde_json::json!(512));
        assert_eq!(body["input_type"], serde_json::json!("document"));
        assert_eq!(body["model"], serde_json::json!("voyage-4"));
        assert_eq!(body["input"], serde_json::json!(["hello"]));
    }
}
