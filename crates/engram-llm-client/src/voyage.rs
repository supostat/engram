use crate::error::{ApiError, map_http_status_to_error};
use crate::provider::EmbeddingProvider;
use crate::retry::{RetryConfig, execute_with_retry};

pub const DEFAULT_VOYAGE_MODEL: &str = "voyage-code-3";
pub const DEFAULT_VOYAGE_DIMENSION: usize = 1024;

pub struct VoyageEmbeddingProvider {
    api_key: String,
    client: reqwest::blocking::Client,
    retry_config: RetryConfig,
    model: String,
    dimension: usize,
    base_url: String,
}

impl VoyageEmbeddingProvider {
    pub fn new(api_key: String) -> Result<Self, ApiError> {
        Self::with_config(
            api_key,
            DEFAULT_VOYAGE_MODEL.into(),
            DEFAULT_VOYAGE_DIMENSION,
            RetryConfig::default(),
            "https://api.voyageai.com".into(),
        )
    }

    pub fn with_config(
        api_key: String,
        model: String,
        dimension: usize,
        retry_config: RetryConfig,
        base_url: String,
    ) -> Result<Self, ApiError> {
        if api_key.is_empty() {
            return Err(ApiError::InvalidApiKey("empty api key".into()));
        }
        let client = reqwest::blocking::Client::new();
        Ok(Self {
            api_key,
            client,
            retry_config,
            model,
            dimension,
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

impl EmbeddingProvider for VoyageEmbeddingProvider {
    fn embed(&self, text: &str) -> Result<Vec<f32>, ApiError> {
        let url = format!("{}/v1/embeddings", self.base_url);
        let body = serde_json::json!({
            "input": [text],
            "model": self.model,
        });

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
