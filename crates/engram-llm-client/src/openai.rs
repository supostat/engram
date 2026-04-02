use crate::error::{ApiError, map_http_status_to_error};
use crate::provider::TextGenerator;
use crate::retry::{RetryConfig, execute_with_retry};

pub const DEFAULT_OPENAI_MODEL: &str = "gpt-4o-mini";

pub struct OpenAITextGenerator {
    api_key: String,
    client: reqwest::blocking::Client,
    retry_config: RetryConfig,
    model: String,
    base_url: String,
}

impl OpenAITextGenerator {
    pub fn new(api_key: String) -> Result<Self, ApiError> {
        Self::with_config(
            api_key,
            DEFAULT_OPENAI_MODEL.into(),
            RetryConfig::default(),
            "https://api.openai.com".into(),
        )
    }

    pub fn with_config(
        api_key: String,
        model: String,
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
            base_url,
        })
    }
}

pub fn map_llm_error(status_code: u16, message: String) -> ApiError {
    map_http_status_to_error(status_code, message, ApiError::LlmApiUnavailable)
}

pub fn parse_chat_response(body: &str) -> Result<String, ApiError> {
    let parsed: serde_json::Value = serde_json::from_str(body)
        .map_err(|error| ApiError::LlmApiUnavailable(error.to_string()))?;

    parsed["choices"][0]["message"]["content"]
        .as_str()
        .map(String::from)
        .ok_or_else(|| ApiError::LlmApiUnavailable("missing content in response".into()))
}

impl TextGenerator for OpenAITextGenerator {
    fn generate(&self, prompt: &str) -> Result<String, ApiError> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let body = serde_json::json!({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        });

        execute_with_retry(&self.retry_config, || {
            let response = self
                .client
                .post(&url)
                .bearer_auth(&self.api_key)
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

            parse_chat_response(&response_body)
        })
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}
