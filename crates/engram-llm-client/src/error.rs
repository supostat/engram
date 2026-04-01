use std::fmt;

#[derive(Debug)]
pub enum ApiError {
    EmbeddingApiUnavailable(String),
    LlmApiUnavailable(String),
    RateLimitExceeded(String),
    InvalidApiKey(String),
    HyDeGenerationFailed(String),
}

impl ApiError {
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::EmbeddingApiUnavailable(_)
                | Self::LlmApiUnavailable(_)
                | Self::RateLimitExceeded(_)
        )
    }
}

impl fmt::Display for ApiError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmbeddingApiUnavailable(message) => {
                write!(formatter, "[2001] embedding api unavailable: {message}")
            }
            Self::LlmApiUnavailable(message) => {
                write!(formatter, "[2002] llm api unavailable: {message}")
            }
            Self::RateLimitExceeded(message) => {
                write!(formatter, "[2003] rate limit exceeded: {message}")
            }
            Self::InvalidApiKey(message) => {
                write!(formatter, "[2004] invalid api key: {message}")
            }
            Self::HyDeGenerationFailed(message) => {
                write!(formatter, "[2005] hyde generation failed: {message}")
            }
        }
    }
}

impl std::error::Error for ApiError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}

pub fn map_http_status_to_error(
    status_code: u16,
    message: String,
    unavailable_variant: fn(String) -> ApiError,
) -> ApiError {
    match status_code {
        401 => ApiError::InvalidApiKey(message),
        429 => ApiError::RateLimitExceeded(message),
        code if code >= 500 => unavailable_variant(message),
        _ => unavailable_variant(message),
    }
}
