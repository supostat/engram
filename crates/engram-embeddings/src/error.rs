use std::fmt;

use engram_llm_client::ApiError;

#[derive(Debug)]
pub enum EmbeddingError {
    ProviderError(ApiError),
}

impl fmt::Display for EmbeddingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ProviderError(error) => {
                write!(formatter, "embedding provider error: {error}")
            }
        }
    }
}

impl std::error::Error for EmbeddingError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ProviderError(error) => Some(error),
        }
    }
}

impl From<ApiError> for EmbeddingError {
    fn from(error: ApiError) -> Self {
        Self::ProviderError(error)
    }
}
