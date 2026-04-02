pub mod error;
#[cfg(feature = "local")]
pub mod local;
pub mod openai;
pub mod provider;
pub mod retry;
pub mod voyage;

pub use error::ApiError;
#[cfg(feature = "local")]
pub use local::LocalTextGenerator;
pub use openai::OpenAITextGenerator;
pub use provider::{EmbeddingProvider, TextGenerator};
pub use retry::{compute_backoff, execute_with_retry, RetryConfig};
pub use voyage::VoyageEmbeddingProvider;
