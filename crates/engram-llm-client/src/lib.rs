pub mod error;
#[cfg(feature = "local")]
pub mod local;
pub mod ollama;
pub mod ollama_text;
pub mod openai;
pub mod provider;
pub mod retry;
pub mod voyage;

pub use error::ApiError;
#[cfg(feature = "local")]
pub use local::LocalTextGenerator;
pub use ollama::OllamaEmbeddingProvider;
pub use ollama_text::OllamaTextGenerator;
pub use openai::OpenAITextGenerator;
pub use provider::{EmbeddingProvider, TextGenerator};
pub use retry::{RetryConfig, compute_backoff, execute_with_retry};
pub use voyage::VoyageEmbeddingProvider;
