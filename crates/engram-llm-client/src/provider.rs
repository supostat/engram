use crate::error::ApiError;

pub trait EmbeddingProvider: Send + Sync {
    /// Embed `text` into a vector.
    ///
    /// `input_type` is a Voyage-style task hint (`"document"` for ingestion,
    /// `"query"` for retrieval). Providers that do not differentiate (e.g.
    /// deterministic test fixtures) ignore it. The same text with different
    /// `input_type` may produce different embeddings — callers must treat
    /// the pair `(text, input_type)` as the cache key.
    fn embed(&self, text: &str, input_type: Option<&str>) -> Result<Vec<f32>, ApiError>;
    fn dimension(&self) -> usize;
    fn model_name(&self) -> &str;
}

pub trait TextGenerator: Send + Sync {
    fn generate(&self, prompt: &str) -> Result<String, ApiError>;
    fn model_name(&self) -> &str;
}
