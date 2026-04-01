use crate::error::ApiError;

pub trait EmbeddingProvider: Send + Sync {
    fn embed(&self, text: &str) -> Result<Vec<f32>, ApiError>;
    fn dimension(&self) -> usize;
    fn model_name(&self) -> &str;
}

pub trait TextGenerator: Send + Sync {
    fn generate(&self, prompt: &str) -> Result<String, ApiError>;
    fn model_name(&self) -> &str;
}
