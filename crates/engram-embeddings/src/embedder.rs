use engram_llm_client::provider::{EmbeddingProvider, TextGenerator};

use crate::cache::EmbeddingCache;
use crate::error::EmbeddingError;
use crate::hyde;

pub struct ThreeFieldEmbedding {
    pub context: Vec<f32>,
    pub action: Vec<f32>,
    pub result: Vec<f32>,
}

pub struct Embedder {
    cache: EmbeddingCache,
    hyde_threshold: usize,
}

impl Default for Embedder {
    fn default() -> Self {
        Self::new(0)
    }
}

impl Embedder {
    pub fn new(hyde_threshold: usize) -> Self {
        Self {
            cache: EmbeddingCache::new(),
            hyde_threshold,
        }
    }

    pub fn embed_fields(
        &self,
        context: &str,
        action: &str,
        result: &str,
        provider: &dyn EmbeddingProvider,
        text_generator: Option<&dyn TextGenerator>,
    ) -> Result<ThreeFieldEmbedding, EmbeddingError> {
        let context_text = self.prepare_text(context, text_generator);
        let action_text = self.prepare_text(action, text_generator);
        let result_text = self.prepare_text(result, text_generator);

        let context_embedding = self.get_or_embed(&context_text, provider)?;
        let action_embedding = self.get_or_embed(&action_text, provider)?;
        let result_embedding = self.get_or_embed(&result_text, provider)?;

        Ok(ThreeFieldEmbedding {
            context: context_embedding,
            action: action_embedding,
            result: result_embedding,
        })
    }

    pub fn embed_query(
        &self,
        query: &str,
        provider: &dyn EmbeddingProvider,
        text_generator: Option<&dyn TextGenerator>,
    ) -> Result<Vec<f32>, EmbeddingError> {
        if let Some(cached) = self.cache.get(query) {
            return Ok(cached);
        }
        let prepared = self.prepare_text(query, text_generator);
        let embedding = provider
            .embed(&prepared)
            .map_err(EmbeddingError::ProviderError)?;
        self.cache.insert(query.to_string(), embedding.clone());
        Ok(embedding)
    }

    pub fn cache(&self) -> &EmbeddingCache {
        &self.cache
    }

    pub fn clear_cache(&self) {
        self.cache.clear();
    }

    fn prepare_text(&self, text: &str, text_generator: Option<&dyn TextGenerator>) -> String {
        if hyde::should_use_hyde(text, self.hyde_threshold)
            && let Some(generator) = text_generator
            && let Ok(hypothesis) = hyde::generate_hypothesis(text, generator)
        {
            return hypothesis;
        }
        text.to_string()
    }

    fn get_or_embed(
        &self,
        text: &str,
        provider: &dyn EmbeddingProvider,
    ) -> Result<Vec<f32>, EmbeddingError> {
        if let Some(cached) = self.cache.get(text) {
            return Ok(cached);
        }
        let embedding = provider.embed(text)?;
        self.cache.insert(text.to_string(), embedding.clone());
        Ok(embedding)
    }
}
