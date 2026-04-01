use std::collections::HashMap;

pub struct EmbeddingCache {
    entries: HashMap<String, Vec<f32>>,
}

impl Default for EmbeddingCache {
    fn default() -> Self {
        Self::new()
    }
}

impl EmbeddingCache {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    pub fn get(&self, text: &str) -> Option<&Vec<f32>> {
        self.entries.get(text)
    }

    pub fn insert(&mut self, text: String, embedding: Vec<f32>) {
        self.entries.insert(text, embedding);
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }
}
