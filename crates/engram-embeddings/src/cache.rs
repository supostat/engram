use std::collections::HashMap;
use std::sync::RwLock;

pub struct EmbeddingCache {
    entries: RwLock<HashMap<String, Vec<f32>>>,
}

impl Default for EmbeddingCache {
    fn default() -> Self {
        Self::new()
    }
}

impl EmbeddingCache {
    pub fn new() -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
        }
    }

    pub fn get(&self, text: &str) -> Option<Vec<f32>> {
        self.entries.read().unwrap().get(text).cloned()
    }

    pub fn insert(&self, text: String, embedding: Vec<f32>) {
        self.entries.write().unwrap().insert(text, embedding);
    }

    pub fn len(&self) -> usize {
        self.entries.read().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.read().unwrap().is_empty()
    }

    pub fn clear(&self) {
        self.entries.write().unwrap().clear();
    }
}
