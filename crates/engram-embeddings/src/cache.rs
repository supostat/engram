use std::collections::HashMap;
use std::sync::RwLock;

/// Cache keyed by (text, input_type). The pair is required because Voyage and
/// similar providers return different embeddings for the same text under
/// different `input_type` hints — collapsing them would silently return a
/// document vector for a query and vice versa.
type CacheKey = (String, Option<String>);

pub struct EmbeddingCache {
    entries: RwLock<HashMap<CacheKey, Vec<f32>>>,
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

    pub fn get(&self, text: &str, input_type: Option<&str>) -> Option<Vec<f32>> {
        let key = make_key(text, input_type);
        self.entries.read().unwrap().get(&key).cloned()
    }

    pub fn insert(&self, text: &str, input_type: Option<&str>, embedding: Vec<f32>) {
        let key = make_key(text, input_type);
        self.entries.write().unwrap().insert(key, embedding);
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

fn make_key(text: &str, input_type: Option<&str>) -> CacheKey {
    (text.to_owned(), input_type.map(str::to_owned))
}
