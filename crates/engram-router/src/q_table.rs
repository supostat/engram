use std::collections::HashMap;

const MAX_ENTRIES: usize = 100_000;

pub struct QTable {
    values: HashMap<(String, String), f32>,
    counts: HashMap<(String, String), u32>,
}

impl Default for QTable {
    fn default() -> Self {
        Self::new()
    }
}

impl QTable {
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
            counts: HashMap::new(),
        }
    }

    pub fn get(&self, state: &str, action: &str) -> f32 {
        let key = (state.to_string(), action.to_string());
        self.values.get(&key).copied().unwrap_or(0.0)
    }

    /// Q(s,a) = Q(s,a) + alpha * (reward - Q(s,a))
    pub fn update(&mut self, state: &str, action: &str, reward: f32, alpha: f32) {
        let key = (state.to_string(), action.to_string());
        let is_new = !self.values.contains_key(&key);
        if is_new && self.values.len() >= MAX_ENTRIES {
            return;
        }
        let current = self.values.get(&key).copied().unwrap_or(0.0);
        let updated = current + alpha * (reward - current);
        self.values.insert(key.clone(), updated);
        *self.counts.entry(key).or_insert(0) += 1;
    }

    pub fn update_count(&self, state: &str, action: &str) -> u32 {
        let key = (state.to_string(), action.to_string());
        self.counts.get(&key).copied().unwrap_or(0)
    }

    pub fn actions_for_state(&self, state: &str) -> Vec<(String, f32)> {
        self.values
            .iter()
            .filter(|((s, _), _)| s == state)
            .map(|((_, action), value)| (action.clone(), *value))
            .collect()
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}
