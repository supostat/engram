use std::collections::HashMap;
use std::io::{Read, Write};

use engram_embeddings::ThreeFieldEmbedding;
use engram_hnsw::{HnswError, HnswGraph, HnswParams};

use crate::error::CoreError;

const DEDUP_SEARCH_K: usize = 16;

pub struct IndexSet {
    context_index: HnswGraph,
    action_index: HnswGraph,
    result_index: HnswGraph,
    id_map: HashMap<u64, String>,
}

impl IndexSet {
    pub fn new(
        build_params: impl Fn() -> Result<HnswParams, CoreError>,
    ) -> Result<Self, CoreError> {
        let context_index = HnswGraph::new(build_params()?);
        let action_index = HnswGraph::new(build_params()?);
        let result_index = HnswGraph::new(build_params()?);
        Ok(Self {
            context_index,
            action_index,
            result_index,
            id_map: HashMap::new(),
        })
    }

    pub fn insert_atomic(
        &mut self,
        id: u64,
        memory_id: &str,
        embedding: &ThreeFieldEmbedding,
        rng_value: f64,
    ) -> Result<(), CoreError> {
        if let Some(existing_id) = self.id_map.get(&id) {
            if existing_id == memory_id {
                return Ok(());
            }
            return Err(CoreError::IndexHashCollision {
                hash: id,
                existing_id: existing_id.clone(),
                conflicting_id: memory_id.to_string(),
            });
        }

        self.context_index
            .insert(id, embedding.context.clone(), rng_value)?;
        if let Err(error) = self
            .action_index
            .insert(id, embedding.action.clone(), rng_value)
        {
            let _ = self.context_index.delete(id);
            return Err(CoreError::from(error));
        }
        if let Err(error) = self
            .result_index
            .insert(id, embedding.result.clone(), rng_value)
        {
            let _ = self.context_index.delete(id);
            let _ = self.action_index.delete(id);
            return Err(CoreError::from(error));
        }
        self.id_map.insert(id, memory_id.to_string());
        Ok(())
    }

    pub fn resolve_node_id(&self, node_id: u64) -> Option<&str> {
        self.id_map.get(&node_id).map(String::as_str)
    }

    pub fn rebuild_id_map(&mut self, entries: impl Iterator<Item = (u64, String)>) {
        self.id_map.clear();
        for (hash, memory_id) in entries {
            self.id_map.insert(hash, memory_id);
        }
    }

    pub fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<(u64, f32)>, HnswError> {
        let mut scores: HashMap<u64, f32> = HashMap::new();
        merge_scores(&mut scores, &self.context_index.search(query, top_k)?);
        merge_scores(&mut scores, &self.action_index.search(query, top_k)?);
        merge_scores(&mut scores, &self.result_index.search(query, top_k)?);
        let mut ranked: Vec<(u64, f32)> = scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        ranked.truncate(top_k);
        Ok(ranked)
    }

    /// Find the best existing near-duplicate of `embedding` under the ALL-THREE policy:
    /// a candidate qualifies only when its context, action AND result similarities each
    /// exceed `threshold`. Returns the qualifying node with the highest minimum-field
    /// similarity, or None. Uses only per-graph HNSW search (top_k=K) intersected across
    /// the three fields — with top_k=1 the three graphs return different top nodes and the
    /// all-three condition would almost never hold.
    pub fn find_duplicate(
        &self,
        embedding: &ThreeFieldEmbedding,
        threshold: f32,
    ) -> Result<Option<(u64, f32)>, HnswError> {
        let context_hits = self
            .context_index
            .search(&embedding.context, DEDUP_SEARCH_K)?;
        let action_scores: HashMap<u64, f32> = self
            .action_index
            .search(&embedding.action, DEDUP_SEARCH_K)?
            .into_iter()
            .collect();
        let result_scores: HashMap<u64, f32> = self
            .result_index
            .search(&embedding.result, DEDUP_SEARCH_K)?
            .into_iter()
            .collect();
        let mut best: Option<(u64, f32)> = None;
        for (node, context_score) in context_hits {
            let (Some(&action_score), Some(&result_score)) =
                (action_scores.get(&node), result_scores.get(&node))
            else {
                continue;
            };
            if context_score > threshold && action_score > threshold && result_score > threshold {
                let min_similarity = context_score.min(action_score).min(result_score);
                if best.is_none_or(|(_, best_min)| min_similarity > best_min) {
                    best = Some((node, min_similarity));
                }
            }
        }
        Ok(best)
    }

    pub fn delete(&mut self, id: u64) -> Result<(), HnswError> {
        self.context_index.delete(id)?;
        self.action_index.delete(id)?;
        self.result_index.delete(id)?;
        self.id_map.remove(&id);
        Ok(())
    }

    pub fn contains(&self, id: u64) -> bool {
        self.context_index.contains(id)
    }

    pub fn len(&self) -> usize {
        self.context_index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.context_index.is_empty()
    }

    pub fn serialize<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        self.context_index.serialize(writer)?;
        self.action_index.serialize(writer)?;
        self.result_index.serialize(writer)?;
        Ok(())
    }

    pub fn deserialize<R: Read>(reader: &mut R) -> Result<Self, HnswError> {
        let context_index = HnswGraph::deserialize(reader)?;
        let action_index = HnswGraph::deserialize(reader)?;
        let result_index = HnswGraph::deserialize(reader)?;
        Ok(Self {
            context_index,
            action_index,
            result_index,
            id_map: HashMap::new(),
        })
    }
}

fn merge_scores(scores: &mut HashMap<u64, f32>, results: &[(u64, f32)]) {
    for &(id, similarity) in results {
        scores
            .entry(id)
            .and_modify(|existing| *existing = existing.max(similarity))
            .or_insert(similarity);
    }
}

pub mod instrumentation {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    static READER_TRACKING_ENABLED: AtomicBool = AtomicBool::new(false);
    static CURRENT_READERS: AtomicUsize = AtomicUsize::new(0);
    static MAX_READERS: AtomicUsize = AtomicUsize::new(0);

    pub fn enable_reader_tracking() {
        READER_TRACKING_ENABLED.store(true, Ordering::Relaxed);
    }

    pub fn disable_reader_tracking() {
        READER_TRACKING_ENABLED.store(false, Ordering::Relaxed);
    }

    pub fn reset_reader_counters() {
        CURRENT_READERS.store(0, Ordering::Relaxed);
        MAX_READERS.store(0, Ordering::Relaxed);
    }

    pub fn concurrent_readers_max() -> usize {
        MAX_READERS.load(Ordering::Relaxed)
    }

    pub struct ReaderTracker;

    impl ReaderTracker {
        pub fn new() -> Self {
            if READER_TRACKING_ENABLED.load(Ordering::Relaxed) {
                let current = CURRENT_READERS.fetch_add(1, Ordering::AcqRel) + 1;
                MAX_READERS.fetch_max(current, Ordering::AcqRel);
            }
            Self
        }
    }

    impl Drop for ReaderTracker {
        fn drop(&mut self) {
            if READER_TRACKING_ENABLED.load(Ordering::Relaxed) {
                CURRENT_READERS.fetch_sub(1, Ordering::AcqRel);
            }
        }
    }

    impl Default for ReaderTracker {
        fn default() -> Self {
            Self::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIMENSION: usize = 8;
    const THRESHOLD: f32 = 0.95;

    fn new_index_set() -> IndexSet {
        IndexSet::new(|| HnswParams::new(DIMENSION).map_err(CoreError::from))
            .expect("index set construction succeeds")
    }

    /// Builds a one-hot unit vector with its nonzero coordinate at `axis`.
    /// Two such vectors are identical (cosine 1.0) when the axes match and
    /// orthogonal (cosine 0.0) when the axes differ.
    fn axis_vector(axis: usize) -> Vec<f32> {
        let mut vector = vec![0.0_f32; DIMENSION];
        vector[axis % DIMENSION] = 1.0;
        vector
    }

    fn embedding(
        context_axis: usize,
        action_axis: usize,
        result_axis: usize,
    ) -> ThreeFieldEmbedding {
        ThreeFieldEmbedding {
            context: axis_vector(context_axis),
            action: axis_vector(action_axis),
            result: axis_vector(result_axis),
        }
    }

    #[test]
    fn identical_embedding_is_a_duplicate() {
        let mut index_set = new_index_set();
        let stored = embedding(0, 1, 2);
        index_set
            .insert_atomic(7, "memory-7", &stored, 0.5)
            .expect("insert succeeds");

        let found = index_set
            .find_duplicate(&stored, THRESHOLD)
            .expect("search succeeds");

        let (node, min_similarity) = found.expect("identical embedding qualifies as duplicate");
        assert_eq!(node, 7);
        assert_eq!(index_set.resolve_node_id(node), Some("memory-7"));
        // Identical vectors yield cosine 1.0 on every field, so the all-three
        // minimum is 1.0.
        assert!((min_similarity - 1.0).abs() < 1e-4);
    }

    #[test]
    fn only_context_matching_is_not_a_duplicate() {
        let mut index_set = new_index_set();
        let stored = embedding(0, 1, 2);
        index_set
            .insert_atomic(7, "memory-7", &stored, 0.5)
            .expect("insert succeeds");

        // Context shares axis 0 (cosine 1.0) while action/result use disjoint
        // axes (cosine 0.0). Under all-three this fails; a max-across-one policy
        // would wrongly accept the perfect context match.
        let query = embedding(0, 4, 5);
        let found = index_set
            .find_duplicate(&query, THRESHOLD)
            .expect("search succeeds");

        assert_eq!(found, None);
    }

    #[test]
    fn empty_index_has_no_duplicate() {
        let index_set = new_index_set();
        let query = embedding(0, 1, 2);

        let found = index_set
            .find_duplicate(&query, THRESHOLD)
            .expect("search succeeds");

        assert_eq!(found, None);
    }

    #[test]
    fn embedding_below_threshold_is_not_a_duplicate() {
        let mut index_set = new_index_set();
        let stored = embedding(0, 1, 2);
        index_set
            .insert_atomic(7, "memory-7", &stored, 0.5)
            .expect("insert succeeds");

        // Every field uses a disjoint axis, so all three cosine similarities are
        // 0.0 — far below the 0.95 threshold.
        let query = embedding(3, 4, 5);
        let found = index_set
            .find_duplicate(&query, THRESHOLD)
            .expect("search succeeds");

        assert_eq!(found, None);
    }
}
