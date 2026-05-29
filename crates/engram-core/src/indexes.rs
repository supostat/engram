use std::collections::HashMap;
use std::io::{Read, Write};

use engram_embeddings::ThreeFieldEmbedding;
use engram_hnsw::{HnswError, HnswGraph, HnswParams};

use crate::error::CoreError;

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
