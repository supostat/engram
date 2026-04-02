use std::collections::HashMap;
use std::io::{Read, Write};

use engram_embeddings::ThreeFieldEmbedding;
use engram_hnsw::{HnswError, HnswGraph, HnswParams};

use crate::error::CoreError;

pub struct IndexSet {
    context_index: HnswGraph,
    action_index: HnswGraph,
    result_index: HnswGraph,
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
        })
    }

    pub fn insert(
        &mut self,
        id: u64,
        embedding: &ThreeFieldEmbedding,
        rng_value: f64,
    ) -> Result<(), HnswError> {
        self.context_index
            .insert(id, embedding.context.clone(), rng_value)?;
        self.action_index
            .insert(id, embedding.action.clone(), rng_value)?;
        self.result_index
            .insert(id, embedding.result.clone(), rng_value)?;
        Ok(())
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
