use crate::error::HnswError;
use crate::graph::HnswGraph;
use crate::node::Node;
use crate::search::{ScoredNode, search_layer, select_neighbors};
use crate::similarity::cosine_similarity;

impl HnswGraph {
    /// Insert a vector with the given ID.
    /// `rng_value` must be in (0.0, 1.0) — caller provides randomness.
    pub fn insert(&mut self, id: u64, vector: Vec<f32>, rng_value: f64) -> Result<(), HnswError> {
        self.validate_insert(id, &vector)?;
        let node_level = self.random_level(rng_value);
        let node = Node::new(id, vector, node_level);

        if self.is_empty() {
            return self.insert_first_node(node);
        }

        let entry_point_id = self.entry_point().expect("non-empty graph has entry point");
        let mut current_entry = self.score_node(entry_point_id, &node.vector)?;
        current_entry = self.descend_to_level(current_entry, &node.vector, node_level);
        self.connect_at_layers(&node, current_entry, node_level);
        self.promote_entry_point_if_higher(id, node_level);
        Ok(())
    }

    fn validate_insert(&self, id: u64, vector: &[f32]) -> Result<(), HnswError> {
        if vector.len() != self.dimension() {
            return Err(HnswError::DimensionMismatch {
                expected: self.dimension(),
                got: vector.len(),
            });
        }
        if self.contains(id) {
            return Err(HnswError::DuplicateNode(id));
        }
        Ok(())
    }

    fn insert_first_node(&mut self, node: Node) -> Result<(), HnswError> {
        let id = node.id;
        let level = node.level;
        self.nodes_mut().insert(id, node);
        self.set_entry_point(Some(id));
        self.set_max_level(level);
        Ok(())
    }

    fn score_node(&self, node_id: u64, query: &[f32]) -> Result<ScoredNode, HnswError> {
        let node = self
            .nodes()
            .get(&node_id)
            .ok_or(HnswError::NodeNotFound(node_id))?;
        let similarity = cosine_similarity(query, &node.vector)?;
        Ok(ScoredNode {
            id: node_id,
            similarity,
        })
    }

    fn descend_to_level(
        &self,
        mut current: ScoredNode,
        query: &[f32],
        target_level: usize,
    ) -> ScoredNode {
        let top = self.max_level();
        let stop = if target_level < top {
            target_level + 1
        } else {
            top + 1
        };
        for layer in (stop..=top).rev() {
            let results = search_layer(self.nodes(), query, &[current.clone()], 1, layer);
            if let Some(best) = results.into_iter().next() {
                current = best;
            }
        }
        current
    }

    fn connect_at_layers(&mut self, node: &Node, entry: ScoredNode, node_level: usize) {
        let query = node.vector.clone();
        let node_id = node.id;
        let ef_construction = self.params().ef_construction;
        let max_level_to_connect = node_level.min(self.max_level());

        self.nodes_mut().insert(node_id, node.clone());

        let mut current_entries = vec![entry];

        for layer in (0..=max_level_to_connect).rev() {
            let max_conn = self.max_connections_for_layer(layer);
            let results = search_layer(
                self.nodes(),
                &query,
                &current_entries,
                ef_construction,
                layer,
            );
            let selected = select_neighbors(&results, max_conn);

            self.set_neighbors_at_layer(node_id, layer, &selected);
            self.add_bidirectional_connections(node_id, layer, &selected, max_conn);

            current_entries = if results.is_empty() {
                current_entries
            } else {
                results
            };
        }
    }

    fn set_neighbors_at_layer(&mut self, node_id: u64, layer: usize, selected: &[ScoredNode]) {
        let neighbor_ids: Vec<u64> = selected.iter().map(|s| s.id).collect();
        if let Some(node) = self.nodes_mut().get_mut(&node_id)
            && layer < node.neighbors.len()
        {
            node.neighbors[layer] = neighbor_ids;
        }
    }

    fn add_bidirectional_connections(
        &mut self,
        node_id: u64,
        layer: usize,
        selected: &[ScoredNode],
        max_connections: usize,
    ) {
        let neighbor_ids: Vec<u64> = selected.iter().map(|s| s.id).collect();
        for &neighbor_id in &neighbor_ids {
            self.add_neighbor_and_trim(neighbor_id, node_id, layer, max_connections);
        }
    }

    fn add_neighbor_and_trim(
        &mut self,
        node_id: u64,
        new_neighbor: u64,
        layer: usize,
        max_connections: usize,
    ) {
        let Some(node) = self.nodes_mut().get_mut(&node_id) else {
            return;
        };
        if layer >= node.neighbors.len() {
            return;
        }
        if !node.neighbors[layer].contains(&new_neighbor) {
            node.neighbors[layer].push(new_neighbor);
        }
        if node.neighbors[layer].len() <= max_connections {
            return;
        }
        self.trim_neighbors(node_id, layer, max_connections);
    }

    fn trim_neighbors(&mut self, node_id: u64, layer: usize, max_connections: usize) {
        let (vector, neighbor_ids) = {
            let node = self.nodes().get(&node_id).unwrap();
            (node.vector.clone(), node.neighbors[layer].clone())
        };
        let scored: Vec<ScoredNode> = neighbor_ids
            .iter()
            .filter_map(|&nid| {
                let neighbor = self.nodes().get(&nid)?;
                let sim = cosine_similarity(&vector, &neighbor.vector).ok()?;
                Some(ScoredNode {
                    id: nid,
                    similarity: sim,
                })
            })
            .collect();
        let selected = select_neighbors(&scored, max_connections);
        let trimmed: Vec<u64> = selected.iter().map(|s| s.id).collect();
        if let Some(node) = self.nodes_mut().get_mut(&node_id) {
            node.neighbors[layer] = trimmed;
        }
    }

    fn promote_entry_point_if_higher(&mut self, node_id: u64, node_level: usize) {
        if node_level > self.max_level() {
            self.set_entry_point(Some(node_id));
            self.set_max_level(node_level);
        }
    }
}

impl HnswGraph {
    /// Search for k nearest neighbors by cosine similarity.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u64, f32)>, HnswError> {
        self.validate_search_query(query)?;
        if self.is_empty() || k == 0 {
            return Ok(Vec::new());
        }
        let entry_point_id = self.entry_point().expect("non-empty graph has entry point");
        let mut current = self.score_node(entry_point_id, query)?;
        current = self.descend_to_level(current, query, 0);
        let ef = self.params().ef_search.max(k);
        let results = search_layer(self.nodes(), query, &[current], ef, 0);
        Ok(results
            .into_iter()
            .take(k)
            .map(|s| (s.id, s.similarity))
            .collect())
    }

    fn validate_search_query(&self, query: &[f32]) -> Result<(), HnswError> {
        if query.len() != self.dimension() {
            return Err(HnswError::DimensionMismatch {
                expected: self.dimension(),
                got: query.len(),
            });
        }
        Ok(())
    }
}

impl HnswGraph {
    /// Delete a node by ID. Cleans up neighbor references.
    pub fn delete(&mut self, id: u64) -> Result<(), HnswError> {
        let node = self.nodes().get(&id).ok_or(HnswError::NodeNotFound(id))?;
        let node_level = node.level;
        let neighbor_lists: Vec<Vec<u64>> = node.neighbors.clone();

        self.remove_from_all_neighbors(id, &neighbor_lists);
        self.nodes_mut().remove(&id);
        self.update_entry_point_after_delete(id, node_level);
        Ok(())
    }

    fn remove_from_all_neighbors(&mut self, deleted_id: u64, neighbor_lists: &[Vec<u64>]) {
        for (layer, neighbors) in neighbor_lists.iter().enumerate() {
            for &neighbor_id in neighbors {
                if let Some(neighbor) = self.nodes_mut().get_mut(&neighbor_id)
                    && layer < neighbor.neighbors.len()
                {
                    neighbor.neighbors[layer].retain(|&nid| nid != deleted_id);
                }
            }
        }
    }

    fn update_entry_point_after_delete(&mut self, deleted_id: u64, _deleted_level: usize) {
        if self.entry_point() != Some(deleted_id) {
            return;
        }
        if self.is_empty() {
            self.set_entry_point(None);
            self.set_max_level(0);
            return;
        }
        let (new_entry_id, new_max_level) = self
            .nodes()
            .values()
            .map(|n| (n.id, n.level))
            .max_by_key(|&(_, level)| level)
            .unwrap();
        self.set_entry_point(Some(new_entry_id));
        self.set_max_level(new_max_level);
    }
}
