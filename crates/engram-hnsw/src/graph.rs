use std::collections::HashMap;

use crate::error::HnswError;
use crate::node::Node;

/// HNSW graph parameters.
pub struct HnswParams {
    /// Max number of connections per node per layer (default: 16)
    pub max_connections: usize,
    /// Max connections at layer 0 (typically 2 * max_connections = 32)
    pub max_connections_layer0: usize,
    /// Size of dynamic candidate list during construction (default: 200)
    pub ef_construction: usize,
    /// Size of dynamic candidate list during search (default: 50)
    pub ef_search: usize,
    /// Vector dimension
    pub dimension: usize,
}

impl HnswParams {
    pub fn new(dimension: usize) -> Result<Self, HnswError> {
        if dimension == 0 || dimension > 65536 {
            return Err(HnswError::InvalidParameter(format!(
                "dimension must be in 1..=65536, got {dimension}"
            )));
        }
        Ok(Self {
            max_connections: 16,
            max_connections_layer0: 32,
            ef_construction: 200,
            ef_search: 50,
            dimension,
        })
    }

    pub fn with_max_connections(mut self, max_connections: usize) -> Result<Self, HnswError> {
        if !(2..=256).contains(&max_connections) {
            return Err(HnswError::InvalidParameter(format!(
                "max_connections must be in 2..=256, got {max_connections}"
            )));
        }
        self.max_connections = max_connections;
        self.max_connections_layer0 = max_connections * 2;
        Ok(self)
    }

    pub fn with_ef_construction(mut self, ef_construction: usize) -> Result<Self, HnswError> {
        if ef_construction == 0 {
            return Err(HnswError::InvalidParameter(
                "ef_construction must be > 0".to_owned(),
            ));
        }
        self.ef_construction = ef_construction;
        Ok(self)
    }

    pub fn with_ef_search(mut self, ef_search: usize) -> Result<Self, HnswError> {
        if ef_search == 0 {
            return Err(HnswError::InvalidParameter(
                "ef_search must be > 0".to_owned(),
            ));
        }
        self.ef_search = ef_search;
        Ok(self)
    }
}

/// HNSW approximate nearest neighbor index.
pub struct HnswGraph {
    params: HnswParams,
    nodes: HashMap<u64, Node>,
    entry_point: Option<u64>,
    max_level: usize,
    /// Inverse of ln(max_connections), used for random level generation.
    level_multiplier: f64,
}

impl HnswGraph {
    pub fn new(params: HnswParams) -> Self {
        let level_multiplier = 1.0 / (params.max_connections as f64).ln();
        Self {
            params,
            nodes: HashMap::new(),
            entry_point: None,
            max_level: 0,
            level_multiplier,
        }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn dimension(&self) -> usize {
        self.params.dimension
    }

    pub fn contains(&self, id: u64) -> bool {
        self.nodes.contains_key(&id)
    }

    /// Generate random level for a new node.
    /// Level = floor(-ln(uniform_value) * level_multiplier)
    /// where `uniform_value` is drawn from uniform(0, 1).
    pub fn random_level(&self, uniform_value: f64) -> usize {
        let clamped = uniform_value.clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON);
        let level = (-clamped.ln() * self.level_multiplier) as usize;
        level.min(32)
    }

    pub(crate) fn params(&self) -> &HnswParams {
        &self.params
    }

    pub(crate) fn nodes(&self) -> &HashMap<u64, Node> {
        &self.nodes
    }

    pub(crate) fn nodes_mut(&mut self) -> &mut HashMap<u64, Node> {
        &mut self.nodes
    }

    pub(crate) fn entry_point(&self) -> Option<u64> {
        self.entry_point
    }

    pub(crate) fn set_entry_point(&mut self, entry: Option<u64>) {
        self.entry_point = entry;
    }

    pub(crate) fn max_level(&self) -> usize {
        self.max_level
    }

    pub(crate) fn set_max_level(&mut self, level: usize) {
        self.max_level = level;
    }

    pub(crate) fn max_connections_for_layer(&self, layer: usize) -> usize {
        if layer == 0 {
            self.params.max_connections_layer0
        } else {
            self.params.max_connections
        }
    }
}
