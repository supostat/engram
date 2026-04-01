/// A node in the HNSW graph.
/// Each node exists on levels 0..=level, with neighbor lists per level.
#[derive(Clone)]
pub struct Node {
    pub id: u64,
    pub vector: Vec<f32>,
    pub level: usize,
    /// Neighbors per level. `neighbors[i]` = neighbor IDs at level i.
    /// Length = level + 1.
    pub neighbors: Vec<Vec<u64>>,
}

impl Node {
    const MAX_LEVEL: usize = 32;

    pub fn new(id: u64, vector: Vec<f32>, level: usize) -> Self {
        let level = level.min(Self::MAX_LEVEL);
        let neighbors = (0..=level).map(|_| Vec::new()).collect();
        Self {
            id,
            vector,
            level,
            neighbors,
        }
    }
}
