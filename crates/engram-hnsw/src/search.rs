use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::node::Node;
use crate::similarity::cosine_similarity;

/// Neighbor scored by cosine similarity, used in priority queue operations.
#[derive(Clone)]
pub struct ScoredNode {
    pub id: u64,
    pub similarity: f32,
}

impl PartialEq for ScoredNode {
    fn eq(&self, other: &Self) -> bool {
        self.similarity == other.similarity && self.id == other.id
    }
}

impl Eq for ScoredNode {}

/// Max-heap ordering: highest similarity first, then lower ID for stability.
impl Ord for ScoredNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.similarity
            .partial_cmp(&other.similarity)
            .unwrap_or(Ordering::Equal)
            .then(other.id.cmp(&self.id))
    }
}

impl PartialOrd for ScoredNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Min-heap wrapper: inverts ordering so BinaryHeap pops lowest similarity first.
#[derive(Clone, Eq, PartialEq)]
struct MinScoredNode(ScoredNode);

impl Ord for MinScoredNode {
    fn cmp(&self, other: &Self) -> Ordering {
        other.0.cmp(&self.0)
    }
}

impl PartialOrd for MinScoredNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Mutable search state: candidates to explore and current best results.
struct SearchState {
    visited: HashSet<u64>,
    candidates: BinaryHeap<ScoredNode>,
    results: BinaryHeap<MinScoredNode>,
    ef: usize,
}

impl SearchState {
    fn new(entry_points: &[ScoredNode], ef: usize) -> Self {
        let mut state = Self {
            visited: HashSet::new(),
            candidates: BinaryHeap::new(),
            results: BinaryHeap::new(),
            ef,
        };
        for entry in entry_points {
            state.visited.insert(entry.id);
            state.candidates.push(entry.clone());
            state.results.push(MinScoredNode(entry.clone()));
        }
        state
    }

    fn farthest_similarity(&self) -> f32 {
        self.results
            .peek()
            .map_or(f32::NEG_INFINITY, |m| m.0.similarity)
    }

    fn should_stop(&self, candidate_similarity: f32) -> bool {
        candidate_similarity < self.farthest_similarity() && self.results.len() >= self.ef
    }

    fn try_add(&mut self, id: u64, similarity: f32) {
        if similarity > self.farthest_similarity() || self.results.len() < self.ef {
            let scored = ScoredNode { id, similarity };
            self.candidates.push(scored.clone());
            self.results.push(MinScoredNode(scored));
            while self.results.len() > self.ef {
                self.results.pop();
            }
        }
    }
}

/// Search a single layer for nearest neighbors.
/// Returns up to `ef` nearest neighbors by cosine similarity.
///
/// Uses max-heap for candidates (most similar first) and
/// min-heap for results (least similar on top for fast trimming).
pub fn search_layer(
    nodes: &HashMap<u64, Node>,
    query: &[f32],
    entry_points: &[ScoredNode],
    ef: usize,
    layer: usize,
) -> Vec<ScoredNode> {
    let mut state = SearchState::new(entry_points, ef);

    while let Some(candidate) = state.candidates.pop() {
        if state.should_stop(candidate.similarity) {
            break;
        }
        expand_candidate_neighbors(nodes, query, &candidate, layer, &mut state);
    }

    collect_sorted_results(state.results)
}

fn expand_candidate_neighbors(
    nodes: &HashMap<u64, Node>,
    query: &[f32],
    candidate: &ScoredNode,
    layer: usize,
    state: &mut SearchState,
) {
    let Some(candidate_node) = nodes.get(&candidate.id) else {
        return;
    };
    if layer >= candidate_node.neighbors.len() {
        return;
    }
    for &neighbor_id in &candidate_node.neighbors[layer] {
        if !state.visited.insert(neighbor_id) {
            continue;
        }
        let Some(neighbor_node) = nodes.get(&neighbor_id) else {
            continue;
        };
        let Ok(similarity) = cosine_similarity(query, &neighbor_node.vector) else {
            continue;
        };
        state.try_add(neighbor_id, similarity);
    }
}

fn collect_sorted_results(results: BinaryHeap<MinScoredNode>) -> Vec<ScoredNode> {
    let mut sorted: Vec<ScoredNode> = results.into_iter().map(|m| m.0).collect();
    sorted.sort_unstable_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(Ordering::Equal)
    });
    sorted
}

/// Select best neighbors from candidates. Takes top-M by similarity.
pub fn select_neighbors(candidates: &[ScoredNode], max_neighbors: usize) -> Vec<ScoredNode> {
    let mut sorted = candidates.to_vec();
    sorted.sort_unstable_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(Ordering::Equal)
    });
    sorted.truncate(max_neighbors);
    sorted
}
