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

/// Malkov-Yashunin Algorithm 4: diversity-aware neighbor selection.
/// Accept a candidate unless an already-selected neighbor is MORE similar to it
/// than the query is (i.e. it would just duplicate an existing edge's direction);
/// deferred candidates backfill to guarantee up to `max_neighbors`
/// (keepPrunedConnections). `candidate_vectors` maps candidate id -> its vector,
/// pre-extracted by the caller to avoid borrowing the graph during mutation.
pub fn select_neighbors(
    candidates: &[ScoredNode],
    max_neighbors: usize,
    candidate_vectors: &HashMap<u64, Vec<f32>>,
) -> Vec<ScoredNode> {
    if candidates.is_empty() || max_neighbors == 0 {
        return Vec::new();
    }
    let mut sorted = candidates.to_vec();
    sorted.sort_unstable_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(Ordering::Equal)
    });
    let mut selected: Vec<ScoredNode> = Vec::new();
    let mut deferred: Vec<ScoredNode> = Vec::new();
    for candidate in sorted {
        if selected.len() >= max_neighbors {
            break;
        }
        if is_diverse(&candidate, &selected, candidate_vectors) {
            selected.push(candidate);
        } else {
            deferred.push(candidate);
        }
    }
    for candidate in deferred {
        if selected.len() >= max_neighbors {
            break;
        }
        selected.push(candidate);
    }
    selected
}

/// Legacy top-M selection: keep the `max_neighbors` most similar candidates,
/// ignoring inter-neighbor diversity. Retained as the comparison baseline that
/// proves Alg.4 is no worse on recall.
pub fn select_neighbors_naive(candidates: &[ScoredNode], max_neighbors: usize) -> Vec<ScoredNode> {
    let mut sorted = candidates.to_vec();
    sorted.sort_unstable_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(Ordering::Equal)
    });
    sorted.truncate(max_neighbors);
    sorted
}

/// True unless an already-selected neighbor sits closer to `candidate` than the
/// query does. A missing vector is treated defensively as diverse (accept).
fn is_diverse(
    candidate: &ScoredNode,
    selected: &[ScoredNode],
    candidate_vectors: &HashMap<u64, Vec<f32>>,
) -> bool {
    let Some(candidate_vector) = candidate_vectors.get(&candidate.id) else {
        return true;
    };
    !selected.iter().any(|neighbor| {
        candidate_vectors
            .get(&neighbor.id)
            .and_then(|neighbor_vector| cosine_similarity(candidate_vector, neighbor_vector).ok())
            .is_some_and(|neighbor_similarity| neighbor_similarity > candidate.similarity)
    })
}

#[cfg(test)]
mod select_neighbors_tests {
    use super::*;

    fn scored(id: u64, similarity: f32) -> ScoredNode {
        ScoredNode { id, similarity }
    }

    #[test]
    fn defers_near_duplicate_in_favor_of_diverse_candidate() {
        // A and B point in nearly the same direction (cosine(B, A) high), so B
        // duplicates A's edge. C points elsewhere and survives.
        let vectors = HashMap::from([
            (1u64, vec![1.0, 0.0, 0.0]),
            (2u64, vec![0.99, 0.14, 0.0]),
            (3u64, vec![0.0, 0.0, 1.0]),
        ]);
        let candidates = [scored(1, 0.99), scored(2, 0.98), scored(3, 0.5)];

        let cosine_b_a = cosine_similarity(&vectors[&2], &vectors[&1]).unwrap();
        assert!(
            cosine_b_a > 0.98,
            "B must be more similar to A than to query"
        );
        let cosine_c_a = cosine_similarity(&vectors[&3], &vectors[&1]).unwrap();
        assert!(
            cosine_c_a < 0.5,
            "C must be less similar to A than to query"
        );

        let selected = select_neighbors(&candidates, 2, &vectors);
        let ids: Vec<u64> = selected.iter().map(|s| s.id).collect();
        assert_eq!(ids, vec![1, 3], "B deferred as near-duplicate of A");

        // Non-vacuity: naive top-M WOULD pick the near-duplicate B over the
        // diverse C, so the two strategies genuinely diverge here.
        let naive_ids: Vec<u64> = select_neighbors_naive(&candidates, 2)
            .iter()
            .map(|s| s.id)
            .collect();
        assert_eq!(
            naive_ids,
            vec![1, 2],
            "naive top-M keeps the near-duplicate"
        );
    }

    #[test]
    fn backfills_deferred_to_reach_max_neighbors() {
        // Two near-duplicates of A: both deferred, but backfill restores them
        // because we have room and no diverse alternatives.
        let vectors = HashMap::from([
            (1u64, vec![1.0, 0.0, 0.0]),
            (2u64, vec![0.99, 0.14, 0.0]),
            (3u64, vec![0.98, 0.2, 0.0]),
        ]);
        let candidates = [scored(1, 0.99), scored(2, 0.98), scored(3, 0.97)];

        let selected = select_neighbors(&candidates, 3, &vectors);
        assert_eq!(selected.len(), 3, "deferred candidates backfill to max");
    }

    #[test]
    fn keeps_all_when_candidates_within_max() {
        let vectors = HashMap::from([(1u64, vec![1.0, 0.0, 0.0]), (2u64, vec![0.99, 0.14, 0.0])]);
        let candidates = [scored(1, 0.99), scored(2, 0.98)];

        let selected = select_neighbors(&candidates, 5, &vectors);
        assert_eq!(selected.len(), 2, "degenerate case equals old top-M");
    }

    #[test]
    fn accepts_candidate_with_missing_vector_without_panic() {
        let vectors = HashMap::from([(1u64, vec![1.0, 0.0, 0.0])]);
        let candidates = [scored(1, 0.99), scored(2, 0.98)];

        let selected = select_neighbors(&candidates, 2, &vectors);
        let ids: Vec<u64> = selected.iter().map(|s| s.id).collect();
        assert_eq!(ids, vec![1, 2], "missing vector defensively accepted");
    }

    #[test]
    fn returns_empty_for_zero_max_or_no_candidates() {
        let vectors = HashMap::from([(1u64, vec![1.0, 0.0, 0.0])]);
        assert!(select_neighbors(&[scored(1, 0.99)], 0, &vectors).is_empty());
        assert!(select_neighbors(&[], 5, &vectors).is_empty());
    }
}
