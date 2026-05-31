// Proves the "recall@k NO WORSE" guarantee: Alg.4 (Heuristic) diversity-aware
// neighbor selection must not regress recall versus the previous naive top-M.
// Both graphs are built on the SAME seeded corpus with the SAME per-node level
// values, so the only variable is the neighbor-selection strategy. Ground truth
// is a shared brute-force top-k over the corpus.
//
// On uniform-random vectors the two strategies are expected to be ~equal:
// without cluster structure the diversity heuristic finds no near-duplicates to
// prune, so it degenerates toward top-M. Diversity helps on CLUSTERED data
// (out of scope here). The assertion below therefore guards against a recall
// REGRESSION (Alg.4 >= naive within epsilon), not for a quality gain.

use engram_hnsw::{HnswGraph, HnswParams, NeighborSelection, cosine_similarity};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

const CORPUS_SEED: u64 = 0x5EED_C0DE;
const QUERY_SEED: u64 = 0x5EED_F00D;
const CORPUS_SIZE: usize = 1000;
const QUERY_COUNT: usize = 30;
const DIMENSION: usize = 64;
const K: usize = 10;
// "NO WORSE" tolerance: Alg.4 mean recall must be within EPSILON of naive mean.
const EPSILON: f64 = 0.02;
// Catastrophe guard: a wholesale collapse trips this even if naive also dropped.
const ABSOLUTE_FLOOR: f64 = 0.85;

fn unit_vector(rng: &mut StdRng, dimension: usize) -> Vec<f32> {
    let raw: Vec<f32> = (0..dimension)
        .map(|_| rng.random::<f32>() * 2.0 - 1.0)
        .collect();
    let magnitude: f32 = raw
        .iter()
        .map(|component| component * component)
        .sum::<f32>()
        .sqrt();
    raw.iter().map(|component| component / magnitude).collect()
}

fn brute_force_top_k(corpus: &[Vec<f32>], query: &[f32], k: usize) -> Vec<u64> {
    let mut scored: Vec<(u64, f32)> = corpus
        .iter()
        .enumerate()
        .map(|(id, vector)| (id as u64, cosine_similarity(query, vector).unwrap()))
        .collect();
    scored.sort_unstable_by(|left, right| right.1.partial_cmp(&left.1).unwrap());
    scored.into_iter().take(k).map(|(id, _)| id).collect()
}

fn build_graph(
    corpus: &[Vec<f32>],
    level_values: &[f64],
    selection: NeighborSelection,
) -> HnswGraph {
    let mut graph = HnswGraph::new(
        HnswParams::new(DIMENSION)
            .unwrap()
            .with_ef_construction(200)
            .unwrap()
            .with_ef_search(40)
            .unwrap()
            .with_neighbor_selection(selection),
    );
    for (id, vector) in corpus.iter().enumerate() {
        graph
            .insert(id as u64, vector.clone(), level_values[id])
            .unwrap();
    }
    graph
}

fn mean_recall_at_k(graph: &HnswGraph, queries: &[Vec<f32>], truths: &[Vec<u64>]) -> f64 {
    let mut total_recall = 0.0;
    for (query, truth) in queries.iter().zip(truths) {
        let hnsw_ids: Vec<u64> = graph
            .search(query, K)
            .unwrap()
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        let hits = hnsw_ids.iter().filter(|id| truth.contains(id)).count();
        total_recall += hits as f64 / K as f64;
    }
    total_recall / queries.len() as f64
}

#[test]
fn recall_at_10_no_worse_than_naive() {
    let mut corpus_rng = StdRng::seed_from_u64(CORPUS_SEED);
    let mut corpus: Vec<Vec<f32>> = Vec::with_capacity(CORPUS_SIZE);
    let mut level_values: Vec<f64> = Vec::with_capacity(CORPUS_SIZE);
    for _ in 0..CORPUS_SIZE {
        corpus.push(unit_vector(&mut corpus_rng, DIMENSION));
        level_values.push(corpus_rng.random());
    }

    let naive_graph = build_graph(&corpus, &level_values, NeighborSelection::Naive);
    let alg4_graph = build_graph(&corpus, &level_values, NeighborSelection::Heuristic);

    let mut query_rng = StdRng::seed_from_u64(QUERY_SEED);
    let mut queries: Vec<Vec<f32>> = Vec::with_capacity(QUERY_COUNT);
    let mut truths: Vec<Vec<u64>> = Vec::with_capacity(QUERY_COUNT);
    for _ in 0..QUERY_COUNT {
        let query = unit_vector(&mut query_rng, DIMENSION);
        truths.push(brute_force_top_k(&corpus, &query, K));
        queries.push(query);
    }

    let naive_mean = mean_recall_at_k(&naive_graph, &queries, &truths);
    let alg4_mean = mean_recall_at_k(&alg4_graph, &queries, &truths);

    println!("recall@{K}: naive_mean={naive_mean:.4} alg4_mean={alg4_mean:.4} (eps={EPSILON})");

    // Headline guarantee: Alg.4 is no worse than naive top-M within epsilon.
    assert!(
        alg4_mean >= naive_mean - EPSILON,
        "Alg.4 recall regressed below naive: alg4={alg4_mean:.4} < naive {naive_mean:.4} - eps \
         {EPSILON} (corpus={CORPUS_SIZE}, queries={QUERY_COUNT}, dim={DIMENSION})"
    );
    // Catastrophe guard: a wholesale collapse trips regardless of the baseline.
    assert!(
        alg4_mean >= ABSOLUTE_FLOOR,
        "Alg.4 recall@{K} below absolute floor: {alg4_mean:.4} < {ABSOLUTE_FLOOR} \
         (corpus={CORPUS_SIZE}, queries={QUERY_COUNT}, dim={DIMENSION})"
    );
}
