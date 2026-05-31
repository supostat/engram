// Search-latency benchmark over the same synthetic set used by the recall
// regression test (tests/recall.rs). Index is built once; only `graph.search`
// is measured. Recall is printed as metadata only — assertions live in the
// #[test], never here.

use criterion::{Criterion, criterion_group, criterion_main};
use engram_hnsw::{HnswGraph, HnswParams, cosine_similarity};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

const CORPUS_SEED: u64 = 0x5EED_C0DE;
const QUERY_SEED: u64 = 0x5EED_F00D;
const CORPUS_SIZE: usize = 2000;
const QUERY_COUNT: usize = 50;
const DIMENSION: usize = 128;
const K: usize = 10;

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

fn build_corpus() -> (HnswGraph, Vec<Vec<f32>>) {
    let mut corpus_rng = StdRng::seed_from_u64(CORPUS_SEED);
    let mut graph = HnswGraph::new(
        HnswParams::new(DIMENSION)
            .unwrap()
            .with_ef_construction(200)
            .unwrap()
            .with_ef_search(40)
            .unwrap(),
    );
    let mut corpus: Vec<Vec<f32>> = Vec::with_capacity(CORPUS_SIZE);
    for id in 0..CORPUS_SIZE {
        let vector = unit_vector(&mut corpus_rng, DIMENSION);
        corpus.push(vector.clone());
        graph
            .insert(id as u64, vector, corpus_rng.random())
            .unwrap();
    }
    (graph, corpus)
}

fn build_queries() -> Vec<Vec<f32>> {
    let mut query_rng = StdRng::seed_from_u64(QUERY_SEED);
    (0..QUERY_COUNT)
        .map(|_| unit_vector(&mut query_rng, DIMENSION))
        .collect()
}

fn mean_recall(graph: &HnswGraph, corpus: &[Vec<f32>], queries: &[Vec<f32>]) -> f64 {
    let mut total = 0.0;
    for query in queries {
        let truth = brute_force_top_k(corpus, query, K);
        let hits = graph
            .search(query, K)
            .unwrap()
            .into_iter()
            .filter(|(id, _)| truth.contains(id))
            .count();
        total += hits as f64 / K as f64;
    }
    total / queries.len() as f64
}

fn bench_search_latency(criterion: &mut Criterion) {
    let (graph, corpus) = build_corpus();
    let queries = build_queries();

    println!(
        "recall_bench metadata: mean recall@{K} = {:.4} (corpus={CORPUS_SIZE}, queries={QUERY_COUNT}, dim={DIMENSION})",
        mean_recall(&graph, &corpus, &queries)
    );

    let mut query_cursor = 0usize;
    criterion.bench_function("search_top10_in_2k_128d", |bencher| {
        bencher.iter(|| {
            let query = &queries[query_cursor % queries.len()];
            query_cursor += 1;
            graph.search(query, K).unwrap()
        });
    });
}

criterion_group!(benches, bench_search_latency);
criterion_main!(benches);
