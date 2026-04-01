use criterion::{Criterion, criterion_group, criterion_main};
use engram_hnsw::{HnswGraph, HnswParams};
use rand::Rng;

fn random_vector(rng: &mut impl Rng, dimension: usize) -> Vec<f32> {
    (0..dimension)
        .map(|_| rng.random::<f32>() * 2.0 - 1.0)
        .collect()
}

fn build_populated_graph(node_count: u64, dimension: usize) -> HnswGraph {
    let mut rng = rand::rng();
    let mut graph = HnswGraph::new(HnswParams::new(dimension).unwrap());
    for id in 0..node_count {
        let vector = random_vector(&mut rng, dimension);
        graph.insert(id, vector, rng.random()).unwrap();
    }
    graph
}

fn bench_insert_1k(criterion: &mut Criterion) {
    let dimension = 128;
    criterion.bench_function("insert_1k_128d", |bencher| {
        bencher.iter(|| build_populated_graph(1000, dimension));
    });
}

fn bench_search_in_1k(criterion: &mut Criterion) {
    let dimension = 128;
    let graph = build_populated_graph(1000, dimension);
    let mut rng = rand::rng();

    criterion.bench_function("search_top5_in_1k_128d", |bencher| {
        bencher.iter(|| {
            let query = random_vector(&mut rng, dimension);
            graph.search(&query, 5).unwrap()
        });
    });
}

fn bench_search_in_10k(criterion: &mut Criterion) {
    let dimension = 128;
    let graph = build_populated_graph(10_000, dimension);
    let mut rng = rand::rng();

    criterion.bench_function("search_top5_in_10k_128d", |bencher| {
        bencher.iter(|| {
            let query = random_vector(&mut rng, dimension);
            graph.search(&query, 5).unwrap()
        });
    });
}

fn bench_serialize_10k(criterion: &mut Criterion) {
    let graph = build_populated_graph(10_000, 128);

    criterion.bench_function("serialize_10k_128d", |bencher| {
        bencher.iter(|| {
            let mut buffer = Vec::new();
            graph.serialize(&mut buffer).unwrap();
            buffer
        });
    });
}

fn bench_deserialize_10k(criterion: &mut Criterion) {
    let graph = build_populated_graph(10_000, 128);
    let mut buffer = Vec::new();
    graph.serialize(&mut buffer).unwrap();

    criterion.bench_function("deserialize_10k_128d", |bencher| {
        bencher.iter(|| HnswGraph::deserialize(&mut buffer.as_slice()).unwrap());
    });
}

criterion_group!(
    benches,
    bench_insert_1k,
    bench_search_in_1k,
    bench_search_in_10k,
    bench_serialize_10k,
    bench_deserialize_10k,
);
criterion_main!(benches);
