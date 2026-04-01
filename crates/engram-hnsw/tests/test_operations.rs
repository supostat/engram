use engram_hnsw::{HnswError, HnswGraph, HnswParams};
use rand::Rng;

fn build_graph(dimension: usize) -> HnswGraph {
    HnswGraph::new(HnswParams::new(dimension).unwrap())
}

fn random_vector(rng: &mut impl Rng, dimension: usize) -> Vec<f32> {
    (0..dimension)
        .map(|_| rng.random::<f32>() * 2.0 - 1.0)
        .collect()
}

fn normalized(vector: &[f32]) -> Vec<f32> {
    let magnitude: f32 = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    if magnitude == 0.0 {
        return vector.to_vec();
    }
    vector.iter().map(|v| v / magnitude).collect()
}

// --- Insert tests ---

#[test]
fn test_insert_single() {
    let mut graph = build_graph(3);
    graph.insert(1, vec![1.0, 0.0, 0.0], 0.5).unwrap();
    assert_eq!(graph.len(), 1);
    assert!(graph.contains(1));
}

#[test]
fn test_insert_multiple() {
    let mut rng = rand::rng();
    let mut graph = build_graph(128);

    for id in 0..100 {
        let vector = random_vector(&mut rng, 128);
        let rng_value: f64 = rng.random();
        graph.insert(id, vector, rng_value).unwrap();
    }

    assert_eq!(graph.len(), 100);
    for id in 0..100 {
        assert!(graph.contains(id));
    }
}

#[test]
fn test_insert_duplicate_rejected() {
    let mut graph = build_graph(3);
    graph.insert(1, vec![1.0, 0.0, 0.0], 0.5).unwrap();
    let result = graph.insert(1, vec![0.0, 1.0, 0.0], 0.5);
    assert!(matches!(result, Err(HnswError::DuplicateNode(1))));
}

#[test]
fn test_insert_wrong_dimension() {
    let mut graph = build_graph(3);
    let result = graph.insert(1, vec![1.0, 0.0], 0.5);
    assert!(matches!(
        result,
        Err(HnswError::DimensionMismatch {
            expected: 3,
            got: 2
        })
    ));
}

// --- Search tests ---

#[test]
fn test_search_empty_graph() {
    let graph = build_graph(3);
    let results = graph.search(&[1.0, 0.0, 0.0], 5).unwrap();
    assert!(results.is_empty());
}

#[test]
fn test_search_finds_exact_match() {
    let mut graph = build_graph(3);
    graph.insert(1, vec![1.0, 0.0, 0.0], 0.5).unwrap();

    let results = graph.search(&[1.0, 0.0, 0.0], 1).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, 1);
    assert!((results[0].1 - 1.0).abs() < 1e-6);
}

#[test]
fn test_search_top_k() {
    let mut rng = rand::rng();
    let mut graph = build_graph(128);

    for id in 0..100 {
        let vector = random_vector(&mut rng, 128);
        graph.insert(id, vector, rng.random()).unwrap();
    }

    let query = random_vector(&mut rng, 128);
    let results = graph.search(&query, 5).unwrap();

    assert_eq!(results.len(), 5);
    for window in results.windows(2) {
        assert!(
            window[0].1 >= window[1].1,
            "results must be sorted by similarity descending: {} < {}",
            window[0].1,
            window[1].1
        );
    }
}

#[test]
fn test_search_orthogonal() {
    let dim = 128;
    let mut a = vec![0.0f32; dim];
    a[0] = 1.0;
    let mut b = vec![0.0f32; dim];
    b[1] = 1.0;

    let mut graph = build_graph(dim);
    graph.insert(1, a, 0.5).unwrap();

    let results = graph.search(&b, 1).unwrap();
    assert_eq!(results.len(), 1);
    assert!(
        results[0].1.abs() < 0.01,
        "orthogonal vectors should have ~0 similarity, got {}",
        results[0].1
    );
}

#[test]
fn test_search_wrong_dimension() {
    let graph = build_graph(3);
    let result = graph.search(&[1.0, 0.0], 1);
    assert!(matches!(
        result,
        Err(HnswError::DimensionMismatch {
            expected: 3,
            got: 2
        })
    ));
}

// --- Delete tests ---

#[test]
fn test_delete_existing() {
    let mut graph = build_graph(3);
    graph.insert(1, vec![1.0, 0.0, 0.0], 0.5).unwrap();
    graph.insert(2, vec![0.0, 1.0, 0.0], 0.5).unwrap();

    assert_eq!(graph.len(), 2);
    graph.delete(1).unwrap();
    assert_eq!(graph.len(), 1);
    assert!(!graph.contains(1));
    assert!(graph.contains(2));
}

#[test]
fn test_delete_nonexistent() {
    let mut graph = build_graph(3);
    let result = graph.delete(999);
    assert!(matches!(result, Err(HnswError::NodeNotFound(999))));
}

#[test]
fn test_delete_and_search() {
    let mut rng = rand::rng();
    let mut graph = build_graph(128);
    let mut vectors: Vec<Vec<f32>> = Vec::new();

    for id in 0..100 {
        let vector = random_vector(&mut rng, 128);
        vectors.push(vector.clone());
        graph.insert(id, vector, rng.random()).unwrap();
    }

    let deleted_id = 50u64;
    graph.delete(deleted_id).unwrap();

    let query = vectors[deleted_id as usize].clone();
    let results = graph.search(&query, 10).unwrap();

    for (id, _similarity) in &results {
        assert_ne!(
            *id, deleted_id,
            "deleted node must not appear in search results"
        );
    }
}

#[test]
fn test_delete_all_nodes() {
    let mut graph = build_graph(3);
    graph.insert(1, vec![1.0, 0.0, 0.0], 0.5).unwrap();
    graph.insert(2, vec![0.0, 1.0, 0.0], 0.3).unwrap();

    graph.delete(1).unwrap();
    graph.delete(2).unwrap();

    assert!(graph.is_empty());
    let results = graph.search(&[1.0, 0.0, 0.0], 5).unwrap();
    assert!(results.is_empty());
}

// --- Recall test ---

#[test]
fn test_recall_at_10() {
    let mut rng = rand::rng();
    let dimension = 128;
    let node_count = 1000;
    let query_count = 50;
    let k = 10;

    let mut graph = build_graph(dimension);
    let mut vectors: Vec<Vec<f32>> = Vec::new();

    for id in 0..node_count {
        let vector = normalized(&random_vector(&mut rng, dimension));
        vectors.push(vector.clone());
        graph.insert(id as u64, vector, rng.random()).unwrap();
    }

    let mut total_recall = 0.0;

    for _ in 0..query_count {
        let query = normalized(&random_vector(&mut rng, dimension));
        let hnsw_results = graph.search(&query, k).unwrap();
        let hnsw_ids: Vec<u64> = hnsw_results.iter().map(|r| r.0).collect();
        let brute_force_top_k = brute_force_search(&vectors, &query, k);
        let hits = hnsw_ids
            .iter()
            .filter(|id| brute_force_top_k.contains(id))
            .count();
        total_recall += hits as f64 / k as f64;
    }

    let average_recall = total_recall / query_count as f64;
    assert!(
        average_recall > 0.95,
        "recall@{k} = {average_recall:.3}, expected > 0.95"
    );
}

fn brute_force_search(vectors: &[Vec<f32>], query: &[f32], k: usize) -> Vec<u64> {
    let mut scored: Vec<(u64, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(id, vector)| {
            let similarity = engram_hnsw::cosine_similarity(query, vector).unwrap();
            (id as u64, similarity)
        })
        .collect();
    scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scored.into_iter().take(k).map(|(id, _)| id).collect()
}
