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

fn roundtrip(graph: &HnswGraph) -> HnswGraph {
    let mut buffer = Vec::new();
    graph.serialize(&mut buffer).unwrap();
    HnswGraph::deserialize(&mut buffer.as_slice()).unwrap()
}

#[test]
fn test_serialize_empty_graph() {
    let graph = build_graph(64);
    let restored = roundtrip(&graph);

    assert_eq!(restored.len(), 0);
    assert!(restored.is_empty());
    assert_eq!(restored.dimension(), 64);
}

#[test]
fn test_serialize_roundtrip() {
    let mut rng = rand::rng();
    let mut graph = build_graph(32);

    for id in 0..100 {
        let vector = random_vector(&mut rng, 32);
        graph.insert(id, vector, rng.random()).unwrap();
    }

    let restored = roundtrip(&graph);
    assert_eq!(restored.len(), 100);

    let query = random_vector(&mut rng, 32);
    let original_results = graph.search(&query, 5).unwrap();
    let restored_results = restored.search(&query, 5).unwrap();

    assert_eq!(original_results.len(), restored_results.len());
    for (original, restored) in original_results.iter().zip(restored_results.iter()) {
        assert_eq!(original.0, restored.0, "IDs must match after roundtrip");
        assert!(
            (original.1 - restored.1).abs() < 1e-6,
            "similarities must match after roundtrip"
        );
    }
}

#[test]
fn test_serialize_invalid_magic() {
    let mut buffer = Vec::new();
    buffer.extend_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
    buffer.extend_from_slice(&1u32.to_le_bytes());

    let result = HnswGraph::deserialize(&mut buffer.as_slice());
    assert!(
        matches!(result, Err(HnswError::IndexCorrupted(_))),
        "expected IndexCorrupted error"
    );
}

#[test]
fn test_serialize_invalid_version() {
    let mut buffer = Vec::new();
    buffer.extend_from_slice(&0x484E_5357u32.to_le_bytes()); // valid magic
    buffer.extend_from_slice(&99u32.to_le_bytes()); // invalid version

    let result = HnswGraph::deserialize(&mut buffer.as_slice());
    assert!(
        matches!(result, Err(HnswError::RebuildRequired)),
        "expected RebuildRequired error"
    );
}

#[test]
fn test_serialize_preserves_params() {
    let params = HnswParams::new(256)
        .unwrap()
        .with_max_connections(32)
        .unwrap()
        .with_ef_construction(100)
        .unwrap()
        .with_ef_search(75)
        .unwrap();
    let graph = HnswGraph::new(params);

    let restored = roundtrip(&graph);
    assert_eq!(restored.dimension(), 256);
}

#[test]
fn test_serialize_single_node() {
    let mut graph = build_graph(3);
    graph.insert(42, vec![1.0, 2.0, 3.0], 0.5).unwrap();

    let restored = roundtrip(&graph);
    assert_eq!(restored.len(), 1);
    assert!(restored.contains(42));

    let results = restored.search(&[1.0, 2.0, 3.0], 1).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, 42);
}

#[test]
fn test_serialize_truncated_data() {
    let mut graph = build_graph(3);
    graph.insert(1, vec![1.0, 0.0, 0.0], 0.5).unwrap();

    let mut buffer = Vec::new();
    graph.serialize(&mut buffer).unwrap();

    let truncated = &buffer[..buffer.len() / 2];
    let result = HnswGraph::deserialize(&mut &*truncated);
    assert!(
        result.is_err(),
        "truncated data should fail deserialization"
    );
}
