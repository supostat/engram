use engram_core::CoreError;
use engram_core::IndexSet;
use engram_embeddings::ThreeFieldEmbedding;
use engram_hnsw::HnswParams;

fn test_params() -> Result<HnswParams, CoreError> {
    HnswParams::new(4)?
        .with_max_connections(4)?
        .with_ef_construction(16)?
        .with_ef_search(8)
        .map_err(CoreError::Hnsw)
}

fn make_embedding(values: [f32; 4]) -> ThreeFieldEmbedding {
    ThreeFieldEmbedding {
        context: values.to_vec(),
        action: values.to_vec(),
        result: values.to_vec(),
    }
}

#[test]
fn new_index_set_is_empty() {
    let indexes = IndexSet::new(test_params).unwrap();
    assert!(indexes.is_empty());
    assert_eq!(indexes.len(), 0);
}

#[test]
fn insert_and_contains() {
    let mut indexes = IndexSet::new(test_params).unwrap();
    let embedding = make_embedding([1.0, 0.0, 0.0, 0.0]);
    indexes.insert(1, &embedding, 0.5).unwrap();
    assert!(indexes.contains(1));
    assert!(!indexes.contains(2));
    assert_eq!(indexes.len(), 1);
}

#[test]
fn search_returns_inserted_items() {
    let mut indexes = IndexSet::new(test_params).unwrap();
    let emb1 = make_embedding([1.0, 0.0, 0.0, 0.0]);
    let emb2 = make_embedding([0.0, 1.0, 0.0, 0.0]);
    indexes.insert(1, &emb1, 0.3).unwrap();
    indexes.insert(2, &emb2, 0.7).unwrap();

    let results = indexes.search(&[1.0, 0.0, 0.0, 0.0], 2).unwrap();
    assert!(!results.is_empty());
    assert_eq!(results[0].0, 1);
}

#[test]
fn delete_removes_from_all_indexes() {
    let mut indexes = IndexSet::new(test_params).unwrap();
    let embedding = make_embedding([1.0, 0.0, 0.0, 0.0]);
    indexes.insert(1, &embedding, 0.5).unwrap();
    indexes.delete(1).unwrap();
    assert!(!indexes.contains(1));
    assert!(indexes.is_empty());
}

#[test]
fn search_empty_index_returns_empty() {
    let indexes = IndexSet::new(test_params).unwrap();
    let results = indexes.search(&[1.0, 0.0, 0.0, 0.0], 10).unwrap();
    assert!(results.is_empty());
}

#[test]
fn serialize_deserialize_roundtrip() {
    let mut indexes = IndexSet::new(test_params).unwrap();
    let emb1 = make_embedding([1.0, 0.0, 0.0, 0.0]);
    let emb2 = make_embedding([0.0, 1.0, 0.0, 0.0]);
    indexes.insert(10, &emb1, 0.3).unwrap();
    indexes.insert(20, &emb2, 0.7).unwrap();

    let mut buffer = Vec::new();
    indexes.serialize(&mut buffer).unwrap();

    let mut reader = std::io::Cursor::new(buffer);
    let restored = IndexSet::deserialize(&mut reader).unwrap();
    assert_eq!(restored.len(), 2);
    assert!(restored.contains(10));
    assert!(restored.contains(20));
}

#[test]
fn search_merges_across_three_indexes() {
    let mut indexes = IndexSet::new(test_params).unwrap();
    let embedding = ThreeFieldEmbedding {
        context: vec![1.0, 0.0, 0.0, 0.0],
        action: vec![0.0, 1.0, 0.0, 0.0],
        result: vec![0.0, 0.0, 1.0, 0.0],
    };
    indexes.insert(1, &embedding, 0.5).unwrap();

    let results_by_context = indexes.search(&[1.0, 0.0, 0.0, 0.0], 5).unwrap();
    let results_by_action = indexes.search(&[0.0, 1.0, 0.0, 0.0], 5).unwrap();
    let results_by_result = indexes.search(&[0.0, 0.0, 1.0, 0.0], 5).unwrap();

    assert!(!results_by_context.is_empty());
    assert!(!results_by_action.is_empty());
    assert!(!results_by_result.is_empty());
    assert_eq!(results_by_context[0].0, 1);
    assert_eq!(results_by_action[0].0, 1);
    assert_eq!(results_by_result[0].0, 1);
}

#[test]
fn duplicate_insert_returns_error() {
    let mut indexes = IndexSet::new(test_params).unwrap();
    let embedding = make_embedding([1.0, 0.0, 0.0, 0.0]);
    indexes.insert(1, &embedding, 0.5).unwrap();
    let result = indexes.insert(1, &embedding, 0.5);
    assert!(result.is_err());
}
