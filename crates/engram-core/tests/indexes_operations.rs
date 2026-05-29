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
    indexes.insert_atomic(1, "mem-1", &embedding, 0.5).unwrap();
    assert!(indexes.contains(1));
    assert!(!indexes.contains(2));
    assert_eq!(indexes.len(), 1);
}

#[test]
fn search_returns_inserted_items() {
    let mut indexes = IndexSet::new(test_params).unwrap();
    let emb1 = make_embedding([1.0, 0.0, 0.0, 0.0]);
    let emb2 = make_embedding([0.0, 1.0, 0.0, 0.0]);
    indexes.insert_atomic(1, "mem-1", &emb1, 0.3).unwrap();
    indexes.insert_atomic(2, "mem-2", &emb2, 0.7).unwrap();

    let results = indexes.search(&[1.0, 0.0, 0.0, 0.0], 2).unwrap();
    assert!(!results.is_empty());
    assert_eq!(results[0].0, 1);
}

#[test]
fn delete_removes_from_all_indexes() {
    let mut indexes = IndexSet::new(test_params).unwrap();
    let embedding = make_embedding([1.0, 0.0, 0.0, 0.0]);
    indexes.insert_atomic(1, "mem-1", &embedding, 0.5).unwrap();
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
    indexes.insert_atomic(10, "mem-10", &emb1, 0.3).unwrap();
    indexes.insert_atomic(20, "mem-20", &emb2, 0.7).unwrap();

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
    indexes.insert_atomic(1, "mem-1", &embedding, 0.5).unwrap();

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
fn insert_atomic_rejects_hash_collision() {
    let mut indexes = IndexSet::new(test_params).unwrap();
    let embedding = make_embedding([1.0, 0.0, 0.0, 0.0]);
    indexes.insert_atomic(7, "mem-1", &embedding, 0.5).unwrap();

    let result = indexes.insert_atomic(7, "mem-2", &embedding, 0.5);
    match result {
        Err(CoreError::IndexHashCollision {
            hash,
            existing_id,
            conflicting_id,
        }) => {
            assert_eq!(hash, 7);
            assert_eq!(existing_id, "mem-1");
            assert_eq!(conflicting_id, "mem-2");
        }
        other => panic!("expected IndexHashCollision, got {other:?}"),
    }
    assert_eq!(indexes.resolve_node_id(7), Some("mem-1"));
}

#[test]
fn insert_atomic_is_idempotent_for_same_memory_id() {
    let mut indexes = IndexSet::new(test_params).unwrap();
    let embedding = make_embedding([1.0, 0.0, 0.0, 0.0]);
    indexes.insert_atomic(3, "mem-1", &embedding, 0.5).unwrap();
    indexes.insert_atomic(3, "mem-1", &embedding, 0.5).unwrap();

    assert!(indexes.contains(3));
    assert_eq!(indexes.len(), 1);
    assert_eq!(indexes.resolve_node_id(3), Some("mem-1"));
}

#[test]
fn insert_atomic_rolls_back_when_a_graph_fails() {
    let mut indexes = IndexSet::new(test_params).unwrap();
    // context/action match the index dimension (4); result has the wrong
    // dimension, so the third graph insert fails after the first two succeed.
    let embedding = ThreeFieldEmbedding {
        context: vec![1.0, 0.0, 0.0, 0.0],
        action: vec![0.0, 1.0, 0.0, 0.0],
        result: vec![0.0, 0.0, 1.0],
    };

    let result = indexes.insert_atomic(5, "mem-1", &embedding, 0.5);
    match result {
        Err(CoreError::Hnsw(_)) => {}
        other => panic!("expected wrapped Hnsw error, got {other:?}"),
    }

    // Without rollback, context_index would still hold id 5 (contains() and
    // is_empty() inspect the context graph).
    assert!(!indexes.contains(5));
    assert!(indexes.is_empty());
    assert_eq!(indexes.resolve_node_id(5), None);

    // The merged search also covers the action graph: a query matching the
    // action vector must return nothing, proving action_index was rolled back.
    let action_hits = indexes.search(&[0.0, 1.0, 0.0, 0.0], 5).unwrap();
    assert!(action_hits.is_empty());
}
