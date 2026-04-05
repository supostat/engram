use engram_core::{CoreError, IndexSet};
use engram_hnsw::HnswParams;

fn test_params() -> Result<HnswParams, CoreError> {
    HnswParams::new(4)?
        .with_max_connections(4)?
        .with_ef_construction(16)?
        .with_ef_search(8)
        .map_err(CoreError::Hnsw)
}

#[test]
fn save_and_load_from_disk() {
    let temp_dir = tempfile::tempdir().unwrap();
    let dir_path = temp_dir.path().to_str().unwrap();

    let mut indexes = IndexSet::new(test_params).unwrap();
    let embedding = engram_embeddings::ThreeFieldEmbedding {
        context: vec![1.0, 0.0, 0.0, 0.0],
        action: vec![0.0, 1.0, 0.0, 0.0],
        result: vec![0.0, 0.0, 1.0, 0.0],
    };
    indexes.insert(42, "mem-42", &embedding, 0.5).unwrap();

    engram_core::persistence::save_to_disk(dir_path, &indexes).unwrap();

    let index_file = temp_dir.path().join("indexes.hnsw");
    assert!(index_file.exists());
    assert!(index_file.metadata().unwrap().len() > 0);
}

#[test]
fn save_creates_directory_if_missing() {
    let temp_dir = tempfile::tempdir().unwrap();
    let nested_path = temp_dir.path().join("nested/deep/dir");
    let dir_path = nested_path.to_str().unwrap();

    let indexes = IndexSet::new(test_params).unwrap();
    engram_core::persistence::save_to_disk(dir_path, &indexes).unwrap();
    assert!(nested_path.join("indexes.hnsw").exists());
}

#[test]
fn load_or_rebuild_with_empty_db_returns_empty_indexes() {
    let temp_dir = tempfile::tempdir().unwrap();
    let dir_path = temp_dir.path().to_str().unwrap();
    let database = engram_storage::Database::in_memory().unwrap();

    let indexes =
        engram_core::persistence::load_or_rebuild(dir_path, &database, test_params).unwrap();
    assert!(indexes.is_empty());
}

#[test]
fn load_or_rebuild_prefers_existing_file() {
    let temp_dir = tempfile::tempdir().unwrap();
    let dir_path = temp_dir.path().to_str().unwrap();

    let mut indexes = IndexSet::new(test_params).unwrap();
    let embedding = engram_embeddings::ThreeFieldEmbedding {
        context: vec![1.0, 0.0, 0.0, 0.0],
        action: vec![0.0, 1.0, 0.0, 0.0],
        result: vec![0.0, 0.0, 1.0, 0.0],
    };
    indexes.insert(99, "mem-99", &embedding, 0.4).unwrap();
    engram_core::persistence::save_to_disk(dir_path, &indexes).unwrap();

    let database = engram_storage::Database::in_memory().unwrap();
    let loaded =
        engram_core::persistence::load_or_rebuild(dir_path, &database, test_params).unwrap();
    assert_eq!(loaded.len(), 1);
    assert!(loaded.contains(99));
}

#[test]
fn load_or_rebuild_with_corrupted_file_rebuilds() {
    let temp_dir = tempfile::tempdir().unwrap();
    let dir_path = temp_dir.path().to_str().unwrap();

    let corrupted_path = temp_dir.path().join("indexes.hnsw");
    std::fs::write(&corrupted_path, b"corrupted garbage bytes").unwrap();

    let database = engram_storage::Database::in_memory().unwrap();
    let indexes =
        engram_core::persistence::load_or_rebuild(dir_path, &database, test_params).unwrap();
    assert!(indexes.is_empty());
}
