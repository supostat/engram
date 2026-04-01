use engram_storage::memory::Memory;
use engram_storage::{Database, StorageError};

fn make_memory(id: &str) -> Memory {
    Memory {
        id: id.to_string(),
        memory_type: "decision".to_string(),
        context: format!("context for {id}"),
        action: format!("action for {id}"),
        result: format!("result for {id}"),
        score: 0.5,
        embedding_context: None,
        embedding_action: None,
        embedding_result: None,
        indexed: false,
        tags: None,
        project: None,
        parent_id: None,
        source_ids: None,
        insight_type: None,
        created_at: "2026-01-01T00:00:00Z".to_string(),
        updated_at: "2026-01-01T00:00:00Z".to_string(),
        used_count: 0,
        last_used_at: None,
        superseded_by: None,
    }
}

#[test]
fn test_insert_and_get() {
    let database = Database::in_memory().unwrap();
    let memory = make_memory("m1");
    database.insert_memory(&memory).unwrap();

    let loaded = database.get_memory("m1").unwrap();
    assert_eq!(loaded.id, "m1");
    assert_eq!(loaded.memory_type, "decision");
    assert_eq!(loaded.context, "context for m1");
    assert_eq!(loaded.action, "action for m1");
    assert_eq!(loaded.result, "result for m1");
    assert!((loaded.score - 0.5).abs() < f32::EPSILON);
    assert!(!loaded.indexed);
    assert_eq!(loaded.used_count, 0);
}

#[test]
fn test_insert_duplicate() {
    let database = Database::in_memory().unwrap();
    let memory = make_memory("m1");
    database.insert_memory(&memory).unwrap();

    let result = database.insert_memory(&memory);
    assert!(matches!(result, Err(StorageError::DuplicateKey(_))));
}

#[test]
fn test_get_not_found() {
    let database = Database::in_memory().unwrap();
    let result = database.get_memory("nonexistent");
    assert!(matches!(result, Err(StorageError::NotFound(_))));
}

#[test]
fn test_delete_memory() {
    let database = Database::in_memory().unwrap();
    database.insert_memory(&make_memory("m1")).unwrap();
    database.delete_memory("m1").unwrap();

    let result = database.get_memory("m1");
    assert!(matches!(result, Err(StorageError::NotFound(_))));
}

#[test]
fn test_delete_not_found() {
    let database = Database::in_memory().unwrap();
    let result = database.delete_memory("nonexistent");
    assert!(matches!(result, Err(StorageError::NotFound(_))));
}

#[test]
fn test_set_indexed() {
    let database = Database::in_memory().unwrap();
    database.insert_memory(&make_memory("m1")).unwrap();
    assert!(!database.get_memory("m1").unwrap().indexed);

    database.set_memory_indexed("m1", true).unwrap();
    assert!(database.get_memory("m1").unwrap().indexed);
}

#[test]
fn test_set_score() {
    let database = Database::in_memory().unwrap();
    database.insert_memory(&make_memory("m1")).unwrap();

    database.set_memory_score("m1", 0.95).unwrap();
    let loaded = database.get_memory("m1").unwrap();
    assert!((loaded.score - 0.95).abs() < f32::EPSILON);
}

#[test]
fn test_touch_memory() {
    let database = Database::in_memory().unwrap();
    database.insert_memory(&make_memory("m1")).unwrap();

    database.touch_memory("m1", "2026-03-15T10:00:00Z").unwrap();
    let loaded = database.get_memory("m1").unwrap();
    assert_eq!(loaded.used_count, 1);
    assert_eq!(loaded.last_used_at.as_deref(), Some("2026-03-15T10:00:00Z"));

    database.touch_memory("m1", "2026-03-15T11:00:00Z").unwrap();
    let loaded = database.get_memory("m1").unwrap();
    assert_eq!(loaded.used_count, 2);
    assert_eq!(loaded.last_used_at.as_deref(), Some("2026-03-15T11:00:00Z"));
}

#[test]
fn test_bulk_insert() {
    let database = Database::in_memory().unwrap();
    let memories: Vec<Memory> = (0..100).map(|i| make_memory(&format!("m{i}"))).collect();

    let inserted = database.bulk_insert_memories(&memories).unwrap();
    assert_eq!(inserted, 100);

    for i in 0..100 {
        let loaded = database.get_memory(&format!("m{i}")).unwrap();
        assert_eq!(loaded.id, format!("m{i}"));
    }
}

#[test]
fn test_get_unindexed() {
    let database = Database::in_memory().unwrap();

    for i in 0..5 {
        let mut memory = make_memory(&format!("m{i}"));
        memory.indexed = i >= 3; // m0, m1, m2 = unindexed; m3, m4 = indexed
        database.insert_memory(&memory).unwrap();
    }

    let unindexed = database.get_unindexed_memories(10).unwrap();
    assert_eq!(unindexed.len(), 3);
    for memory in &unindexed {
        assert!(!memory.indexed);
    }
}

#[test]
fn test_set_indexed_not_found() {
    let database = Database::in_memory().unwrap();
    let result = database.set_memory_indexed("nonexistent", true);
    assert!(matches!(result, Err(StorageError::NotFound(_))));
}

#[test]
fn test_set_score_not_found() {
    let database = Database::in_memory().unwrap();
    let result = database.set_memory_score("nonexistent", 0.9);
    assert!(matches!(result, Err(StorageError::NotFound(_))));
}

#[test]
fn test_touch_memory_not_found() {
    let database = Database::in_memory().unwrap();
    let result = database.touch_memory("nonexistent", "2026-01-01T00:00:00Z");
    assert!(matches!(result, Err(StorageError::NotFound(_))));
}

#[test]
fn test_bulk_insert_with_duplicates() {
    let database = Database::in_memory().unwrap();
    database.insert_memory(&make_memory("m1")).unwrap();

    let memories = vec![make_memory("m2"), make_memory("m1")];
    let result = database.bulk_insert_memories(&memories);
    assert!(result.is_err(), "bulk insert with duplicate IDs must fail");

    // Transaction must have rolled back — m2 should not exist
    let m2_result = database.get_memory("m2");
    assert!(
        matches!(m2_result, Err(StorageError::NotFound(_))),
        "m2 must not exist after rollback"
    );
}
