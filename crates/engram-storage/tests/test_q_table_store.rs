use engram_storage::Database;

#[test]
fn test_upsert_and_get() {
    let database = Database::in_memory().unwrap();
    database
        .upsert_q_value(
            "router_l1",
            "state_a",
            "action_x",
            0.75,
            "2026-01-01T00:00:00Z",
        )
        .unwrap();

    let value = database
        .get_q_value("router_l1", "state_a", "action_x")
        .unwrap();
    assert!((value - 0.75).abs() < f32::EPSILON);
}

#[test]
fn test_upsert_overwrites() {
    let database = Database::in_memory().unwrap();
    database
        .upsert_q_value(
            "router_l1",
            "state_a",
            "action_x",
            0.5,
            "2026-01-01T00:00:00Z",
        )
        .unwrap();
    database
        .upsert_q_value(
            "router_l1",
            "state_a",
            "action_x",
            0.9,
            "2026-01-02T00:00:00Z",
        )
        .unwrap();

    let value = database
        .get_q_value("router_l1", "state_a", "action_x")
        .unwrap();
    assert!((value - 0.9).abs() < f32::EPSILON);
}

#[test]
fn test_get_default() {
    let database = Database::in_memory().unwrap();
    let value = database
        .get_q_value("unknown", "unknown", "unknown")
        .unwrap();
    assert!((value - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_load_q_table() {
    let database = Database::in_memory().unwrap();
    database
        .upsert_q_value("level_a", "s1", "a1", 0.1, "2026-01-01T00:00:00Z")
        .unwrap();
    database
        .upsert_q_value("level_a", "s1", "a2", 0.2, "2026-01-01T00:00:00Z")
        .unwrap();
    database
        .upsert_q_value("level_a", "s2", "a1", 0.3, "2026-01-01T00:00:00Z")
        .unwrap();
    database
        .upsert_q_value("level_b", "s1", "a1", 0.9, "2026-01-01T00:00:00Z")
        .unwrap();

    let entries = database.load_q_table("level_a").unwrap();
    assert_eq!(entries.len(), 3);

    let level_b_entries = database.load_q_table("level_b").unwrap();
    assert_eq!(level_b_entries.len(), 1);
    assert_eq!(level_b_entries[0].0, "s1");
    assert_eq!(level_b_entries[0].1, "a1");
    assert!((level_b_entries[0].2 - 0.9).abs() < f32::EPSILON);
    assert!(level_b_entries[0].3 > 0, "update_count must be > 0");

    // level_a entries were each upserted once, so update_count must be > 0
    for entry in &entries {
        assert!(entry.3 > 0, "update_count must be > 0 for {:?}", entry);
    }
}

#[test]
fn test_load_q_table_empty() {
    let database = Database::in_memory().unwrap();
    let entries = database.load_q_table("nonexistent_level").unwrap();
    assert!(entries.is_empty());
}
