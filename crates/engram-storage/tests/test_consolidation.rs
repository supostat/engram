use engram_storage::Database;

#[test]
fn test_log_consolidation() {
    let database = Database::in_memory().unwrap();
    let memory_ids = vec!["m1".to_string(), "m2".to_string()];

    database
        .log_consolidation(
            "c1",
            "merge",
            &memory_ids,
            Some("similar memories"),
            "consolidation_engine",
            "2026-01-01T00:00:00Z",
        )
        .unwrap();

    let row: (String, String, String, Option<String>, String, String) = database
        .connection()
        .query_row(
            "SELECT id, action, memory_ids, reason, performed_at, performed_by
             FROM consolidation_log WHERE id = ?1",
            ["c1"],
            |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                ))
            },
        )
        .unwrap();

    assert_eq!(row.0, "c1");
    assert_eq!(row.1, "merge");
    assert_eq!(row.2, r#"["m1","m2"]"#);
    assert_eq!(row.3.as_deref(), Some("similar memories"));
    assert_eq!(row.4, "2026-01-01T00:00:00Z");
    assert_eq!(row.5, "consolidation_engine");
}

#[test]
fn test_track_and_judge() {
    let database = Database::in_memory().unwrap();

    // Insert a memory first (foreign key)
    database
        .connection()
        .execute(
            "INSERT INTO memories (id, memory_type, context, action, result, created_at, updated_at)
             VALUES ('m1', 'decision', 'ctx', 'act', 'res', '2026-01-01', '2026-01-01')",
            [],
        )
        .unwrap();

    database.track_search("m1", "2026-01-01T00:00:00Z").unwrap();

    let pending = database.get_pending_judgments(10).unwrap();
    assert_eq!(pending, vec!["m1"]);

    database.mark_judged("m1", "2026-01-01T01:00:00Z").unwrap();

    let pending_after = database.get_pending_judgments(10).unwrap();
    assert!(pending_after.is_empty());
}

#[test]
fn test_pending_judgments_limit() {
    let database = Database::in_memory().unwrap();

    for i in 0..5 {
        let id = format!("m{i}");
        database
            .connection()
            .execute(
                "INSERT INTO memories (id, memory_type, context, action, result, created_at, updated_at)
                 VALUES (?1, 'decision', 'ctx', 'act', 'res', '2026-01-01', '2026-01-01')",
                [&id],
            )
            .unwrap();
        database.track_search(&id, "2026-01-01T00:00:00Z").unwrap();
    }

    let pending = database.get_pending_judgments(3).unwrap();
    assert_eq!(pending.len(), 3);
}

#[test]
fn test_track_search_invalid_memory_id() {
    let database = Database::in_memory().unwrap();
    let result = database.track_search("nonexistent_memory", "2026-01-01T00:00:00Z");
    assert!(result.is_err(), "FK violation must produce an error");
}

#[test]
fn test_mark_judged_nonexistent() {
    let database = Database::in_memory().unwrap();
    // mark_judged on non-existent record should succeed silently (UPDATE WHERE matches 0 rows)
    database
        .mark_judged("nonexistent_memory", "2026-01-01T00:00:00Z")
        .unwrap();
}
