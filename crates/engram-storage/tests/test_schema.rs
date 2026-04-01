use engram_storage::Database;

fn table_exists(database: &Database, table_name: &str) -> bool {
    let count: i64 = database
        .connection()
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type IN ('table','view') AND name = ?1",
            [table_name],
            |row| row.get(0),
        )
        .unwrap();
    count == 1
}

#[test]
fn test_database_creates_all_tables() {
    let database = Database::in_memory().unwrap();
    let expected_tables = [
        "memories",
        "memories_fts",
        "q_table",
        "consolidation_log",
        "feedback_tracking",
        "recommendations",
        "metrics",
    ];
    for table in &expected_tables {
        assert!(
            table_exists(&database, table),
            "table '{table}' should exist"
        );
    }
}

#[test]
fn test_database_wal_mode() {
    let temporary_file =
        std::env::temp_dir().join(format!("engram_test_wal_{}.db", std::process::id()));
    let path = temporary_file.to_str().unwrap();
    let database = Database::open(path).unwrap();

    let journal_mode: String = database
        .connection()
        .query_row("PRAGMA journal_mode", [], |row| row.get(0))
        .unwrap();

    assert_eq!(journal_mode, "wal");

    drop(database);
    let _ = std::fs::remove_file(&temporary_file);
    let _ = std::fs::remove_file(temporary_file.with_extension("db-wal"));
    let _ = std::fs::remove_file(temporary_file.with_extension("db-shm"));
}

#[test]
fn test_database_foreign_keys_enabled() {
    let database = Database::in_memory().unwrap();
    let foreign_keys: i64 = database
        .connection()
        .query_row("PRAGMA foreign_keys", [], |row| row.get(0))
        .unwrap();
    assert_eq!(foreign_keys, 1);
}

#[test]
fn test_memories_type_check() {
    let database = Database::in_memory().unwrap();
    let result = database.connection().execute(
        "INSERT INTO memories (id, memory_type, context, action, result, created_at, updated_at)
         VALUES ('m1', 'invalid_type', 'ctx', 'act', 'res', '2026-01-01', '2026-01-01')",
        [],
    );
    assert!(result.is_err(), "invalid memory_type should be rejected");
}

#[test]
fn test_memories_insight_type_check() {
    let database = Database::in_memory().unwrap();
    let result = database.connection().execute(
        "INSERT INTO memories (id, memory_type, context, action, result, insight_type, created_at, updated_at)
         VALUES ('m1', 'decision', 'ctx', 'act', 'res', 'bad_type', '2026-01-01', '2026-01-01')",
        [],
    );
    assert!(result.is_err(), "invalid insight_type should be rejected");
}

#[test]
fn test_fts_trigger_insert() {
    let database = Database::in_memory().unwrap();
    database
        .connection()
        .execute(
            "INSERT INTO memories (id, memory_type, context, action, result, created_at, updated_at)
             VALUES ('m1', 'decision', 'test context', 'test action', 'test result', '2026-01-01', '2026-01-01')",
            [],
        )
        .unwrap();

    let fts_count: i64 = database
        .connection()
        .query_row(
            "SELECT COUNT(*) FROM memories_fts WHERE memories_fts MATCH 'test'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(fts_count, 1);
}

#[test]
fn test_fts_trigger_delete() {
    let database = Database::in_memory().unwrap();
    let connection = database.connection();

    connection
        .execute(
            "INSERT INTO memories (id, memory_type, context, action, result, created_at, updated_at)
             VALUES ('m1', 'decision', 'unique_ctx', 'unique_act', 'unique_res', '2026-01-01', '2026-01-01')",
            [],
        )
        .unwrap();
    connection
        .execute("DELETE FROM memories WHERE id = 'm1'", [])
        .unwrap();

    let fts_count: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM memories_fts WHERE memories_fts MATCH 'unique_ctx'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(fts_count, 0);
}

#[test]
fn test_fts_trigger_update() {
    let database = Database::in_memory().unwrap();
    let connection = database.connection();

    connection
        .execute(
            "INSERT INTO memories (id, memory_type, context, action, result, created_at, updated_at)
             VALUES ('m1', 'decision', 'old_context', 'old_action', 'old_result', '2026-01-01', '2026-01-01')",
            [],
        )
        .unwrap();
    connection
        .execute(
            "UPDATE memories SET context = 'new_context', action = 'new_action', result = 'new_result' WHERE id = 'm1'",
            [],
        )
        .unwrap();

    let old_count: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM memories_fts WHERE memories_fts MATCH 'old_context'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    let new_count: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM memories_fts WHERE memories_fts MATCH 'new_context'",
            [],
            |row| row.get(0),
        )
        .unwrap();

    assert_eq!(old_count, 0, "old text should not be in FTS index");
    assert_eq!(new_count, 1, "new text should be in FTS index");
}

#[test]
fn test_fts_search() {
    let database = Database::in_memory().unwrap();
    let connection = database.connection();

    connection
        .execute(
            "INSERT INTO memories (id, memory_type, context, action, result, created_at, updated_at)
             VALUES ('m1', 'decision', 'rust ownership', 'use borrow checker', 'no memory leaks', '2026-01-01', '2026-01-01')",
            [],
        )
        .unwrap();
    connection
        .execute(
            "INSERT INTO memories (id, memory_type, context, action, result, created_at, updated_at)
             VALUES ('m2', 'pattern', 'python garbage collection', 'use gc module', 'cleaned up', '2026-01-01', '2026-01-01')",
            [],
        )
        .unwrap();

    let mut statement = connection
        .prepare(
            "SELECT m.id FROM memories m
             JOIN memories_fts f ON m.rowid = f.rowid
             WHERE memories_fts MATCH 'rust'",
        )
        .unwrap();
    let ids: Vec<String> = statement
        .query_map([], |row| row.get(0))
        .unwrap()
        .map(|r| r.unwrap())
        .collect();

    assert_eq!(ids, vec!["m1"]);
}

#[test]
fn test_schema_idempotent() {
    let database = Database::in_memory().unwrap();
    let result = engram_storage::schema::apply_schema(database.connection());
    assert!(result.is_ok(), "applying schema twice should not error");
}

#[test]
fn test_recommendations_status_check() {
    let database = Database::in_memory().unwrap();
    let result = database.connection().execute(
        "INSERT INTO recommendations (id, key, suggested_value, reason, created_at, status)
         VALUES ('r1', 'k1', 'val', 'reason', '2026-01-01', 'invalid_status')",
        [],
    );
    assert!(result.is_err(), "invalid status should be rejected");
}
