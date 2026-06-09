use engram_core::migrations::{
    self, FEEDBACK_QUERY_ID_KEY, FEEDBACK_QUERY_ID_TARGET, feedback_query_id_v1,
};
use engram_storage::{Database, Memory};

fn column_exists(database: &Database, table: &str, column: &str) -> bool {
    let mut statement = database
        .connection()
        .prepare(&format!("PRAGMA table_info({table})"))
        .expect("prepare table_info");
    statement
        .query_map([], |row| row.get::<_, String>(1))
        .expect("query table_info")
        .filter_map(Result::ok)
        .any(|name| name == column)
}

fn read_meta(database: &Database, key: &str) -> Option<String> {
    use rusqlite::OptionalExtension;
    database
        .connection()
        .query_row(
            "SELECT value FROM schema_meta WHERE key = ?1",
            [key],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .expect("query meta")
}

/// Rebuilds `feedback_tracking` without the `query_id` column to emulate a
/// database created before routing instrumentation.
fn strip_query_id_column(database: &Database) {
    database
        .connection()
        .execute_batch(
            "DROP TABLE feedback_tracking; \
             CREATE TABLE feedback_tracking ( \
                 memory_id TEXT NOT NULL, \
                 searched_at TEXT NOT NULL, \
                 judged BOOLEAN DEFAULT FALSE, \
                 judged_at TEXT, \
                 FOREIGN KEY (memory_id) REFERENCES memories(id) \
             );",
        )
        .expect("rebuild legacy feedback_tracking");
}

#[test]
fn legacy_database_gains_query_id_column() {
    let database = Database::in_memory().expect("in-memory db");
    strip_query_id_column(&database);
    assert!(
        !column_exists(&database, "feedback_tracking", "query_id"),
        "precondition: legacy table lacks query_id"
    );

    let applied = feedback_query_id_v1::run(&database).expect("migration runs");
    assert!(applied, "migration must report it ran on a legacy db");
    assert!(
        column_exists(&database, "feedback_tracking", "query_id"),
        "migration must add query_id column"
    );
    assert_eq!(
        read_meta(&database, FEEDBACK_QUERY_ID_KEY).as_deref(),
        Some(FEEDBACK_QUERY_ID_TARGET)
    );
}

#[test]
fn second_run_is_idempotent_noop() {
    let database = Database::in_memory().expect("in-memory db");
    strip_query_id_column(&database);

    let first = feedback_query_id_v1::run(&database).expect("first run");
    assert!(first, "first run applies");

    let second = feedback_query_id_v1::run(&database).expect("second run");
    assert!(
        !second,
        "second run must be a no-op (meta already at target)"
    );
    assert_eq!(
        read_meta(&database, FEEDBACK_QUERY_ID_KEY).as_deref(),
        Some(FEEDBACK_QUERY_ID_TARGET)
    );
}

#[test]
fn fresh_database_skips_alter_and_records_meta() {
    // A fresh schema already has query_id from the schema const; the migration
    // must not error, must skip the ALTER, and must record meta.
    let database = Database::in_memory().expect("in-memory db");
    assert!(
        column_exists(&database, "feedback_tracking", "query_id"),
        "precondition: fresh schema already has query_id"
    );
    assert!(
        read_meta(&database, FEEDBACK_QUERY_ID_KEY).is_none(),
        "precondition: meta not yet written"
    );

    let applied = feedback_query_id_v1::run(&database).expect("migration runs on fresh db");
    assert!(
        applied,
        "first run on a fresh db still writes meta and returns true"
    );
    assert!(column_exists(&database, "feedback_tracking", "query_id"));
    assert_eq!(
        read_meta(&database, FEEDBACK_QUERY_ID_KEY).as_deref(),
        Some(FEEDBACK_QUERY_ID_TARGET)
    );
}

fn seed_memory(database: &Database, id: &str) {
    let memory = Memory {
        id: id.to_string(),
        memory_type: "decision".to_string(),
        context: "context".to_string(),
        action: "action".to_string(),
        result: "result".to_string(),
        score: 0.0,
        embedding_context: None,
        embedding_action: None,
        embedding_result: None,
        indexed: false,
        tags: None,
        project: None,
        parent_id: None,
        source_ids: None,
        insight_type: None,
        created_at: "2026-05-01T00:00:00Z".to_string(),
        updated_at: "2026-05-01T00:00:00Z".to_string(),
        used_count: 0,
        last_used_at: None,
        superseded_by: None,
    };
    database.insert_memory(&memory).expect("seed memory");
}

#[test]
fn track_search_round_trips_query_id_after_migration() {
    // The migrated column must accept a real instrumented insert: the
    // production `track_search` path writes a non-null query_id, and the value
    // must round-trip. This proves the ALTER produced the right type/nullability.
    let database = Database::in_memory().expect("in-memory db");
    strip_query_id_column(&database);
    seed_memory(&database, "memory-1");

    let applied = feedback_query_id_v1::run(&database).expect("migration runs");
    assert!(applied, "migration must report it ran on a legacy db");

    database
        .track_search("memory-1", "query-42", "2026-05-01T00:00:01Z")
        .expect("track_search must succeed against the migrated column");

    let stored: Option<String> = database
        .connection()
        .query_row(
            "SELECT query_id FROM feedback_tracking WHERE memory_id = ?1",
            ["memory-1"],
            |row| row.get(0),
        )
        .expect("read back tracked row");
    assert_eq!(
        stored.as_deref(),
        Some("query-42"),
        "the migrated query_id column must round-trip a real track_search insert"
    );
}

#[test]
fn run_pending_reports_feedback_query_id_applied() {
    let database = Database::in_memory().expect("in-memory db");
    strip_query_id_column(&database);
    let report = migrations::run_pending(&database).expect("run_pending");
    assert!(
        report.feedback_query_id_v1_applied,
        "run_pending must apply the feedback query_id migration on a legacy db"
    );
    assert!(column_exists(&database, "feedback_tracking", "query_id"));
}
