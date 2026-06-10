use engram_core::migrations::{self, FTS_TOKENIZER_KEY, FTS_TOKENIZER_TARGET, fts_tokenizer_v1};
use engram_storage::{Database, Memory};

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

fn seed(database: &Database, id: &str, context: &str) {
    let memory = Memory {
        id: id.to_string(),
        memory_type: "decision".to_string(),
        context: context.to_string(),
        action: "act".to_string(),
        result: "res".to_string(),
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

/// Rebuilds `memories_fts` WITHOUT the porter tokenizer to emulate a database
/// created before the tokenizer migration. The trigger-maintained external-content
/// index is repopulated from the existing `memories` rows via 'rebuild'.
fn downgrade_to_unicode61(database: &Database) {
    database
        .connection()
        .execute_batch(
            "DROP TABLE IF EXISTS memories_fts;
             CREATE VIRTUAL TABLE memories_fts USING fts5(
                 context, action, result,
                 content='memories', content_rowid='rowid');
             INSERT INTO memories_fts(memories_fts) VALUES('rebuild');",
        )
        .expect("rebuild legacy unicode61 fts");
}

fn seed_multi_row_legacy(database: &Database) {
    // Multi-row fixture: only one row stores the base form "run"; the others are
    // noise so the 'rebuild' re-tokenization is exercised across several documents.
    // The query side uses an inflection ("running") that shares a porter stem with
    // "run" but is NOT a prefix of it — so the prefix-`*` sanitizer alone cannot
    // bridge the gap on a plain unicode61 index. Only porter stemming recovers it.
    seed(database, "m1", "schedule a nightly run");
    seed(database, "m2", "unrelated refund topic");
    seed(database, "m3", "database migration notes");
    seed(database, "m4", "queue retry backoff");
    downgrade_to_unicode61(database);
}

#[test]
fn legacy_unicode61_misses_stem_until_migration() {
    let database = Database::in_memory().expect("in-memory db");
    seed_multi_row_legacy(&database);

    // Under plain unicode61, "running" does NOT recall the "run" row: it is not a
    // prefix of "run" and there is no stemming.
    let before = database.search_fts("running", 10).expect("search before");
    assert!(
        before.is_empty(),
        "legacy unicode61 must miss the cross-inflection stem"
    );

    let applied = fts_tokenizer_v1::run(&database).expect("migration runs");
    assert!(applied, "migration must report it ran on a legacy db");

    let after = database.search_fts("running", 10).expect("search after");
    assert_eq!(after.len(), 1, "porter rebuild must recall the stemmed row");
    assert_eq!(after[0].memory.id, "m1");

    assert_eq!(
        read_meta(&database, FTS_TOKENIZER_KEY).as_deref(),
        Some(FTS_TOKENIZER_TARGET),
        "migration must record the meta flag"
    );
}

#[test]
fn second_run_is_idempotent_noop() {
    let database = Database::in_memory().expect("in-memory db");
    seed_multi_row_legacy(&database);

    let first = fts_tokenizer_v1::run(&database).expect("first run");
    assert!(first, "first run applies");

    let second = fts_tokenizer_v1::run(&database).expect("second run");
    assert!(
        !second,
        "second run must be a no-op (meta already at target)"
    );

    // The index still resolves the stem after the no-op second run.
    let after = database.search_fts("running", 10).expect("search after");
    assert_eq!(after.len(), 1);
}

#[test]
fn fresh_database_stems_without_migration() {
    // A fresh schema already builds memories_fts under porter; stemming works before
    // any migration runs.
    let database = Database::in_memory().expect("in-memory db");
    seed(&database, "m1", "schedule a nightly run");

    let results = database.search_fts("running", 10).expect("search fresh");
    assert_eq!(results.len(), 1, "fresh porter index must stem natively");
    assert_eq!(results[0].memory.id, "m1");
}

#[test]
fn run_pending_reports_fts_tokenizer_applied() {
    let database = Database::in_memory().expect("in-memory db");
    seed_multi_row_legacy(&database);

    let report = migrations::run_pending(&database).expect("run_pending");
    assert!(
        report.fts_tokenizer_v1_applied,
        "run_pending must apply the fts tokenizer migration on a legacy db"
    );

    let after = database.search_fts("running", 10).expect("search after");
    assert_eq!(after.len(), 1);
}
