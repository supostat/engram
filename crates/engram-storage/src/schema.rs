use crate::error::StorageError;

pub const CREATE_MEMORIES: &str = r#"
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    memory_type TEXT NOT NULL CHECK(memory_type IN ('decision','pattern','bugfix','context','antipattern','insight')),
    context TEXT NOT NULL,
    action TEXT NOT NULL,
    result TEXT NOT NULL,
    score REAL DEFAULT 0.0,
    embedding_context BLOB,
    embedding_action BLOB,
    embedding_result BLOB,
    indexed BOOLEAN DEFAULT FALSE,
    tags TEXT,
    project TEXT,
    parent_id TEXT,
    source_ids TEXT,
    insight_type TEXT CHECK(insight_type IS NULL OR insight_type IN ('cluster','temporal','causal')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    used_count INTEGER DEFAULT 0,
    last_used_at TEXT,
    superseded_by TEXT,
    FOREIGN KEY (superseded_by) REFERENCES memories(id),
    FOREIGN KEY (parent_id) REFERENCES memories(id)
)
"#;

pub const CREATE_MEMORIES_FTS: &str = r#"
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    context, action, result,
    content='memories',
    content_rowid='rowid'
)
"#;

pub const CREATE_FTS_INSERT_TRIGGER: &str = r#"
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, context, action, result)
    VALUES (new.rowid, new.context, new.action, new.result);
END
"#;

pub const CREATE_FTS_DELETE_TRIGGER: &str = r#"
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, context, action, result)
    VALUES ('delete', old.rowid, old.context, old.action, old.result);
END
"#;

pub const CREATE_FTS_UPDATE_TRIGGER: &str = r#"
CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, context, action, result)
    VALUES ('delete', old.rowid, old.context, old.action, old.result);
    INSERT INTO memories_fts(rowid, context, action, result)
    VALUES (new.rowid, new.context, new.action, new.result);
END
"#;

pub const CREATE_Q_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS q_table (
    router_level TEXT NOT NULL,
    state TEXT NOT NULL,
    action TEXT NOT NULL,
    value REAL DEFAULT 0.0,
    update_count INTEGER DEFAULT 0,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (router_level, state, action)
)
"#;

pub const CREATE_CONSOLIDATION_LOG: &str = r#"
CREATE TABLE IF NOT EXISTS consolidation_log (
    id TEXT PRIMARY KEY,
    action TEXT NOT NULL,
    memory_ids TEXT NOT NULL,
    reason TEXT,
    performed_at TEXT NOT NULL,
    performed_by TEXT NOT NULL
)
"#;

pub const CREATE_FEEDBACK_TRACKING: &str = r#"
CREATE TABLE IF NOT EXISTS feedback_tracking (
    memory_id TEXT NOT NULL,
    searched_at TEXT NOT NULL,
    judged BOOLEAN DEFAULT FALSE,
    judged_at TEXT,
    FOREIGN KEY (memory_id) REFERENCES memories(id)
)
"#;

pub const CREATE_RECOMMENDATIONS: &str = r#"
CREATE TABLE IF NOT EXISTS recommendations (
    id TEXT PRIMARY KEY,
    key TEXT NOT NULL,
    current_value TEXT,
    suggested_value TEXT NOT NULL,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL,
    status TEXT DEFAULT 'pending' CHECK(status IN ('pending','accepted','rejected'))
)
"#;

pub const CREATE_METRICS: &str = r#"
CREATE TABLE IF NOT EXISTS metrics (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    value REAL NOT NULL,
    period_start TEXT NOT NULL,
    period_end TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"#;

const SCHEMA_STATEMENTS: &[&str] = &[
    CREATE_MEMORIES,
    CREATE_MEMORIES_FTS,
    CREATE_FTS_INSERT_TRIGGER,
    CREATE_FTS_DELETE_TRIGGER,
    CREATE_FTS_UPDATE_TRIGGER,
    CREATE_Q_TABLE,
    CREATE_CONSOLIDATION_LOG,
    CREATE_FEEDBACK_TRACKING,
    CREATE_RECOMMENDATIONS,
    CREATE_METRICS,
];

pub fn apply_schema(connection: &rusqlite::Connection) -> Result<(), StorageError> {
    connection.execute_batch("PRAGMA journal_mode = WAL;")?;
    connection.execute_batch("PRAGMA foreign_keys = ON;")?;
    for statement in SCHEMA_STATEMENTS {
        connection.execute_batch(statement)?;
    }
    Ok(())
}
