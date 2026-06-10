//! Re-create `memories_fts` under `porter unicode61` on pre-tokenizer databases,
//! then rebuild the index from the content table. Idempotent, schema_meta-gated.
//!
//! The recreate SQL is pinned here (copied, not referencing
//! `schema::CREATE_MEMORIES_FTS`) so a future schema edit cannot retroactively
//! change this migration's behavior.

use engram_storage::Database;

use crate::error::CoreError;

pub const FTS_TOKENIZER_KEY: &str = "fts_tokenizer";
pub const FTS_TOKENIZER_TARGET: &str = "porter_unicode61_v1";

pub fn run(database: &Database) -> Result<bool, CoreError> {
    if super::read_meta(database, FTS_TOKENIZER_KEY)?.as_deref() == Some(FTS_TOKENIZER_TARGET) {
        return Ok(false);
    }
    let connection = database.connection();
    let transaction = connection
        .unchecked_transaction()
        .map_err(|error| CoreError::MigrationFailed(format!("fts tx begin: {error}")))?;
    transaction
        .execute_batch(
            "DROP TABLE IF EXISTS memories_fts;
             CREATE VIRTUAL TABLE memories_fts USING fts5(
                 context, action, result,
                 content='memories', content_rowid='rowid',
                 tokenize='porter unicode61');
             INSERT INTO memories_fts(memories_fts) VALUES('rebuild');",
        )
        .map_err(|error| CoreError::MigrationFailed(format!("fts recreate/rebuild: {error}")))?;
    transaction
        .commit()
        .map_err(|error| CoreError::MigrationFailed(format!("fts tx commit: {error}")))?;
    super::write_meta(database, FTS_TOKENIZER_KEY, FTS_TOKENIZER_TARGET)?;
    Ok(true)
}
