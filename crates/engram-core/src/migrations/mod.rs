//! Idempotent startup migrations gated by `schema_meta(key, value)`.

pub mod embedding_model_v1;
pub mod feedback_query_id_v1;
pub mod fts_tokenizer_v1;
pub mod tags_format_v1;

use engram_storage::Database;

use crate::error::CoreError;

pub use embedding_model_v1::EMBEDDING_MODEL_KEY;
pub use feedback_query_id_v1::{FEEDBACK_QUERY_ID_KEY, FEEDBACK_QUERY_ID_TARGET};
pub use fts_tokenizer_v1::{FTS_TOKENIZER_KEY, FTS_TOKENIZER_TARGET};
pub use tags_format_v1::{TAGS_FORMAT_KEY, TAGS_FORMAT_TARGET_VALUE, TagsFormatV1Stats};

const ENV_DRY_RUN: &str = "ENGRAM_MIGRATIONS_DRY_RUN";
const ENV_TAGS_STRICT: &str = "ENGRAM_TAGS_MIGRATION_STRICT";

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct MigrationReport {
    pub tags_format_v1: Option<TagsFormatV1Stats>,
    pub feedback_query_id_v1_applied: bool,
    pub fts_tokenizer_v1_applied: bool,
}

pub fn run_pending(database: &Database) -> Result<MigrationReport, CoreError> {
    let dry_run = std::env::var(ENV_DRY_RUN).is_ok();
    let strict = std::env::var(ENV_TAGS_STRICT).is_ok();
    let tags_format_v1 = tags_format_v1::run(database, dry_run, strict)?;
    let feedback_query_id_v1_applied = feedback_query_id_v1::run(database)?;
    let fts_tokenizer_v1_applied = fts_tokenizer_v1::run(database)?;
    Ok(MigrationReport {
        tags_format_v1,
        feedback_query_id_v1_applied,
        fts_tokenizer_v1_applied,
    })
}

pub(crate) fn read_meta(database: &Database, key: &str) -> Result<Option<String>, CoreError> {
    use rusqlite::OptionalExtension;
    database
        .connection()
        .query_row(
            "SELECT value FROM schema_meta WHERE key = ?1",
            [key],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(|error| CoreError::MigrationFailed(format!("schema_meta read: {error}")))
}

pub(crate) fn write_meta(database: &Database, key: &str, value: &str) -> Result<(), CoreError> {
    database
        .connection()
        .execute(
            "INSERT OR REPLACE INTO schema_meta(key, value) VALUES (?1, ?2)",
            [key, value],
        )
        .map_err(|error| CoreError::MigrationFailed(format!("schema_meta write: {error}")))?;
    Ok(())
}
