//! Idempotent startup migrations gated by `schema_meta(key, value)`.

pub mod tags_format_v1;

use engram_storage::Database;

use crate::error::CoreError;

pub use tags_format_v1::{TAGS_FORMAT_KEY, TAGS_FORMAT_TARGET_VALUE, TagsFormatV1Stats};

const ENV_DRY_RUN: &str = "ENGRAM_MIGRATIONS_DRY_RUN";
const ENV_TAGS_STRICT: &str = "ENGRAM_TAGS_MIGRATION_STRICT";

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct MigrationReport {
    pub tags_format_v1: Option<TagsFormatV1Stats>,
}

pub fn run_pending(database: &Database) -> Result<MigrationReport, CoreError> {
    let dry_run = std::env::var(ENV_DRY_RUN).is_ok();
    let strict = std::env::var(ENV_TAGS_STRICT).is_ok();
    let tags_format_v1 = tags_format_v1::run(database, dry_run, strict)?;
    Ok(MigrationReport { tags_format_v1 })
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
