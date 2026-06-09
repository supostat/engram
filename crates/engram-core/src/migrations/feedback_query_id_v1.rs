//! Add the `query_id` column to `feedback_tracking` on databases created before
//! routing instrumentation. Idempotent and gated by `schema_meta`.

use engram_storage::Database;

use crate::error::CoreError;

pub const FEEDBACK_QUERY_ID_KEY: &str = "feedback_query_id";
pub const FEEDBACK_QUERY_ID_TARGET: &str = "v1";

pub fn run(database: &Database) -> Result<bool, CoreError> {
    if super::read_meta(database, FEEDBACK_QUERY_ID_KEY)?.as_deref()
        == Some(FEEDBACK_QUERY_ID_TARGET)
    {
        return Ok(false);
    }
    if !column_exists(database, "feedback_tracking", "query_id")? {
        let connection = database.connection();
        let transaction = connection
            .unchecked_transaction()
            .map_err(|error| CoreError::MigrationFailed(format!("tx begin: {error}")))?;
        transaction
            .execute_batch("ALTER TABLE feedback_tracking ADD COLUMN query_id TEXT")
            .map_err(|error| CoreError::MigrationFailed(format!("alter: {error}")))?;
        transaction
            .commit()
            .map_err(|error| CoreError::MigrationFailed(format!("tx commit: {error}")))?;
    }
    super::write_meta(database, FEEDBACK_QUERY_ID_KEY, FEEDBACK_QUERY_ID_TARGET)?;
    Ok(true)
}

fn column_exists(database: &Database, table: &str, column: &str) -> Result<bool, CoreError> {
    let mut statement = database
        .connection()
        .prepare(&format!("PRAGMA table_info({table})"))
        .map_err(|error| CoreError::MigrationFailed(format!("prepare table_info: {error}")))?;
    let names = statement
        .query_map([], |row| row.get::<_, String>(1))
        .map_err(|error| CoreError::MigrationFailed(format!("query table_info: {error}")))?;
    for name in names {
        let name =
            name.map_err(|error| CoreError::MigrationFailed(format!("row table_info: {error}")))?;
        if name == column {
            return Ok(true);
        }
    }
    Ok(false)
}
