//! Guard against running the daemon with an embedding model different from
//! the one used to compute the stored vectors. See ADR
//! 2026-05-14-voyage-4-migration-via-reembed-cli §Decision step 3.
//!
//! Two entry points:
//!
//! - [`check`] — read `schema_meta.embedding_model` and compare against the
//!   configured model. Called once at daemon startup from `server::run`.
//!   Returns [`CoreError::EmbeddingModelMismatch`] (`[6020]`) on conflict.
//!   On an empty key (bootstrap), writes the configured model and returns
//!   `Ok`. If the database already contains memories at bootstrap time
//!   (legacy upgrade path), prints a stderr warning advising `engram reembed`.
//! - [`record`] — overwrite `schema_meta.embedding_model` with the model
//!   that just re-embedded the database. Called by `reembed_handler` after
//!   a successful run so the next daemon start clears the mismatch.

use engram_storage::Database;

use crate::error::CoreError;

pub const EMBEDDING_MODEL_KEY: &str = "embedding_model";

pub fn check(database: &Database, configured_model: &str) -> Result<(), CoreError> {
    match super::read_meta(database, EMBEDDING_MODEL_KEY)? {
        Some(stored) if stored == configured_model => Ok(()),
        Some(stored) => Err(CoreError::EmbeddingModelMismatch {
            stored,
            configured: configured_model.to_string(),
        }),
        None => bootstrap(database, configured_model),
    }
}

pub fn record(database: &Database, model: &str) -> Result<(), CoreError> {
    super::write_meta(database, EMBEDDING_MODEL_KEY, model)
}

fn bootstrap(database: &Database, configured_model: &str) -> Result<(), CoreError> {
    if memories_count(database)? > 0 {
        eprintln!(
            "warning: schema_meta.embedding_model bootstrap on non-empty database — \
             if existing embeddings were computed with a model other than `{configured_model}`, \
             run `engram reembed` before further use"
        );
    }
    super::write_meta(database, EMBEDDING_MODEL_KEY, configured_model)
}

fn memories_count(database: &Database) -> Result<i64, CoreError> {
    database
        .connection()
        .query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))
        .map_err(|error| CoreError::MigrationFailed(format!("memories count: {error}")))
}
