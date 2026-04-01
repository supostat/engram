//! SQLite-backed storage layer with FTS5 full-text search.

pub mod consolidation;
pub mod database;
pub mod error;
pub mod fts;
pub mod memory;
pub mod q_table_store;
pub mod schema;

pub use database::Database;
pub use error::StorageError;
pub use fts::FtsResult;
pub use memory::Memory;
