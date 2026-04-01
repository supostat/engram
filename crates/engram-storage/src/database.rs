use rusqlite::Connection;

use crate::error::StorageError;
use crate::schema;

pub struct Database {
    connection: Connection,
}

impl Database {
    /// Open or create database at the given path.
    pub fn open(path: &str) -> Result<Self, StorageError> {
        let connection = Connection::open(path)?;
        schema::apply_schema(&connection)?;
        Ok(Self { connection })
    }

    /// Create an in-memory database (for testing).
    pub fn in_memory() -> Result<Self, StorageError> {
        Self::open(":memory:")
    }

    pub fn connection(&self) -> &Connection {
        &self.connection
    }
}
