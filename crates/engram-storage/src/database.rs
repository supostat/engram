use rusqlite::{Connection, OpenFlags};

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

    /// Open an existing database in read-only mode.
    ///
    /// Skips schema application: the caller must ensure the schema already exists
    /// (a read-only connection cannot run `CREATE TABLE` or set `PRAGMA journal_mode`).
    /// Intended for defense-in-depth scenarios where the source database must not
    /// be mutated (e.g. `engram migrate`).
    pub fn open_read_only(path: &str) -> Result<Self, StorageError> {
        let connection = Connection::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI,
        )?;
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
