use std::fmt;

#[derive(Debug)]
pub enum StorageError {
    /// 1001: Database connection failed or unavailable
    DatabaseUnavailable(String),
    /// 1002: Record not found
    NotFound(String),
    /// 1003: Duplicate key
    DuplicateKey(String),
    /// 1004: Migration required
    MigrationRequired(String),
    /// Wrapper for rusqlite errors
    Sqlite(rusqlite::Error),
}

impl fmt::Display for StorageError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DatabaseUnavailable(message) => {
                write!(formatter, "[1001] database unavailable: {message}")
            }
            Self::NotFound(message) => {
                write!(formatter, "[1002] not found: {message}")
            }
            Self::DuplicateKey(message) => {
                write!(formatter, "[1003] duplicate key: {message}")
            }
            Self::MigrationRequired(message) => {
                write!(formatter, "[1004] migration required: {message}")
            }
            Self::Sqlite(error) => {
                write!(formatter, "sqlite error: {error}")
            }
        }
    }
}

impl std::error::Error for StorageError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Sqlite(error) => Some(error),
            _ => None,
        }
    }
}

impl From<rusqlite::Error> for StorageError {
    fn from(error: rusqlite::Error) -> Self {
        Self::Sqlite(error)
    }
}
