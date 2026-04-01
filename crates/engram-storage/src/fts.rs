use rusqlite::params;

use crate::database::Database;
use crate::error::StorageError;
use crate::memory::{self, Memory};

#[derive(Debug, Clone)]
pub struct FtsResult {
    pub memory: Memory,
    pub rank: f64,
}

impl Database {
    pub fn search_fts(&self, query: &str, limit: usize) -> Result<Vec<FtsResult>, StorageError> {
        let mut statement = self.connection().prepare(
            "SELECT m.*, rank
             FROM memories m
             JOIN memories_fts ON memories_fts.rowid = m.rowid
             WHERE memories_fts MATCH ?1
             ORDER BY rank
             LIMIT ?2",
        )?;
        let rows = statement.query_map(params![query, limit as i64], |row| {
            let mem = memory::row_to_memory(row)?;
            let rank: f64 = row.get("rank")?;
            Ok(FtsResult { memory: mem, rank })
        })?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }
}
