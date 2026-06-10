use rusqlite::params;

use crate::database::Database;
use crate::error::StorageError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConsolidationLogEntry {
    pub id: String,
    pub action: String,
    pub memory_ids_json: String,
    pub reason: Option<String>,
    pub performed_at: String,
    pub performed_by: String,
}

impl Database {
    pub fn list_consolidation_log(
        &self,
        limit: usize,
    ) -> Result<Vec<ConsolidationLogEntry>, StorageError> {
        let mut statement = self.connection().prepare(
            "SELECT id, action, memory_ids, reason, performed_at, performed_by
             FROM consolidation_log
             ORDER BY performed_at DESC, id DESC
             LIMIT ?1",
        )?;
        let rows = statement.query_map(params![limit as i64], |row| {
            Ok(ConsolidationLogEntry {
                id: row.get(0)?,
                action: row.get(1)?,
                memory_ids_json: row.get(2)?,
                reason: row.get(3)?,
                performed_at: row.get(4)?,
                performed_by: row.get(5)?,
            })
        })?;
        let mut entries = Vec::new();
        for row in rows {
            entries.push(row?);
        }
        Ok(entries)
    }
}
