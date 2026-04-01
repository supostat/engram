use rusqlite::params;

use crate::database::Database;
use crate::error::StorageError;

impl Database {
    pub fn log_consolidation(
        &self,
        id: &str,
        action: &str,
        memory_ids: &[String],
        reason: Option<&str>,
        performed_by: &str,
        timestamp: &str,
    ) -> Result<(), StorageError> {
        let memory_ids_json = serde_json::to_string(memory_ids)
            .map_err(|err| StorageError::DatabaseUnavailable(err.to_string()))?;
        self.connection().execute(
            "INSERT INTO consolidation_log (id, action, memory_ids, reason, performed_at, performed_by)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![id, action, memory_ids_json, reason, timestamp, performed_by],
        )?;
        Ok(())
    }

    pub fn track_search(&self, memory_id: &str, timestamp: &str) -> Result<(), StorageError> {
        self.connection().execute(
            "INSERT INTO feedback_tracking (memory_id, searched_at) VALUES (?1, ?2)",
            params![memory_id, timestamp],
        )?;
        Ok(())
    }

    pub fn mark_judged(&self, memory_id: &str, timestamp: &str) -> Result<(), StorageError> {
        self.connection().execute(
            "UPDATE feedback_tracking SET judged = TRUE, judged_at = ?1
             WHERE memory_id = ?2 AND judged = FALSE",
            params![timestamp, memory_id],
        )?;
        Ok(())
    }

    pub fn get_pending_judgments(&self, limit: usize) -> Result<Vec<String>, StorageError> {
        let mut statement = self.connection().prepare(
            "SELECT DISTINCT memory_id FROM feedback_tracking WHERE judged = FALSE LIMIT ?1",
        )?;
        let rows = statement.query_map(params![limit as i64], |row| row.get(0))?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }
}
