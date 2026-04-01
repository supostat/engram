use rusqlite::params;

use crate::database::Database;
use crate::error::StorageError;

impl Database {
    pub fn upsert_q_value(
        &self,
        level: &str,
        state: &str,
        action: &str,
        value: f32,
        timestamp: &str,
    ) -> Result<(), StorageError> {
        self.connection().execute(
            "INSERT INTO q_table (router_level, state, action, value, update_count, updated_at)
             VALUES (?1, ?2, ?3, ?4, 1, ?5)
             ON CONFLICT(router_level, state, action)
             DO UPDATE SET value = ?4, update_count = update_count + 1, updated_at = ?5",
            params![level, state, action, value, timestamp],
        )?;
        Ok(())
    }

    pub fn get_q_value(&self, level: &str, state: &str, action: &str) -> Result<f32, StorageError> {
        let result = self.connection().query_row(
            "SELECT value FROM q_table WHERE router_level = ?1 AND state = ?2 AND action = ?3",
            params![level, state, action],
            |row| row.get(0),
        );
        match result {
            Ok(value) => Ok(value),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(0.0),
            Err(other) => Err(StorageError::Sqlite(other)),
        }
    }

    pub fn load_q_table(
        &self,
        level: &str,
    ) -> Result<Vec<(String, String, f32, u32)>, StorageError> {
        let mut statement = self.connection().prepare(
            "SELECT state, action, value, update_count
             FROM q_table
             WHERE router_level = ?1",
        )?;
        let rows = statement.query_map(params![level], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
        })?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }
}
