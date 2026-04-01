use rusqlite::params;
use serde::{Deserialize, Serialize};

use crate::database::Database;
use crate::error::StorageError;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub id: String,
    pub memory_type: String,
    pub context: String,
    pub action: String,
    pub result: String,
    pub score: f32,
    pub embedding_context: Option<Vec<u8>>,
    pub embedding_action: Option<Vec<u8>>,
    pub embedding_result: Option<Vec<u8>>,
    pub indexed: bool,
    pub tags: Option<String>,
    pub project: Option<String>,
    pub parent_id: Option<String>,
    pub source_ids: Option<String>,
    pub insight_type: Option<String>,
    pub created_at: String,
    pub updated_at: String,
    pub used_count: i64,
    pub last_used_at: Option<String>,
    pub superseded_by: Option<String>,
}

pub(crate) fn row_to_memory(row: &rusqlite::Row) -> rusqlite::Result<Memory> {
    Ok(Memory {
        id: row.get("id")?,
        memory_type: row.get("memory_type")?,
        context: row.get("context")?,
        action: row.get("action")?,
        result: row.get("result")?,
        score: row.get("score")?,
        embedding_context: row.get("embedding_context")?,
        embedding_action: row.get("embedding_action")?,
        embedding_result: row.get("embedding_result")?,
        indexed: row.get("indexed")?,
        tags: row.get("tags")?,
        project: row.get("project")?,
        parent_id: row.get("parent_id")?,
        source_ids: row.get("source_ids")?,
        insight_type: row.get("insight_type")?,
        created_at: row.get("created_at")?,
        updated_at: row.get("updated_at")?,
        used_count: row.get("used_count")?,
        last_used_at: row.get("last_used_at")?,
        superseded_by: row.get("superseded_by")?,
    })
}

const INSERT_SQL: &str = r#"
    INSERT INTO memories (
        id, memory_type, context, action, result, score,
        embedding_context, embedding_action, embedding_result,
        indexed, tags, project, parent_id, source_ids, insight_type,
        created_at, updated_at, used_count, last_used_at, superseded_by
    ) VALUES (
        ?1, ?2, ?3, ?4, ?5, ?6,
        ?7, ?8, ?9,
        ?10, ?11, ?12, ?13, ?14, ?15,
        ?16, ?17, ?18, ?19, ?20
    )
"#;

impl Database {
    pub fn insert_memory(&self, memory: &Memory) -> Result<(), StorageError> {
        self.connection()
            .execute(
                INSERT_SQL,
                params![
                    memory.id,
                    memory.memory_type,
                    memory.context,
                    memory.action,
                    memory.result,
                    memory.score,
                    memory.embedding_context,
                    memory.embedding_action,
                    memory.embedding_result,
                    memory.indexed,
                    memory.tags,
                    memory.project,
                    memory.parent_id,
                    memory.source_ids,
                    memory.insight_type,
                    memory.created_at,
                    memory.updated_at,
                    memory.used_count,
                    memory.last_used_at,
                    memory.superseded_by,
                ],
            )
            .map_err(|error| match error {
                rusqlite::Error::SqliteFailure(sql_error, _)
                    if sql_error.extended_code == rusqlite::ffi::SQLITE_CONSTRAINT_PRIMARYKEY
                        || sql_error.extended_code == rusqlite::ffi::SQLITE_CONSTRAINT_UNIQUE =>
                {
                    StorageError::DuplicateKey(format!("memory id={}", memory.id))
                }
                other => StorageError::Sqlite(other),
            })?;
        Ok(())
    }

    pub fn get_memory(&self, id: &str) -> Result<Memory, StorageError> {
        self.connection()
            .query_row(
                "SELECT * FROM memories WHERE id = ?1",
                params![id],
                row_to_memory,
            )
            .map_err(|error| match error {
                rusqlite::Error::QueryReturnedNoRows => {
                    StorageError::NotFound(format!("memory id={id}"))
                }
                other => StorageError::Sqlite(other),
            })
    }

    pub fn set_memory_indexed(&self, id: &str, indexed: bool) -> Result<(), StorageError> {
        let affected = self.connection().execute(
            "UPDATE memories SET indexed = ?1 WHERE id = ?2",
            params![indexed, id],
        )?;
        if affected == 0 {
            return Err(StorageError::NotFound(format!("memory id={id}")));
        }
        Ok(())
    }

    pub fn set_memory_score(&self, id: &str, score: f32) -> Result<(), StorageError> {
        let affected = self.connection().execute(
            "UPDATE memories SET score = ?1 WHERE id = ?2",
            params![score, id],
        )?;
        if affected == 0 {
            return Err(StorageError::NotFound(format!("memory id={id}")));
        }
        Ok(())
    }

    pub fn touch_memory(&self, id: &str, timestamp: &str) -> Result<(), StorageError> {
        let affected = self.connection().execute(
            "UPDATE memories SET used_count = used_count + 1, last_used_at = ?1 WHERE id = ?2",
            params![timestamp, id],
        )?;
        if affected == 0 {
            return Err(StorageError::NotFound(format!("memory id={id}")));
        }
        Ok(())
    }

    pub fn delete_memory(&self, id: &str) -> Result<(), StorageError> {
        let affected = self
            .connection()
            .execute("DELETE FROM memories WHERE id = ?1", params![id])?;
        if affected == 0 {
            return Err(StorageError::NotFound(format!("memory id={id}")));
        }
        Ok(())
    }

    pub fn bulk_insert_memories(&self, memories: &[Memory]) -> Result<usize, StorageError> {
        let transaction = self.connection().unchecked_transaction()?;
        let mut statement = transaction.prepare(INSERT_SQL)?;
        let mut count = 0;
        for memory in memories {
            statement.execute(params![
                memory.id,
                memory.memory_type,
                memory.context,
                memory.action,
                memory.result,
                memory.score,
                memory.embedding_context,
                memory.embedding_action,
                memory.embedding_result,
                memory.indexed,
                memory.tags,
                memory.project,
                memory.parent_id,
                memory.source_ids,
                memory.insight_type,
                memory.created_at,
                memory.updated_at,
                memory.used_count,
                memory.last_used_at,
                memory.superseded_by,
            ])?;
            count += 1;
        }
        drop(statement);
        transaction.commit()?;
        Ok(count)
    }

    pub fn get_unindexed_memories(&self, limit: usize) -> Result<Vec<Memory>, StorageError> {
        let mut statement = self
            .connection()
            .prepare("SELECT * FROM memories WHERE indexed = FALSE LIMIT ?1")?;
        let rows = statement.query_map(params![limit as i64], row_to_memory)?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }
}
