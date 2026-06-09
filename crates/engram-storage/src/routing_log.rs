use rusqlite::params;

use crate::database::Database;
use crate::error::StorageError;

pub struct RoutingLogEntry<'a> {
    pub query_id: &'a str,
    pub mode: &'a str,
    pub search_strategy: &'a str,
    pub llm_selection: &'a str,
    pub contextualization: &'a str,
    pub proactivity: &'a str,
    pub top_k: usize,
    pub shadow_rewards_json: Option<&'a str>,
    pub created_at: &'a str,
}

impl Database {
    pub fn log_routing_decision(&self, entry: &RoutingLogEntry<'_>) -> Result<(), StorageError> {
        self.connection().execute(
            "INSERT INTO routing_log \
             (query_id, mode, search_strategy, llm_selection, contextualization, proactivity, top_k, shadow_rewards, created_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                entry.query_id,
                entry.mode,
                entry.search_strategy,
                entry.llm_selection,
                entry.contextualization,
                entry.proactivity,
                entry.top_k as i64,
                entry.shadow_rewards_json,
                entry.created_at,
            ],
        )?;
        Ok(())
    }
}
