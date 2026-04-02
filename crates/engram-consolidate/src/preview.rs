use std::collections::HashSet;

use engram_storage::{Database, StorageError};

use crate::error::ConsolidateError;

const FTS_DUPLICATE_LIMIT: usize = 5;

#[derive(Debug, Clone)]
pub struct DuplicateGroup {
    pub primary_id: String,
    pub duplicate_ids: Vec<String>,
    pub similarity: f32,
}

#[derive(Debug, Clone)]
pub struct PreviewResult {
    pub duplicates: Vec<DuplicateGroup>,
    pub stale: Vec<String>,
    pub garbage: Vec<String>,
}

pub fn preview(
    database: &Database,
    stale_days: u32,
    min_score: f64,
) -> Result<PreviewResult, ConsolidateError> {
    let duplicates = find_duplicates(database)?;
    let stale = find_stale(database, stale_days, min_score)?;
    let garbage = find_garbage(database)?;
    Ok(PreviewResult {
        duplicates,
        stale,
        garbage,
    })
}

fn find_duplicates(database: &Database) -> Result<Vec<DuplicateGroup>, ConsolidateError> {
    let memory_ids = list_memory_ids(database)?;
    let mut already_grouped: HashSet<String> = HashSet::new();
    let mut groups: Vec<DuplicateGroup> = Vec::new();

    for memory_id in &memory_ids {
        if already_grouped.contains(memory_id) {
            continue;
        }
        if let Some(group) = try_build_group(database, memory_id, &mut already_grouped)? {
            groups.push(group);
        }
    }
    Ok(groups)
}

fn try_build_group(
    database: &Database,
    memory_id: &str,
    already_grouped: &mut HashSet<String>,
) -> Result<Option<DuplicateGroup>, ConsolidateError> {
    let matches = find_fts_duplicates_for(database, memory_id)?;
    if matches.is_empty() {
        return Ok(None);
    }
    let duplicate_ids: Vec<String> = matches
        .iter()
        .filter(|(id, _)| !already_grouped.contains(id))
        .map(|(id, _)| id.clone())
        .collect();
    if duplicate_ids.is_empty() {
        return Ok(None);
    }
    let average_rank = compute_average_rank(&matches);
    already_grouped.insert(memory_id.to_string());
    for id in &duplicate_ids {
        already_grouped.insert(id.clone());
    }
    Ok(Some(DuplicateGroup {
        primary_id: memory_id.to_string(),
        duplicate_ids,
        similarity: average_rank,
    }))
}

fn list_memory_ids(database: &Database) -> Result<Vec<String>, ConsolidateError> {
    let mut statement = database
        .connection()
        .prepare("SELECT id FROM memories ORDER BY created_at")
        .map_err(StorageError::from)?;
    let rows = statement
        .query_map([], |row| row.get::<_, String>(0))
        .map_err(StorageError::from)?;
    let mut ids = Vec::new();
    for row in rows {
        ids.push(row.map_err(StorageError::from)?);
    }
    Ok(ids)
}

fn find_fts_duplicates_for(
    database: &Database,
    memory_id: &str,
) -> Result<Vec<(String, f64)>, ConsolidateError> {
    let memory = database.get_memory(memory_id)?;
    let fts_query = sanitize_fts_query(&memory.context);
    if fts_query.is_empty() {
        return Ok(Vec::new());
    }
    let fts_results = match database.search_fts(&fts_query, FTS_DUPLICATE_LIMIT) {
        Ok(results) => results,
        Err(_) => return Ok(Vec::new()),
    };
    let duplicates: Vec<(String, f64)> = fts_results
        .into_iter()
        .filter(|fts_result| fts_result.memory.id != memory_id)
        .map(|fts_result| (fts_result.memory.id, fts_result.rank))
        .collect();
    Ok(duplicates)
}

fn sanitize_fts_query(text: &str) -> String {
    text.chars()
        .filter(|character| character.is_alphanumeric() || character.is_whitespace())
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<&str>>()
        .join(" ")
}

fn compute_average_rank(matches: &[(String, f64)]) -> f32 {
    if matches.is_empty() {
        return 0.0;
    }
    let sum: f64 = matches.iter().map(|(_, rank)| rank.abs()).sum();
    (sum / matches.len() as f64) as f32
}

fn find_stale(
    database: &Database,
    stale_days: u32,
    min_score: f64,
) -> Result<Vec<String>, ConsolidateError> {
    let stale_ids = query_stale_memory_ids(database, stale_days, min_score)?;
    Ok(stale_ids)
}

fn query_stale_memory_ids(
    database: &Database,
    stale_days: u32,
    min_score: f64,
) -> Result<Vec<String>, ConsolidateError> {
    let mut statement = database
        .connection()
        .prepare(
            "SELECT id FROM memories
             WHERE score < ?1
               AND used_count = 0
               AND julianday('now') - julianday(created_at) > ?2",
        )
        .map_err(StorageError::from)?;
    let rows = statement
        .query_map(rusqlite::params![min_score, stale_days], |row| {
            row.get::<_, String>(0)
        })
        .map_err(StorageError::from)?;
    let mut ids = Vec::new();
    for row in rows {
        ids.push(row.map_err(StorageError::from)?);
    }
    Ok(ids)
}

fn find_garbage(database: &Database) -> Result<Vec<String>, ConsolidateError> {
    let orphan_ids = query_orphan_memory_ids(database)?;
    Ok(orphan_ids)
}

fn query_orphan_memory_ids(database: &Database) -> Result<Vec<String>, ConsolidateError> {
    let mut statement = database
        .connection()
        .prepare(
            "SELECT m.id FROM memories m
             WHERE m.parent_id IS NOT NULL
               AND NOT EXISTS (SELECT 1 FROM memories p WHERE p.id = m.parent_id)",
        )
        .map_err(StorageError::from)?;
    let rows = statement
        .query_map([], |row| row.get::<_, String>(0))
        .map_err(StorageError::from)?;
    let mut ids = Vec::new();
    for row in rows {
        ids.push(row.map_err(StorageError::from)?);
    }
    Ok(ids)
}
