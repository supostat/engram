use std::collections::HashSet;

use engram_storage::{Database, StorageError};

use crate::error::ConsolidateError;

const FTS_DUPLICATE_LIMIT: usize = 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchType {
    Exact,
    Fts,
}

impl MatchType {
    pub fn as_str(&self) -> &'static str {
        match self {
            MatchType::Exact => "exact",
            MatchType::Fts => "fts",
        }
    }
}

/// A set of memories the preview pass considers duplicates of each other.
///
/// `primary_id` is the deterministic group representative — the
/// lexicographically smallest id for exact groups, the probe memory for FTS
/// groups. It is NOT necessarily the merge survivor; analyze picks the
/// survivor independently.
///
/// The `similarity` scale is discriminated by `match_type`: exact groups
/// carry the sentinel `1.0`, FTS groups carry the mean |bm25| rank of the
/// group members (non-negative, unbounded above, NOT cosine).
#[derive(Debug, Clone)]
pub struct DuplicateGroup {
    pub primary_id: String,
    pub duplicate_ids: Vec<String>,
    pub similarity: f32,
    pub match_type: MatchType,
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
    fts_similarity_floor: f32,
) -> Result<PreviewResult, ConsolidateError> {
    let duplicates = find_duplicates(database, fts_similarity_floor)?;
    let stale = find_stale(database, stale_days, min_score)?;
    let garbage = find_garbage(database)?;
    Ok(PreviewResult {
        duplicates,
        stale,
        garbage,
    })
}

fn find_duplicates(
    database: &Database,
    fts_similarity_floor: f32,
) -> Result<Vec<DuplicateGroup>, ConsolidateError> {
    let mut already_grouped: HashSet<String> = HashSet::new();
    let mut groups: Vec<DuplicateGroup> = Vec::new();

    for group in find_exact_duplicate_groups(database)? {
        already_grouped.insert(group.primary_id.clone());
        for id in &group.duplicate_ids {
            already_grouped.insert(id.clone());
        }
        groups.push(group);
    }

    let memory_ids = list_memory_ids(database)?;
    for memory_id in &memory_ids {
        if already_grouped.contains(memory_id) {
            continue;
        }
        if let Some(group) = try_build_group(
            database,
            memory_id,
            &mut already_grouped,
            fts_similarity_floor,
        )? {
            groups.push(group);
        }
    }
    Ok(groups)
}

// Groups memories whose (context, action, result) triplet is byte-identical — exact
// duplicates that FTS top-K ranking can miss once a group grows beyond the FTS limit.
// Ids are concatenated with a newline (char(10)); UUID memory ids never contain a
// newline, so splitting on it recovers the exact member set.
fn find_exact_duplicate_groups(
    database: &Database,
) -> Result<Vec<DuplicateGroup>, ConsolidateError> {
    let mut statement = database
        .connection()
        .prepare(
            "SELECT GROUP_CONCAT(id, char(10)) FROM memories
             WHERE superseded_by IS NULL AND memory_type != 'insight'
             GROUP BY context, action, result
             HAVING COUNT(*) > 1
             ORDER BY MIN(created_at)",
        )
        .map_err(StorageError::from)?;
    let rows = statement
        .query_map([], |row| row.get::<_, String>(0))
        .map_err(StorageError::from)?;
    let mut groups = Vec::new();
    for row in rows {
        let concatenated = row.map_err(StorageError::from)?;
        let mut ids: Vec<String> = concatenated.split('\n').map(str::to_string).collect();
        ids.sort();
        let mut members = ids.into_iter();
        let primary_id = match members.next() {
            Some(id) => id,
            None => continue,
        };
        groups.push(DuplicateGroup {
            primary_id,
            duplicate_ids: members.collect(),
            similarity: 1.0,
            match_type: MatchType::Exact,
        });
    }
    Ok(groups)
}

fn try_build_group(
    database: &Database,
    memory_id: &str,
    already_grouped: &mut HashSet<String>,
    fts_similarity_floor: f32,
) -> Result<Option<DuplicateGroup>, ConsolidateError> {
    let members: Vec<(String, f64)> = find_fts_duplicates_for(database, memory_id)?
        .into_iter()
        .filter(|(id, _)| !already_grouped.contains(id))
        .collect();
    if members.is_empty() {
        return Ok(None);
    }
    let average_rank = compute_average_rank(&members);
    if average_rank < fts_similarity_floor {
        return Ok(None);
    }
    already_grouped.insert(memory_id.to_string());
    let mut duplicate_ids = Vec::with_capacity(members.len());
    for (id, _) in members {
        already_grouped.insert(id.clone());
        duplicate_ids.push(id);
    }
    Ok(Some(DuplicateGroup {
        primary_id: memory_id.to_string(),
        duplicate_ids,
        similarity: average_rank,
        match_type: MatchType::Fts,
    }))
}

fn list_memory_ids(database: &Database) -> Result<Vec<String>, ConsolidateError> {
    let mut statement = database
        .connection()
        .prepare(
            "SELECT id FROM memories
             WHERE superseded_by IS NULL AND memory_type != 'insight'
             ORDER BY created_at",
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
        .filter(|fts_result| {
            fts_result.memory.id != memory_id
                && fts_result.memory.memory_type != "insight"
                && fts_result.memory.superseded_by.is_none()
        })
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
