//! Migrate `memories.tags` from legacy formats (CSV, naked token, non-canonical JSON)
//! to canonical JSON-array per ADR 2026-05-01.

use rusqlite::params;

use engram_storage::Database;

use crate::error::CoreError;

pub const TAGS_FORMAT_KEY: &str = "tags_format";
pub const TAGS_FORMAT_TARGET_VALUE: &str = "json_array_v1";

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct TagsFormatV1Stats {
    pub scanned: usize,
    pub already_json: usize,
    pub rewritten_csv: usize,
    pub rewritten_naked: usize,
    pub dry_run: bool,
}

pub fn run(
    database: &Database,
    dry_run: bool,
    strict: bool,
) -> Result<Option<TagsFormatV1Stats>, CoreError> {
    if super::read_meta(database, TAGS_FORMAT_KEY)?.as_deref() == Some(TAGS_FORMAT_TARGET_VALUE) {
        return Ok(None);
    }
    audit_comma_in_tags(database, strict)?;
    let mut stats = TagsFormatV1Stats {
        dry_run,
        ..Default::default()
    };
    let rows = collect_rows_with_tags(database)?;
    stats.scanned = rows.len();
    let updates: Vec<(String, String)> = rows
        .into_iter()
        .filter_map(|(id, raw)| classify_and_rewrite(&raw, &mut stats).map(|new| (id, new)))
        .collect();
    if dry_run {
        log_dry_run_diff(&updates);
    } else {
        apply_updates(database, &updates)?;
        super::write_meta(database, TAGS_FORMAT_KEY, TAGS_FORMAT_TARGET_VALUE)?;
    }
    log_summary(&stats);
    Ok(Some(stats))
}

fn collect_rows_with_tags(database: &Database) -> Result<Vec<(String, String)>, CoreError> {
    let mut statement = database
        .connection()
        .prepare("SELECT id, tags FROM memories WHERE tags IS NOT NULL AND tags <> ''")
        .map_err(|error| CoreError::MigrationFailed(format!("prepare scan: {error}")))?;
    let rows = statement
        .query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })
        .map_err(|error| CoreError::MigrationFailed(format!("query scan: {error}")))?;
    let mut collected = Vec::new();
    for row in rows {
        collected
            .push(row.map_err(|error| CoreError::MigrationFailed(format!("row scan: {error}")))?);
    }
    Ok(collected)
}

fn classify_and_rewrite(raw: &str, stats: &mut TagsFormatV1Stats) -> Option<String> {
    if let Ok(parsed) = serde_json::from_str::<Vec<String>>(raw) {
        let canonical = serde_json::to_string(&parsed).expect("Vec<String> serializes to JSON");
        stats.already_json += 1;
        return if canonical == raw {
            None
        } else {
            Some(canonical)
        };
    }
    if raw.contains(',') {
        let parts: Vec<String> = raw
            .split(',')
            .map(|tag| tag.trim().to_string())
            .filter(|tag| !tag.is_empty())
            .collect();
        stats.rewritten_csv += 1;
        return Some(serde_json::to_string(&parts).expect("Vec<String> serializes to JSON"));
    }
    let single = vec![raw.trim().to_string()];
    stats.rewritten_naked += 1;
    Some(serde_json::to_string(&single).expect("Vec<String> serializes to JSON"))
}

fn audit_comma_in_tags(database: &Database, strict: bool) -> Result<(), CoreError> {
    let suspects = collect_comma_suspects(database)?;
    if suspects.is_empty() {
        return Ok(());
    }
    if strict {
        return Err(CoreError::MigrationFailed(format!(
            "tag-format audit: {} row(s) with comma in non-JSON tags (strict mode); first id={}",
            suspects.len(),
            suspects[0].0
        )));
    }
    for (id, raw) in &suspects {
        eprintln!(
            "tag-format migration: comma may be content in id={id} tags={raw} \
             (will split on ','); set ENGRAM_TAGS_MIGRATION_STRICT=1 to block"
        );
    }
    Ok(())
}

fn collect_comma_suspects(database: &Database) -> Result<Vec<(String, String)>, CoreError> {
    let mut statement = database
        .connection()
        .prepare(
            "SELECT id, tags FROM memories \
             WHERE tags IS NOT NULL AND tags NOT LIKE '[%' AND tags LIKE '%,%'",
        )
        .map_err(|error| CoreError::MigrationFailed(format!("prepare audit: {error}")))?;
    let rows = statement
        .query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })
        .map_err(|error| CoreError::MigrationFailed(format!("query audit: {error}")))?;
    let mut collected = Vec::new();
    for row in rows {
        collected
            .push(row.map_err(|error| CoreError::MigrationFailed(format!("row audit: {error}")))?);
    }
    Ok(collected)
}

fn apply_updates(database: &Database, updates: &[(String, String)]) -> Result<(), CoreError> {
    let connection = database.connection();
    let transaction = connection
        .unchecked_transaction()
        .map_err(|error| CoreError::MigrationFailed(format!("tx begin: {error}")))?;
    for (id, new_tags) in updates {
        transaction
            .execute(
                "UPDATE memories SET tags = ?1 WHERE id = ?2",
                params![new_tags, id],
            )
            .map_err(|error| CoreError::MigrationFailed(format!("update {id}: {error}")))?;
    }
    transaction
        .commit()
        .map_err(|error| CoreError::MigrationFailed(format!("tx commit: {error}")))?;
    Ok(())
}

fn log_dry_run_diff(updates: &[(String, String)]) {
    for (id, new_tags) in updates {
        eprintln!("tag-format migration: would rewrite id={id} -> {new_tags}");
    }
}

fn log_summary(stats: &TagsFormatV1Stats) {
    let suffix = if stats.dry_run { " [dry-run]" } else { "" };
    eprintln!(
        "tag-format migration: {} scanned, {} json (noop or re-emit), \
         {} csv-rewritten, {} naked-rewritten{suffix}",
        stats.scanned, stats.already_json, stats.rewritten_csv, stats.rewritten_naked
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_canonical_json_is_noop() {
        let mut stats = TagsFormatV1Stats::default();
        let result = classify_and_rewrite(r#"["rust","bugfix"]"#, &mut stats);
        assert!(result.is_none(), "canonical json must not rewrite");
        assert_eq!(stats.already_json, 1);
        assert_eq!(stats.rewritten_csv, 0);
        assert_eq!(stats.rewritten_naked, 0);
    }

    #[test]
    fn classify_non_canonical_json_re_emits() {
        let mut stats = TagsFormatV1Stats::default();
        let result = classify_and_rewrite(r#"[ "rust" , "bugfix" ]"#, &mut stats);
        assert_eq!(result.as_deref(), Some(r#"["rust","bugfix"]"#));
        assert_eq!(stats.already_json, 1);
    }

    #[test]
    fn classify_csv_splits() {
        let mut stats = TagsFormatV1Stats::default();
        let result = classify_and_rewrite("rust,bugfix", &mut stats);
        assert_eq!(result.as_deref(), Some(r#"["rust","bugfix"]"#));
        assert_eq!(stats.rewritten_csv, 1);
    }

    #[test]
    fn classify_naked_wraps() {
        let mut stats = TagsFormatV1Stats::default();
        let result = classify_and_rewrite("rust", &mut stats);
        assert_eq!(result.as_deref(), Some(r#"["rust"]"#));
        assert_eq!(stats.rewritten_naked, 1);
    }

    #[test]
    fn classify_csv_with_empty_segments_filters() {
        let mut stats = TagsFormatV1Stats::default();
        let result = classify_and_rewrite(",a,,b,", &mut stats);
        assert_eq!(result.as_deref(), Some(r#"["a","b"]"#));
        assert_eq!(stats.rewritten_csv, 1);
    }
}
