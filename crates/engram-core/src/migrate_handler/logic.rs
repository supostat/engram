//! Pure migration logic: filtering, duplicate detection, bulk insert.
//!
//! No filesystem or environment access — all inputs are passed explicitly.

use serde_json::{Value, json};

use engram_storage::{Database, Memory, StorageError};

use crate::error::CoreError;

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct MigrationStats {
    pub read: usize,
    pub matched: usize,
    pub migrated: usize,
    pub skipped_duplicate: usize,
    pub skipped_null_project: usize,
    pub skipped_other_project: usize,
}

impl MigrationStats {
    pub fn to_json(&self, dry_run: bool, all: bool, project_hint: Option<&str>) -> Value {
        json!({
            "read": self.read,
            "matched": self.matched,
            "migrated": self.migrated,
            "skipped_duplicate": self.skipped_duplicate,
            "skipped_null_project": self.skipped_null_project,
            "skipped_other_project": self.skipped_other_project,
            "dry_run": dry_run,
            "all": all,
            "project_hint": project_hint,
        })
    }
}

/// Copies memories from `source` into `dest`.
///
/// Transaction semantics: `dest.bulk_insert_memories` wraps the inserts in a single
/// SQLite transaction — either all queued rows commit or none do (all-or-nothing).
/// Duplicates MUST therefore be filtered out before the bulk call; otherwise a single
/// duplicate ID rolls back the entire batch.
pub fn perform_migration_impl(
    source: &Database,
    dest: &Database,
    project_hint: Option<&str>,
    all: bool,
    dry_run: bool,
) -> Result<MigrationStats, CoreError> {
    let memories = source.list_all_memories()?;
    let mut stats = MigrationStats {
        read: memories.len(),
        ..MigrationStats::default()
    };

    let mut to_insert: Vec<Memory> = Vec::new();
    for memory in memories {
        if !memory_matches_filter(&memory, project_hint, all, &mut stats) {
            continue;
        }
        stats.matched += 1;
        if dest_contains_memory(dest, &memory.id)? {
            stats.skipped_duplicate += 1;
            continue;
        }
        let mut copy = memory;
        // Force HNSW rebuild from embeddings on server startup; skip if already unindexed.
        copy.indexed = false;
        to_insert.push(copy);
    }

    if !dry_run && !to_insert.is_empty() {
        dest.bulk_insert_memories(&to_insert)
            .map_err(|error| CoreError::MigrationFailed(error.to_string()))?;
    }
    stats.migrated = if dry_run { 0 } else { to_insert.len() };
    Ok(stats)
}

fn memory_matches_filter(
    memory: &Memory,
    project_hint: Option<&str>,
    all: bool,
    stats: &mut MigrationStats,
) -> bool {
    if all {
        return true;
    }
    match (project_hint, memory.project.as_deref()) {
        (None, _) => true,
        (Some(_), None) => {
            stats.skipped_null_project += 1;
            false
        }
        (Some(hint), Some(project)) if project == hint => true,
        (Some(_), Some(_)) => {
            stats.skipped_other_project += 1;
            false
        }
    }
}

fn dest_contains_memory(dest: &Database, id: &str) -> Result<bool, CoreError> {
    match dest.get_memory(id) {
        Ok(_) => Ok(true),
        Err(StorageError::NotFound(_)) => Ok(false),
        Err(error) => Err(CoreError::MigrationFailed(error.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    fn sample_memory(id: &str, project: Option<&str>) -> Memory {
        Memory {
            id: id.to_string(),
            memory_type: "decision".into(),
            context: "context".into(),
            action: "action".into(),
            result: "result".into(),
            score: 0.5,
            embedding_context: None,
            embedding_action: None,
            embedding_result: None,
            indexed: true,
            tags: None,
            project: project.map(|value| value.to_string()),
            parent_id: None,
            source_ids: None,
            insight_type: None,
            created_at: "2025-01-01T00:00:00Z".into(),
            updated_at: "2025-01-01T00:00:00Z".into(),
            used_count: 0,
            last_used_at: None,
            superseded_by: None,
        }
    }

    fn insert_sample(database: &Database, memory: &Memory) {
        database.insert_memory(memory).expect("seed insert");
    }

    #[test]
    fn empty_source_returns_zero_stats() {
        let source = Database::in_memory().expect("source");
        let dest = Database::in_memory().expect("dest");
        let stats = perform_migration_impl(&source, &dest, Some("engram"), false, false)
            .expect("migration");
        assert_eq!(stats, MigrationStats::default());
    }

    #[test]
    fn all_flag_copies_all_rows_including_null_project() {
        let source = Database::in_memory().expect("source");
        let dest = Database::in_memory().expect("dest");
        insert_sample(&source, &sample_memory("a", Some("engram")));
        insert_sample(&source, &sample_memory("b", None));
        insert_sample(&source, &sample_memory("c", Some("other")));

        let stats = perform_migration_impl(&source, &dest, None, true, false).expect("migration");

        assert_eq!(stats.read, 3);
        assert_eq!(stats.matched, 3);
        assert_eq!(stats.migrated, 3);
        assert_eq!(stats.skipped_null_project, 0);
        assert_eq!(stats.skipped_other_project, 0);
        for id in ["a", "b", "c"] {
            dest.get_memory(id).expect("row migrated");
        }
    }

    #[test]
    fn default_filter_matches_project_hint_exactly() {
        let source = Database::in_memory().expect("source");
        let dest = Database::in_memory().expect("dest");
        insert_sample(&source, &sample_memory("match", Some("engram")));
        insert_sample(&source, &sample_memory("null", None));
        insert_sample(&source, &sample_memory("other", Some("different")));

        let stats = perform_migration_impl(&source, &dest, Some("engram"), false, false)
            .expect("migration");

        assert_eq!(stats.read, 3);
        assert_eq!(stats.matched, 1);
        assert_eq!(stats.migrated, 1);
        assert_eq!(stats.skipped_null_project, 1);
        assert_eq!(stats.skipped_other_project, 1);
        dest.get_memory("match").expect("match migrated");
        assert!(matches!(
            dest.get_memory("null"),
            Err(StorageError::NotFound(_))
        ));
        assert!(matches!(
            dest.get_memory("other"),
            Err(StorageError::NotFound(_))
        ));
    }

    #[test]
    fn skips_duplicate_ids_without_rolling_back_batch() {
        let source = Database::in_memory().expect("source");
        let dest = Database::in_memory().expect("dest");
        insert_sample(&source, &sample_memory("dup", Some("engram")));
        insert_sample(&source, &sample_memory("new", Some("engram")));
        insert_sample(&dest, &sample_memory("dup", Some("engram")));

        let stats = perform_migration_impl(&source, &dest, Some("engram"), false, false)
            .expect("migration");

        assert_eq!(stats.read, 2);
        assert_eq!(stats.matched, 2);
        assert_eq!(stats.migrated, 1);
        assert_eq!(stats.skipped_duplicate, 1);
        dest.get_memory("new").expect("new migrated");
    }

    #[test]
    fn dry_run_counts_without_writing() {
        let source = Database::in_memory().expect("source");
        let dest = Database::in_memory().expect("dest");
        insert_sample(&source, &sample_memory("a", Some("engram")));
        insert_sample(&source, &sample_memory("b", Some("engram")));

        let stats =
            perform_migration_impl(&source, &dest, Some("engram"), false, true).expect("migration");

        assert_eq!(stats.read, 2);
        assert_eq!(stats.matched, 2);
        assert_eq!(stats.migrated, 0);
        assert_eq!(stats.skipped_duplicate, 0);
        assert!(matches!(
            dest.get_memory("a"),
            Err(StorageError::NotFound(_))
        ));
        assert!(matches!(
            dest.get_memory("b"),
            Err(StorageError::NotFound(_))
        ));
    }

    #[test]
    fn migrated_row_has_indexed_false_for_hnsw_rebuild() {
        let source = Database::in_memory().expect("source");
        let dest = Database::in_memory().expect("dest");
        let mut memory = sample_memory("x", Some("engram"));
        memory.indexed = true;
        insert_sample(&source, &memory);

        let stats = perform_migration_impl(&source, &dest, Some("engram"), false, false)
            .expect("migration");
        assert_eq!(stats.migrated, 1);

        let migrated = dest.get_memory("x").expect("migrated row");
        assert!(
            !migrated.indexed,
            "migrated rows must be unindexed so server rebuilds HNSW"
        );
    }

    #[test]
    fn migrate_reports_6019_when_dest_write_fails() {
        let source = Database::in_memory().expect("source");
        insert_sample(&source, &sample_memory("id-1", Some("proj")));

        let dest_file = NamedTempFile::new().expect("temp dest file");
        let dest_path = dest_file.path().to_str().expect("utf-8");
        // Initialize schema with a read-write open, then drop the handle before reopening read-only.
        {
            let _schema = Database::open(dest_path).expect("init schema");
        }
        let dest = Database::open_read_only(dest_path).expect("reopen dest read-only");

        let result = perform_migration_impl(&source, &dest, Some("proj"), false, false);

        let error = result.expect_err("write to read-only dest must fail");
        assert!(matches!(error, CoreError::MigrationFailed(_)));
        let message = error.to_string();
        assert!(
            message.contains("[6019]"),
            "expected 6019 migration failure, got: {message}"
        );
    }
}
