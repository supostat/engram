use engram_consolidate::analyze::{analyze, RecommendedAction};
use engram_consolidate::apply::apply;
use engram_consolidate::error::ConsolidateError;
use engram_consolidate::preview::{preview, DuplicateGroup, PreviewResult};
use engram_llm_client::{ApiError, TextGenerator};
use engram_storage::memory::Memory;
use engram_storage::Database;

struct MockTextGenerator {
    response: String,
}

impl MockTextGenerator {
    fn responding(response: &str) -> Self {
        Self {
            response: response.to_string(),
        }
    }
}

impl TextGenerator for MockTextGenerator {
    fn generate(&self, _prompt: &str) -> Result<String, ApiError> {
        Ok(self.response.clone())
    }

    fn model_name(&self) -> &str {
        "mock-model"
    }
}

struct FailingTextGenerator;

impl TextGenerator for FailingTextGenerator {
    fn generate(&self, _prompt: &str) -> Result<String, ApiError> {
        Err(ApiError::LlmApiUnavailable("service down".to_string()))
    }

    fn model_name(&self) -> &str {
        "failing-model"
    }
}

fn make_memory(id: &str, context: &str) -> Memory {
    Memory {
        id: id.to_string(),
        memory_type: "decision".to_string(),
        context: context.to_string(),
        action: format!("action for {id}"),
        result: format!("result for {id}"),
        score: 0.5,
        embedding_context: None,
        embedding_action: None,
        embedding_result: None,
        indexed: true,
        tags: None,
        project: None,
        parent_id: None,
        source_ids: None,
        insight_type: None,
        created_at: "2025-01-01T00:00:00Z".to_string(),
        updated_at: "2025-01-01T00:00:00Z".to_string(),
        used_count: 0,
        last_used_at: None,
        superseded_by: None,
    }
}

fn make_stale_memory(id: &str) -> Memory {
    let mut memory = make_memory(id, &format!("stale context {id}"));
    memory.score = 0.05;
    memory.used_count = 0;
    memory.created_at = "2020-01-01T00:00:00Z".to_string();
    memory.updated_at = "2020-01-01T00:00:00Z".to_string();
    memory
}

fn insert_orphan_memory(database: &Database, id: &str) {
    let parent = make_memory("temp-parent-for-orphan", "temporary parent");
    database.insert_memory(&parent).unwrap();
    let mut child = make_memory(id, &format!("orphan context {id}"));
    child.parent_id = Some("temp-parent-for-orphan".to_string());
    database.insert_memory(&child).unwrap();
    database
        .connection()
        .execute_batch("PRAGMA foreign_keys = OFF;")
        .unwrap();
    database
        .connection()
        .execute(
            "DELETE FROM memories WHERE id = 'temp-parent-for-orphan'",
            [],
        )
        .unwrap();
    database
        .connection()
        .execute_batch("PRAGMA foreign_keys = ON;")
        .unwrap();
}

#[test]
fn test_preview_finds_duplicates_via_fts() {
    let database = Database::in_memory().unwrap();
    database
        .insert_memory(&make_memory("m1", "rust memory management"))
        .unwrap();
    database
        .insert_memory(&make_memory("m2", "rust memory management"))
        .unwrap();

    let result = preview(&database, 365, 0.1).unwrap();
    assert!(
        !result.duplicates.is_empty(),
        "should find duplicate memories with identical context"
    );
}

#[test]
fn test_preview_finds_stale_memories() {
    let database = Database::in_memory().unwrap();
    database.insert_memory(&make_stale_memory("s1")).unwrap();
    database.insert_memory(&make_stale_memory("s2")).unwrap();

    let result = preview(&database, 30, 0.1).unwrap();
    assert_eq!(result.stale.len(), 2);
    assert!(result.stale.contains(&"s1".to_string()));
    assert!(result.stale.contains(&"s2".to_string()));
}

#[test]
fn test_preview_finds_garbage_broken_parent() {
    let database = Database::in_memory().unwrap();
    insert_orphan_memory(&database, "orphan1");

    let result = preview(&database, 365, 0.1).unwrap();
    assert_eq!(result.garbage.len(), 1);
    assert_eq!(result.garbage[0], "orphan1");
}

#[test]
fn test_preview_empty_when_no_candidates() {
    let database = Database::in_memory().unwrap();
    let mut healthy = make_memory("h1", "unique context alpha");
    healthy.score = 0.9;
    healthy.used_count = 5;
    database.insert_memory(&healthy).unwrap();

    let result = preview(&database, 365, 0.1).unwrap();
    assert!(result.duplicates.is_empty());
    assert!(result.stale.is_empty());
    assert!(result.garbage.is_empty());
}

#[test]
fn test_analyze_with_mock_llm_produces_merge() {
    let database = Database::in_memory().unwrap();
    database
        .insert_memory(&make_memory("m1", "rust ownership"))
        .unwrap();
    database
        .insert_memory(&make_memory("m2", "rust ownership"))
        .unwrap();

    let preview_result = PreviewResult {
        duplicates: vec![DuplicateGroup {
            primary_id: "m1".to_string(),
            duplicate_ids: vec!["m2".to_string()],
            similarity: 0.95,
        }],
        stale: Vec::new(),
        garbage: Vec::new(),
    };

    let generator = MockTextGenerator::responding("MERGE");
    let analysis =
        analyze(&database, &preview_result, Some(&generator)).unwrap();
    assert_eq!(analysis.analyzed_count, 2);
    assert!(!analysis.recommendations.is_empty());

    let first = &analysis.recommendations[0];
    assert!(
        matches!(&first.action, RecommendedAction::Merge { .. }),
        "LLM responding MERGE should produce Merge recommendation"
    );
}

#[test]
fn test_analyze_without_llm_uses_heuristic() {
    let database = Database::in_memory().unwrap();
    let mut high_score = make_memory("m1", "rust borrowing");
    high_score.score = 0.9;
    high_score.used_count = 10;
    database.insert_memory(&high_score).unwrap();

    let mut low_score = make_memory("m2", "rust borrowing");
    low_score.score = 0.3;
    low_score.used_count = 1;
    database.insert_memory(&low_score).unwrap();

    let preview_result = PreviewResult {
        duplicates: vec![DuplicateGroup {
            primary_id: "m1".to_string(),
            duplicate_ids: vec!["m2".to_string()],
            similarity: 0.9,
        }],
        stale: Vec::new(),
        garbage: Vec::new(),
    };

    let analysis = analyze(&database, &preview_result, None).unwrap();
    assert_eq!(analysis.analyzed_count, 2);

    let first = &analysis.recommendations[0];
    match &first.action {
        RecommendedAction::Merge {
            source_id,
            target_id,
        } => {
            assert_eq!(source_id, "m1", "higher score memory should be source");
            assert_eq!(target_id, "m2", "lower score memory should be target");
        }
        other => panic!("expected Merge, got {other:?}"),
    }
}

#[test]
fn test_apply_merge_sets_superseded_by() {
    let database = Database::in_memory().unwrap();
    database
        .insert_memory(&make_memory("m1", "context alpha"))
        .unwrap();
    database
        .insert_memory(&make_memory("m2", "context alpha"))
        .unwrap();

    let recommendations = vec![engram_consolidate::Recommendation {
        action: RecommendedAction::Merge {
            source_id: "m1".to_string(),
            target_id: "m2".to_string(),
        },
        confidence: 0.8,
        reasoning: "test merge".to_string(),
    }];

    let result = apply(&database, &recommendations, "test-agent").unwrap();
    assert_eq!(result.merged, 1);
    assert!(result.errors.is_empty());

    let merged_memory = database.get_memory("m2").unwrap();
    assert_eq!(merged_memory.superseded_by.as_deref(), Some("m1"));
}

#[test]
fn test_apply_delete_removes_memory() {
    let database = Database::in_memory().unwrap();
    insert_orphan_memory(&database, "orphan1");

    let recommendations = vec![engram_consolidate::Recommendation {
        action: RecommendedAction::Delete {
            memory_id: "orphan1".to_string(),
        },
        confidence: 0.95,
        reasoning: "garbage".to_string(),
    }];

    let result = apply(&database, &recommendations, "test-agent").unwrap();
    assert_eq!(result.deleted, 1);
    assert!(result.errors.is_empty());

    let get_result = database.get_memory("orphan1");
    assert!(
        get_result.is_err(),
        "deleted memory should not be retrievable"
    );
}

#[test]
fn test_apply_archive_sets_indexed_false() {
    let database = Database::in_memory().unwrap();
    let mut memory = make_memory("stale1", "stale context");
    memory.indexed = true;
    database.insert_memory(&memory).unwrap();

    let recommendations = vec![engram_consolidate::Recommendation {
        action: RecommendedAction::Archive {
            memory_id: "stale1".to_string(),
        },
        confidence: 0.9,
        reasoning: "stale".to_string(),
    }];

    let result = apply(&database, &recommendations, "test-agent").unwrap();
    assert_eq!(result.archived, 1);
    assert!(result.errors.is_empty());

    let archived = database.get_memory("stale1").unwrap();
    assert!(!archived.indexed, "archived memory should have indexed=false");
}

#[test]
fn test_apply_logs_to_consolidation_log() {
    let database = Database::in_memory().unwrap();
    database
        .insert_memory(&make_memory("m1", "context one"))
        .unwrap();
    database
        .insert_memory(&make_memory("m2", "context one"))
        .unwrap();

    let recommendations = vec![engram_consolidate::Recommendation {
        action: RecommendedAction::Merge {
            source_id: "m1".to_string(),
            target_id: "m2".to_string(),
        },
        confidence: 0.8,
        reasoning: "test".to_string(),
    }];

    apply(&database, &recommendations, "test-agent").unwrap();

    let count: i64 = database
        .connection()
        .query_row(
            "SELECT COUNT(*) FROM consolidation_log WHERE action = 'merge'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(count, 1, "merge should be logged in consolidation_log");
}

#[test]
fn test_apply_keep_increments_counter() {
    let database = Database::in_memory().unwrap();
    let recommendations = vec![engram_consolidate::Recommendation {
        action: RecommendedAction::Keep {
            memory_id: "m1".to_string(),
        },
        confidence: 0.7,
        reasoning: "keep both".to_string(),
    }];

    let result = apply(&database, &recommendations, "test-agent").unwrap();
    assert_eq!(result.kept, 1);
    assert_eq!(result.merged, 0);
    assert_eq!(result.deleted, 0);
    assert_eq!(result.archived, 0);
}

#[test]
fn test_error_display_codes() {
    let no_candidates = ConsolidateError::NoCandidates;
    assert!(no_candidates.to_string().contains("[5001]"));

    let index_stale = ConsolidateError::IndexStale;
    assert!(index_stale.to_string().contains("[5002]"));

    let invalid_params =
        ConsolidateError::InvalidMergeParams("bad input".to_string());
    assert!(invalid_params.to_string().contains("[5003]"));

    let analysis_failed =
        ConsolidateError::AnalysisFailed("timeout".to_string());
    assert!(analysis_failed.to_string().contains("[5004]"));

    let apply_failed =
        ConsolidateError::ApplyFailed("db locked".to_string());
    assert!(apply_failed.to_string().contains("[5005]"));
}

#[test]
fn test_analyze_with_llm_failure_returns_error() {
    let database = Database::in_memory().unwrap();
    database
        .insert_memory(&make_memory("m1", "rust ownership"))
        .unwrap();
    database
        .insert_memory(&make_memory("m2", "rust ownership"))
        .unwrap();

    let preview_result = PreviewResult {
        duplicates: vec![DuplicateGroup {
            primary_id: "m1".to_string(),
            duplicate_ids: vec!["m2".to_string()],
            similarity: 0.95,
        }],
        stale: Vec::new(),
        garbage: Vec::new(),
    };

    let generator = FailingTextGenerator;
    let result = analyze(&database, &preview_result, Some(&generator));
    assert!(result.is_err(), "LLM failure should propagate as error");
    let error = result.unwrap_err();
    assert!(
        matches!(error, ConsolidateError::AnalysisFailed(_)),
        "should be AnalysisFailed, got: {error}"
    );
}

#[test]
fn test_apply_merge_nonexistent_memory_collects_error() {
    let database = Database::in_memory().unwrap();

    let recommendations = vec![engram_consolidate::Recommendation {
        action: RecommendedAction::Merge {
            source_id: "nonexistent-source".to_string(),
            target_id: "nonexistent-target".to_string(),
        },
        confidence: 0.8,
        reasoning: "test merge nonexistent".to_string(),
    }];

    let result = apply(&database, &recommendations, "test-agent").unwrap();
    assert_eq!(result.merged, 0, "merge should not succeed");
    assert!(
        !result.errors.is_empty(),
        "should collect error for nonexistent memory"
    );
    assert!(
        result.errors[0].contains("nonexistent-target"),
        "error should reference the target memory id"
    );
}

#[test]
fn test_analyze_heuristic_tie_breaks_on_used_count() {
    let database = Database::in_memory().unwrap();
    let mut primary = make_memory("m1", "rust concurrency patterns");
    primary.score = 0.5;
    primary.used_count = 10;
    database.insert_memory(&primary).unwrap();

    let mut duplicate = make_memory("m2", "rust concurrency patterns");
    duplicate.score = 0.5;
    duplicate.used_count = 2;
    database.insert_memory(&duplicate).unwrap();

    let preview_result = PreviewResult {
        duplicates: vec![DuplicateGroup {
            primary_id: "m1".to_string(),
            duplicate_ids: vec!["m2".to_string()],
            similarity: 0.9,
        }],
        stale: Vec::new(),
        garbage: Vec::new(),
    };

    let analysis = analyze(&database, &preview_result, None).unwrap();
    assert_eq!(analysis.recommendations.len(), 1);

    let recommendation = &analysis.recommendations[0];
    match &recommendation.action {
        RecommendedAction::Merge {
            source_id,
            target_id,
        } => {
            assert_eq!(
                source_id, "m1",
                "equal scores: higher used_count (m1=10) should be source"
            );
            assert_eq!(
                target_id, "m2",
                "equal scores: lower used_count (m2=2) should be target"
            );
        }
        other => panic!("expected Merge, got {other:?}"),
    }
}
