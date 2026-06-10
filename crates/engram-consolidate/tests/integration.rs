use engram_consolidate::analyze::{RecommendedAction, analyze};
use engram_consolidate::apply::apply;
use engram_consolidate::error::ConsolidateError;
use engram_consolidate::preview::{DuplicateGroup, MatchType, PreviewResult, preview};
use engram_llm_client::{ApiError, TextGenerator};
use engram_storage::Database;
use engram_storage::memory::Memory;

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

struct SelectiveFailingTextGenerator {
    fail_marker: String,
}

impl TextGenerator for SelectiveFailingTextGenerator {
    fn generate(&self, prompt: &str) -> Result<String, ApiError> {
        if prompt.contains(&self.fail_marker) {
            return Err(ApiError::LlmApiUnavailable("selective outage".to_string()));
        }
        Ok("MERGE".to_string())
    }

    fn model_name(&self) -> &str {
        "selective-failing-model"
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

    let result = preview(&database, 365, 0.1, 0.0).unwrap();
    assert!(
        !result.duplicates.is_empty(),
        "should find duplicate memories with identical context"
    );
    assert_eq!(
        result.duplicates[0].match_type,
        MatchType::Fts,
        "context-only overlap is found via FTS, not the exact pass"
    );
}

#[test]
fn test_preview_finds_stale_memories() {
    let database = Database::in_memory().unwrap();
    database.insert_memory(&make_stale_memory("s1")).unwrap();
    database.insert_memory(&make_stale_memory("s2")).unwrap();

    let result = preview(&database, 30, 0.1, 0.0).unwrap();
    assert_eq!(result.stale.len(), 2);
    assert!(result.stale.contains(&"s1".to_string()));
    assert!(result.stale.contains(&"s2".to_string()));
}

#[test]
fn test_preview_finds_garbage_broken_parent() {
    let database = Database::in_memory().unwrap();
    insert_orphan_memory(&database, "orphan1");

    let result = preview(&database, 365, 0.1, 0.0).unwrap();
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

    let result = preview(&database, 365, 0.1, 0.0).unwrap();
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
            match_type: MatchType::Fts,
        }],
        stale: Vec::new(),
        garbage: Vec::new(),
    };

    let generator = MockTextGenerator::responding("MERGE");
    let analysis = analyze(&database, &preview_result, Some(&generator)).unwrap();
    assert_eq!(analysis.analyzed_count, 2);
    assert!(!analysis.recommendations.is_empty());
    assert!(analysis.errors.is_empty());

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
            match_type: MatchType::Fts,
        }],
        stale: Vec::new(),
        garbage: Vec::new(),
    };

    let analysis = analyze(&database, &preview_result, None).unwrap();
    assert_eq!(analysis.analyzed_count, 2);
    assert!(analysis.errors.is_empty());

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

    let result = apply(&database, &recommendations, "test-agent", 0.0).unwrap();
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

    let result = apply(&database, &recommendations, "test-agent", 0.0).unwrap();
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

    let result = apply(&database, &recommendations, "test-agent", 0.0).unwrap();
    assert_eq!(result.archived, 1);
    assert!(result.errors.is_empty());

    let archived = database.get_memory("stale1").unwrap();
    assert!(
        !archived.indexed,
        "archived memory should have indexed=false"
    );
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

    apply(&database, &recommendations, "test-agent", 0.0).unwrap();

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

    let result = apply(&database, &recommendations, "test-agent", 0.0).unwrap();
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

    let invalid_params = ConsolidateError::InvalidMergeParams("bad input".to_string());
    assert!(invalid_params.to_string().contains("[5003]"));

    let analysis_failed = ConsolidateError::AnalysisFailed("timeout".to_string());
    assert!(analysis_failed.to_string().contains("[5004]"));

    let apply_failed = ConsolidateError::ApplyFailed("db locked".to_string());
    assert!(apply_failed.to_string().contains("[5005]"));
}

#[test]
fn test_analyze_with_llm_failure_collects_error_and_continues() {
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
            match_type: MatchType::Fts,
        }],
        stale: Vec::new(),
        garbage: Vec::new(),
    };

    let generator = FailingTextGenerator;
    let analysis = analyze(&database, &preview_result, Some(&generator)).unwrap();
    assert!(
        analysis.recommendations.is_empty(),
        "a failed LLM verdict must not produce a recommendation"
    );
    assert_eq!(analysis.errors.len(), 1, "one failed member, one error");
    assert!(
        analysis.errors[0].contains("m2"),
        "error must reference the failed member, got: {}",
        analysis.errors[0]
    );
    assert_eq!(
        analysis.analyzed_count, 1,
        "only the survivor counts as analyzed when the member call fails"
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

    let result = apply(&database, &recommendations, "test-agent", 0.0).unwrap();
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

fn make_identical(id: &str, shared_token: &str) -> Memory {
    let mut memory = make_memory(id, shared_token);
    memory.action = "identical action body".to_string();
    memory.result = "identical result body".to_string();
    memory
}

// Exact-duplicate grouping must collect ALL rows with a byte-identical
// (context, action, result) triplet into one group, even past the FTS top-5 limit
// that the FTS-only pass would truncate.
#[test]
fn test_preview_exact_dedup_beyond_fts_top5() {
    let database = Database::in_memory().unwrap();
    let shared = "sharedtoken";
    let ids = ["e1", "e2", "e3", "e4", "e5", "e6"];
    for id in ids {
        database.insert_memory(&make_identical(id, shared)).unwrap();
    }

    let result = preview(&database, 365, 0.1, 0.0).unwrap();
    let exact_group = result
        .duplicates
        .iter()
        .find(|group| {
            let mut members: Vec<&str> = std::iter::once(group.primary_id.as_str())
                .chain(group.duplicate_ids.iter().map(String::as_str))
                .collect();
            members.sort();
            members == ids
        })
        .expect("a single exact-duplicate group must contain all six identical rows");
    assert_eq!(
        exact_group.duplicate_ids.len(),
        5,
        "the primary plus five duplicates make all six members"
    );
    assert_eq!(exact_group.similarity, 1.0, "exact duplicates score 1.0");
    assert_eq!(
        exact_group.match_type,
        MatchType::Exact,
        "byte-identical triplets come from the exact pass"
    );
}

// Insights and superseded rows must never be grouped as duplicates of a live source,
// not via the exact pass nor via the FTS pass.
#[test]
fn test_preview_excludes_insights_and_superseded() {
    let database = Database::in_memory().unwrap();
    let shared = "duplicate detection corpus tokens";
    database
        .insert_memory(&make_memory("src1", shared))
        .unwrap();

    let mut insight = make_memory("insight1", shared);
    insight.memory_type = "insight".to_string();
    insight.action = "action for src1".to_string();
    insight.result = "result for src1".to_string();
    database.insert_memory(&insight).unwrap();

    let mut retired = make_memory("retired1", shared);
    retired.action = "action for src1".to_string();
    retired.result = "result for src1".to_string();
    database.insert_memory(&retired).unwrap();
    database.set_superseded_by("retired1", "src1").unwrap();

    let result = preview(&database, 365, 0.1, 0.0).unwrap();
    let grouped_ids: Vec<String> = result
        .duplicates
        .iter()
        .flat_map(|group| {
            std::iter::once(group.primary_id.clone()).chain(group.duplicate_ids.iter().cloned())
        })
        .collect();
    assert!(
        !grouped_ids.contains(&"insight1".to_string()),
        "an insight must never be grouped with a live source"
    );
    assert!(
        !grouped_ids.contains(&"retired1".to_string()),
        "a superseded row must never be grouped with a live source"
    );
}

#[test]
fn test_analyze_single_survivor() {
    let database = Database::in_memory().unwrap();
    let shared = "single survivor corpus";

    let mut insight = make_memory("ins1", shared);
    insight.memory_type = "insight".to_string();
    insight.score = 0.99;
    database.insert_memory(&insight).unwrap();

    let mut high = make_memory("high1", shared);
    high.score = 0.8;
    database.insert_memory(&high).unwrap();

    let mut low = make_memory("low1", shared);
    low.score = 0.2;
    database.insert_memory(&low).unwrap();

    let preview_result = PreviewResult {
        duplicates: vec![DuplicateGroup {
            primary_id: "ins1".to_string(),
            duplicate_ids: vec!["high1".to_string(), "low1".to_string()],
            similarity: 1.0,
            match_type: MatchType::Exact,
        }],
        stale: Vec::new(),
        garbage: Vec::new(),
    };

    let analysis = analyze(&database, &preview_result, None).unwrap();
    assert_eq!(
        analysis.recommendations.len(),
        2,
        "a 3-member group produces exactly two non-survivor recommendations"
    );

    let mut survivors = std::collections::HashSet::new();
    let mut targets = std::collections::HashSet::new();
    for recommendation in &analysis.recommendations {
        match &recommendation.action {
            RecommendedAction::Merge {
                source_id,
                target_id,
            } => {
                survivors.insert(source_id.clone());
                targets.insert(target_id.clone());
            }
            other => panic!("expected Merge, got {other:?}"),
        }
    }
    assert_eq!(survivors.len(), 1, "every merge must share one survivor");
    let survivor = survivors.into_iter().next().unwrap();
    assert_eq!(
        survivor, "high1",
        "the highest-score non-insight must be the survivor"
    );
    assert_ne!(survivor, "ins1", "an insight must never be the survivor");
    assert_eq!(
        targets,
        ["high1", "low1", "ins1"]
            .into_iter()
            .filter(|id| *id != survivor)
            .map(str::to_string)
            .collect()
    );
}

#[test]
fn test_apply_returns_pruned_ids() {
    let database = Database::in_memory().unwrap();
    database
        .insert_memory(&make_memory("survivor", "merge survivor context"))
        .unwrap();
    database
        .insert_memory(&make_memory("merged", "merge survivor context"))
        .unwrap();
    insert_orphan_memory(&database, "garbage");
    let mut stale = make_memory("stale", "stale archive context");
    stale.indexed = true;
    database.insert_memory(&stale).unwrap();

    let recommendations = vec![
        engram_consolidate::Recommendation {
            action: RecommendedAction::Merge {
                source_id: "survivor".to_string(),
                target_id: "merged".to_string(),
            },
            confidence: 0.8,
            reasoning: "merge".to_string(),
        },
        engram_consolidate::Recommendation {
            action: RecommendedAction::Delete {
                memory_id: "garbage".to_string(),
            },
            confidence: 0.95,
            reasoning: "delete".to_string(),
        },
        engram_consolidate::Recommendation {
            action: RecommendedAction::Archive {
                memory_id: "stale".to_string(),
            },
            confidence: 0.9,
            reasoning: "archive".to_string(),
        },
        engram_consolidate::Recommendation {
            action: RecommendedAction::Keep {
                memory_id: "survivor".to_string(),
            },
            confidence: 0.7,
            reasoning: "keep".to_string(),
        },
    ];

    let result = apply(&database, &recommendations, "test-agent", 0.0).unwrap();
    let pruned: std::collections::HashSet<&str> =
        result.pruned_ids.iter().map(String::as_str).collect();
    assert!(pruned.contains("merged"), "merge target must be pruned");
    assert!(pruned.contains("garbage"), "deleted id must be pruned");
    assert!(pruned.contains("stale"), "archived id must be pruned");
    assert!(
        !pruned.contains("survivor"),
        "the merge survivor must NOT be pruned"
    );
    assert_eq!(
        result.pruned_ids.len(),
        3,
        "kept survivor adds nothing to pruned_ids"
    );
}

#[test]
fn test_apply_min_confidence_gate() {
    let database = Database::in_memory().unwrap();
    database
        .insert_memory(&make_memory("g1", "confidence gate context"))
        .unwrap();
    database
        .insert_memory(&make_memory("g2", "confidence gate context"))
        .unwrap();

    let recommendation = engram_consolidate::Recommendation {
        action: RecommendedAction::Merge {
            source_id: "g1".to_string(),
            target_id: "g2".to_string(),
        },
        confidence: 0.6,
        reasoning: "borderline".to_string(),
    };

    let skipped = apply(
        &database,
        std::slice::from_ref(&recommendation),
        "test-agent",
        0.7,
    )
    .unwrap();
    assert_eq!(skipped.merged, 0, "a rec below min_confidence is skipped");
    assert!(skipped.pruned_ids.is_empty());

    let applied = apply(
        &database,
        std::slice::from_ref(&recommendation),
        "test-agent",
        0.5,
    )
    .unwrap();
    assert_eq!(applied.merged, 1, "a rec at/above min_confidence applies");
    assert_eq!(applied.pruned_ids, vec!["g2".to_string()]);
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
            match_type: MatchType::Fts,
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

// The reported FTS similarity must average only the ranks of the memories that
// actually end up in the group. The old code averaged over every FTS match,
// including already-grouped memories that were filtered out of the member list.
#[test]
fn test_preview_fts_similarity_uses_only_group_members() {
    let database = Database::in_memory().unwrap();
    database
        .insert_memory(&make_identical("a1", "alpha beta sigma tau"))
        .unwrap();
    database
        .insert_memory(&make_identical("b1", "alpha beta sigma tau"))
        .unwrap();

    let mut probe = make_memory("c1", "alpha beta gamma epsilon");
    probe.created_at = "2025-01-02T00:00:00Z".to_string();
    probe.updated_at = "2025-01-02T00:00:00Z".to_string();
    database.insert_memory(&probe).unwrap();

    let mut strong_match = make_memory("d1", "alpha beta gamma epsilon");
    strong_match.created_at = "2025-01-03T00:00:00Z".to_string();
    strong_match.updated_at = "2025-01-03T00:00:00Z".to_string();
    database.insert_memory(&strong_match).unwrap();

    let expected_similarity = database
        .search_fts(&probe.context, 5)
        .unwrap()
        .into_iter()
        .find(|fts_result| fts_result.memory.id == "d1")
        .map(|fts_result| fts_result.rank.abs() as f32)
        .expect("d1 shares every probe token and must appear in FTS results");

    let result = preview(&database, 365, 0.1, 0.0).unwrap();
    let probe_group = result
        .duplicates
        .iter()
        .find(|group| group.primary_id == "c1")
        .expect("the probe must form an FTS group");
    assert_eq!(probe_group.duplicate_ids, vec!["d1".to_string()]);
    assert_eq!(probe_group.match_type, MatchType::Fts);
    // Relative tolerance: |bm25| magnitudes shrink with corpus size, so an
    // absolute epsilon could swallow the difference between member-scoped
    // and all-matches averaging.
    assert!(
        (probe_group.similarity - expected_similarity).abs() < expected_similarity * 1e-3,
        "similarity must be the mean |bm25| of group members only, \
         expected {expected_similarity}, got {}",
        probe_group.similarity
    );
}

// The floor filters only FTS groups; exact groups always survive, and a 0.0
// floor keeps the pre-floor behavior.
#[test]
fn test_preview_fts_floor_drops_weak_groups_keeps_exact() {
    let database = Database::in_memory().unwrap();
    database
        .insert_memory(&make_identical("x1", "omicron lambda kappa"))
        .unwrap();
    database
        .insert_memory(&make_identical("x2", "omicron lambda kappa"))
        .unwrap();
    database
        .insert_memory(&make_memory("f1", "zeta theta iota"))
        .unwrap();
    database
        .insert_memory(&make_memory("f2", "zeta theta iota"))
        .unwrap();

    let filtered = preview(&database, 365, 0.1, 1e6).unwrap();
    assert_eq!(
        filtered.duplicates.len(),
        1,
        "an unreachable floor must drop the FTS group and keep the exact group"
    );
    assert_eq!(filtered.duplicates[0].match_type, MatchType::Exact);

    let unfiltered = preview(&database, 365, 0.1, 0.0).unwrap();
    assert_eq!(
        unfiltered.duplicates.len(),
        2,
        "a 0.0 floor must keep both the exact and the FTS group"
    );
}

#[test]
fn test_analyze_partial_llm_failure_keeps_other_groups() {
    let database = Database::in_memory().unwrap();
    database
        .insert_memory(&make_memory("ok1", "healthy corpus tokens"))
        .unwrap();
    database
        .insert_memory(&make_memory("ok2", "healthy corpus tokens"))
        .unwrap();
    database
        .insert_memory(&make_memory("p1", "poisoned corpus tokens"))
        .unwrap();
    database
        .insert_memory(&make_memory("p2", "poisontoken corpus tokens"))
        .unwrap();

    let preview_result = PreviewResult {
        duplicates: vec![
            DuplicateGroup {
                primary_id: "ok1".to_string(),
                duplicate_ids: vec!["ok2".to_string()],
                similarity: 0.9,
                match_type: MatchType::Fts,
            },
            DuplicateGroup {
                primary_id: "p1".to_string(),
                duplicate_ids: vec!["p2".to_string()],
                similarity: 0.9,
                match_type: MatchType::Fts,
            },
        ],
        stale: Vec::new(),
        garbage: Vec::new(),
    };

    let generator = SelectiveFailingTextGenerator {
        fail_marker: "poisontoken".to_string(),
    };
    let analysis = analyze(&database, &preview_result, Some(&generator)).unwrap();
    assert_eq!(
        analysis.recommendations.len(),
        1,
        "the healthy group must still produce its recommendation"
    );
    assert!(
        matches!(
            &analysis.recommendations[0].action,
            RecommendedAction::Merge { source_id, target_id }
                if source_id == "ok1" && target_id == "ok2"
        ),
        "expected Merge ok2 -> ok1, got {:?}",
        analysis.recommendations[0].action
    );
    assert_eq!(analysis.errors.len(), 1);
    assert!(
        analysis.errors[0].contains("p2"),
        "error must reference the poisoned member, got: {}",
        analysis.errors[0]
    );
    assert_eq!(
        analysis.analyzed_count, 3,
        "two healthy members plus the poisoned group's survivor"
    );
}

#[test]
fn test_analyze_missing_member_collects_error() {
    let database = Database::in_memory().unwrap();

    let preview_result = PreviewResult {
        duplicates: vec![DuplicateGroup {
            primary_id: "ghost-primary".to_string(),
            duplicate_ids: vec!["ghost-duplicate".to_string()],
            similarity: 0.9,
            match_type: MatchType::Fts,
        }],
        stale: Vec::new(),
        garbage: Vec::new(),
    };

    let analysis = analyze(&database, &preview_result, None).unwrap();
    assert!(analysis.recommendations.is_empty());
    assert_eq!(analysis.errors.len(), 1);
    assert!(
        analysis.errors[0].starts_with("analyze group ghost-primary:"),
        "error must name the unloadable group, got: {}",
        analysis.errors[0]
    );
    assert_eq!(analysis.analyzed_count, 0);
}
