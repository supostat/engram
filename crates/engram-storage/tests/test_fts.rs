use engram_storage::Database;
use engram_storage::memory::Memory;

fn make_memory(id: &str, context: &str, action: &str, result: &str) -> Memory {
    Memory {
        id: id.to_string(),
        memory_type: "decision".to_string(),
        context: context.to_string(),
        action: action.to_string(),
        result: result.to_string(),
        score: 0.5,
        embedding_context: None,
        embedding_action: None,
        embedding_result: None,
        indexed: false,
        tags: None,
        project: None,
        parent_id: None,
        source_ids: None,
        insight_type: None,
        created_at: "2026-01-01T00:00:00Z".to_string(),
        updated_at: "2026-01-01T00:00:00Z".to_string(),
        used_count: 0,
        last_used_at: None,
        superseded_by: None,
    }
}

#[test]
fn test_fts_search_basic() {
    let database = Database::in_memory().unwrap();
    database
        .insert_memory(&make_memory(
            "m1",
            "rust ownership model",
            "use borrow checker",
            "safe memory management",
        ))
        .unwrap();
    database
        .insert_memory(&make_memory(
            "m2",
            "python gc",
            "use garbage collector",
            "cleaned up objects",
        ))
        .unwrap();
    database
        .insert_memory(&make_memory(
            "m3",
            "rust async runtime",
            "use tokio",
            "concurrent execution",
        ))
        .unwrap();

    let results = database.search_fts("rust", 10).unwrap();
    assert_eq!(results.len(), 2);

    let ids: Vec<&str> = results.iter().map(|r| r.memory.id.as_str()).collect();
    assert!(ids.contains(&"m1"));
    assert!(ids.contains(&"m3"));
}

#[test]
fn test_fts_search_ranking() {
    let database = Database::in_memory().unwrap();
    // m1 has "rust" in all three fields — should rank higher
    database
        .insert_memory(&make_memory(
            "m1",
            "rust language",
            "rust borrow checker",
            "rust safety",
        ))
        .unwrap();
    // m2 has "rust" only in context
    database
        .insert_memory(&make_memory(
            "m2",
            "rust overview",
            "read documentation",
            "learned basics",
        ))
        .unwrap();

    let results = database.search_fts("rust", 10).unwrap();
    assert_eq!(results.len(), 2);
    // Better rank = lower (more negative) value
    assert!(results[0].rank <= results[1].rank);
}

#[test]
fn test_fts_search_no_results() {
    let database = Database::in_memory().unwrap();
    database
        .insert_memory(&make_memory("m1", "context", "action", "result"))
        .unwrap();

    let results = database.search_fts("nonexistent_term_xyz", 10).unwrap();
    assert!(results.is_empty());
}

#[test]
fn test_fts_search_limit() {
    let database = Database::in_memory().unwrap();
    for i in 0..10 {
        database
            .insert_memory(&make_memory(
                &format!("m{i}"),
                "shared keyword searchable",
                "action text",
                "result text",
            ))
            .unwrap();
    }

    let results = database.search_fts("searchable", 3).unwrap();
    assert_eq!(results.len(), 3);
}

#[test]
fn test_fts_search_after_update() {
    let database = Database::in_memory().unwrap();
    database
        .insert_memory(&make_memory(
            "m1",
            "original context",
            "original action",
            "original result",
        ))
        .unwrap();

    // FTS trigger fires on UPDATE — update via raw SQL to test trigger
    database
        .connection()
        .execute(
            "UPDATE memories SET context = 'updated butterfly context' WHERE id = 'm1'",
            [],
        )
        .unwrap();

    let old_results = database.search_fts("original", 10).unwrap();
    // "original" still in action and result columns
    assert_eq!(old_results.len(), 1);

    let new_results = database.search_fts("butterfly", 10).unwrap();
    assert_eq!(new_results.len(), 1);
    assert_eq!(new_results[0].memory.id, "m1");
}

#[test]
fn test_fts_search_special_characters() {
    let database = Database::in_memory().unwrap();
    database
        .insert_memory(&make_memory(
            "m1",
            "some context",
            "some action",
            "some result",
        ))
        .unwrap();

    let special_queries = [
        "\"unclosed quote",
        "term AND OR NOT",
        "col:value",
        "prefix*",
        "(unbalanced",
        "a + b",
        "some-context",
        "dash-injection OR 1=1",
        "***",
        "@#$%^&",
    ];
    for query in &special_queries {
        let results = database.search_fts(query, 10);
        assert!(results.is_ok(), "query {query:?} must not error");
    }

    let all_specials = database.search_fts("@#$%^&", 10).unwrap();
    assert!(all_specials.is_empty(), "all-special-chars query must return empty");
}
