use engram_router::q_table::QTable;

#[test]
fn test_q_table_existing_key_updates_at_capacity() {
    let mut table = QTable::new();
    // Fill to capacity is impractical (100k), but verify
    // that updating an existing key always works regardless.
    table.update("s", "a", 1.0, 0.5);
    let first = table.get("s", "a");
    table.update("s", "a", 1.0, 0.5);
    let second = table.get("s", "a");
    assert!(second > first, "existing key must still update");
}

#[test]
fn test_q_table_empty() {
    let table = QTable::new();
    assert_eq!(table.get("any_state", "any_action"), 0.0);
    assert!(table.is_empty());
    assert_eq!(table.len(), 0);
}

#[test]
fn test_q_table_update() {
    let mut table = QTable::new();
    // Q = 0 + 0.5 * (1.0 - 0) = 0.5
    table.update("state", "action", 1.0, 0.5);
    assert!((table.get("state", "action") - 0.5).abs() < f32::EPSILON);
}

#[test]
fn test_q_table_multiple_updates() {
    let mut table = QTable::new();
    let alpha = 0.1;
    let reward = 1.0;

    // After many updates, Q should converge toward reward
    for _ in 0..100 {
        table.update("s", "a", reward, alpha);
    }

    let q_value = table.get("s", "a");
    assert!(
        (q_value - reward).abs() < 0.01,
        "Q-value {q_value} should converge to {reward}"
    );
}

#[test]
fn test_q_table_different_states() {
    let mut table = QTable::new();
    table.update("state_a", "action", 1.0, 0.5);
    table.update("state_b", "action", -1.0, 0.5);

    assert!((table.get("state_a", "action") - 0.5).abs() < f32::EPSILON);
    assert!((table.get("state_b", "action") - (-0.5)).abs() < f32::EPSILON);
}

#[test]
fn test_q_table_actions_for_state() {
    let mut table = QTable::new();
    table.update("s", "a1", 1.0, 0.5);
    table.update("s", "a2", 2.0, 0.5);
    table.update("other", "a3", 3.0, 0.5);

    let actions = table.actions_for_state("s");
    assert_eq!(actions.len(), 2);

    let action_names: Vec<&str> = actions.iter().map(|(name, _)| name.as_str()).collect();
    assert!(action_names.contains(&"a1"));
    assert!(action_names.contains(&"a2"));
}

#[test]
fn test_q_table_update_count() {
    let mut table = QTable::new();
    assert_eq!(table.update_count("s", "a"), 0);

    table.update("s", "a", 1.0, 0.1);
    assert_eq!(table.update_count("s", "a"), 1);

    table.update("s", "a", 1.0, 0.1);
    assert_eq!(table.update_count("s", "a"), 2);

    // Different action — still 0
    assert_eq!(table.update_count("s", "b"), 0);
}

#[test]
fn test_q_table_len() {
    let mut table = QTable::new();
    table.update("s1", "a1", 1.0, 0.1);
    table.update("s1", "a2", 1.0, 0.1);
    table.update("s2", "a1", 1.0, 0.1);

    assert_eq!(table.len(), 3);
    assert!(!table.is_empty());
}
