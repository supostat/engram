use engram_router::action::{Contextualization, LlmSelection, Proactivity, SearchStrategy};
use engram_router::error::RouterError;
use engram_router::mode::Mode;

#[test]
fn test_mode_from_str_all() {
    assert_eq!(Mode::parse("debug").unwrap(), Mode::Debug);
    assert_eq!(Mode::parse("architecture").unwrap(), Mode::Architecture);
    assert_eq!(Mode::parse("coding").unwrap(), Mode::Coding);
    assert_eq!(Mode::parse("review").unwrap(), Mode::Review);
    assert_eq!(Mode::parse("plan").unwrap(), Mode::Plan);
    assert_eq!(Mode::parse("routine").unwrap(), Mode::Routine);
}

#[test]
fn test_mode_from_str_case_insensitive() {
    assert_eq!(Mode::parse("DEBUG").unwrap(), Mode::Debug);
    assert_eq!(Mode::parse("Debug").unwrap(), Mode::Debug);
    assert_eq!(Mode::parse("debug").unwrap(), Mode::Debug);
    assert_eq!(Mode::parse("DeBuG").unwrap(), Mode::Debug);
}

#[test]
fn test_mode_from_str_unknown() {
    let result = Mode::parse("nonexistent");
    assert_eq!(
        result,
        Err(RouterError::UnknownMode("nonexistent".to_string()))
    );
}

#[test]
fn test_mode_detect_debug() {
    assert_eq!(Mode::detect("found a bug in the trace"), Mode::Debug);
    assert_eq!(Mode::detect("there is an error here"), Mode::Debug);
    assert_eq!(Mode::detect("application crash report"), Mode::Debug);
}

#[test]
fn test_mode_detect_architecture() {
    assert_eq!(
        Mode::detect("design the component structure"),
        Mode::Architecture
    );
    assert_eq!(
        Mode::detect("choose a framework for this"),
        Mode::Architecture
    );
}

#[test]
fn test_mode_detect_coding() {
    assert_eq!(Mode::detect("implement the feature"), Mode::Coding);
    assert_eq!(Mode::detect("write a function for parsing"), Mode::Coding);
}

#[test]
fn test_mode_detect_review() {
    assert_eq!(Mode::detect("refactor this code"), Mode::Review);
    assert_eq!(Mode::detect("improve code quality"), Mode::Review);
}

#[test]
fn test_mode_detect_plan() {
    assert_eq!(Mode::detect("estimate the timeline"), Mode::Plan);
    assert_eq!(Mode::detect("assess the risk and scope"), Mode::Plan);
}

#[test]
fn test_mode_detect_routine() {
    assert_eq!(Mode::detect("update dependencies"), Mode::Routine);
    assert_eq!(Mode::detect("bump the version"), Mode::Routine);
}

#[test]
fn test_mode_detect_fallback() {
    assert_eq!(Mode::detect("hello world"), Mode::Routine);
    assert_eq!(Mode::detect(""), Mode::Routine);
}

#[test]
fn test_mode_detect_priority() {
    // "fix" → Debug, "design" → Architecture. Debug has higher priority.
    assert_eq!(Mode::detect("fix the design bug"), Mode::Debug);
    // "plan" > "architecture"
    assert_eq!(Mode::detect("plan the design structure"), Mode::Plan);
}

#[test]
fn test_mode_as_str_roundtrip() {
    for mode in Mode::all_variants() {
        let name = mode.as_str();
        assert_eq!(Mode::parse(name).unwrap(), *mode);
    }
}

#[test]
fn test_action_parse_unknown_returns_unknown_action() {
    let result = SearchStrategy::parse("bogus");
    assert_eq!(result, Err(RouterError::UnknownAction("bogus".to_string())),);

    let result = LlmSelection::parse("bogus");
    assert_eq!(result, Err(RouterError::UnknownAction("bogus".to_string())),);

    let result = Contextualization::parse("bogus");
    assert_eq!(result, Err(RouterError::UnknownAction("bogus".to_string())),);

    let result = Proactivity::parse("bogus");
    assert_eq!(result, Err(RouterError::UnknownAction("bogus".to_string())),);
}
