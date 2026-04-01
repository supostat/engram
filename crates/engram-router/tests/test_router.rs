use engram_router::action::{Contextualization, LlmSelection, Proactivity, SearchStrategy};
use engram_router::defaults::defaults_for_mode;
use engram_router::mode::Mode;
use engram_router::router::Router;

#[test]
fn test_router_decide_defaults() {
    let router = Router::new(0.1, 0.0);
    let decision = router.decide(Mode::Debug, 0.5);

    let expected = defaults_for_mode(Mode::Debug);
    assert_eq!(decision.mode, Mode::Debug);
    assert_eq!(decision.search_strategy, expected.search_strategy);
    assert_eq!(decision.llm_selection, expected.llm_selection);
    assert_eq!(decision.contextualization, expected.contextualization);
    assert_eq!(decision.proactivity, expected.proactivity);
}

#[test]
fn test_router_decide_all_modes() {
    let router = Router::new(0.1, 0.0);

    for mode in Mode::all_variants() {
        let decision = router.decide(*mode, 0.5);
        let expected = defaults_for_mode(*mode);

        assert_eq!(decision.mode, *mode);
        assert_eq!(decision.search_strategy, expected.search_strategy);
        assert_eq!(decision.llm_selection, expected.llm_selection);
        assert_eq!(decision.contextualization, expected.contextualization);
        assert_eq!(decision.proactivity, expected.proactivity);
        assert_eq!(decision.similarity_threshold, expected.similarity_threshold,);
    }
}

#[test]
fn test_router_update_and_decide() {
    let mut router = Router::new(0.5, 0.0);

    for _ in 0..20 {
        let mut decision = router.decide(Mode::Debug, 0.5);
        decision.search_strategy = SearchStrategy::LowThreshold;
        router.update(Mode::Debug, &decision, 1.0);
    }

    let decision = router.decide(Mode::Debug, 0.5);
    assert_eq!(decision.search_strategy, SearchStrategy::LowThreshold);
}

#[test]
fn test_router_epsilon_zero_exploits() {
    let mut router = Router::new(0.5, 0.0);

    for _ in 0..10 {
        let mut decision = router.decide(Mode::Coding, 0.5);
        decision.llm_selection = LlmSelection::Expensive;
        router.update(Mode::Coding, &decision, 1.0);
    }

    for _ in 0..10 {
        let decision = router.decide(Mode::Coding, 0.5);
        assert_eq!(decision.llm_selection, LlmSelection::Expensive);
    }
}

#[test]
fn test_router_epsilon_one_explores_all_search_variants() {
    let router = Router::new(0.1, 1.0);
    let variant_count = SearchStrategy::all_variants().len();

    let mut seen = std::collections::HashSet::new();
    for index in 0..variant_count {
        let rng_value = (index as f32 / variant_count as f32) * 1.0 + 0.001;
        let decision = router.decide(Mode::Coding, rng_value);
        seen.insert(decision.search_strategy);
    }

    assert_eq!(
        seen.len(),
        variant_count,
        "epsilon=1.0 should reach all {variant_count} search variants",
    );
}

#[test]
fn test_router_epsilon_half_boundary() {
    let mut router = Router::new(0.5, 0.5);

    for _ in 0..20 {
        let mut decision = router.decide(Mode::Coding, 0.9);
        decision.llm_selection = LlmSelection::Expensive;
        router.update(Mode::Coding, &decision, 1.0);
    }

    // rng_value=0.9 > epsilon=0.5 -> exploit -> trained best
    let exploited = router.decide(Mode::Coding, 0.9);
    assert_eq!(exploited.llm_selection, LlmSelection::Expensive);

    // rng_value=0.1 < epsilon=0.5 -> explore -> random pick
    // just verify it doesn't panic and returns a valid variant
    let explored = router.decide(Mode::Coding, 0.1);
    let valid = [
        LlmSelection::Cheap,
        LlmSelection::Balanced,
        LlmSelection::Expensive,
    ];
    assert!(
        valid.contains(&explored.llm_selection),
        "explored action must be a valid variant",
    );
}

#[test]
fn test_router_decision_memory_type_priority_and_top_k() {
    let router = Router::new(0.1, 0.0);

    for mode in Mode::all_variants() {
        let decision = router.decide(*mode, 0.5);
        let expected = defaults_for_mode(*mode);

        let expected_priority: Vec<String> = expected
            .memory_type_priority
            .iter()
            .map(|s| (*s).to_string())
            .collect();

        assert_eq!(
            decision.memory_type_priority, expected_priority,
            "memory_type_priority mismatch for {:?}",
            mode,
        );
        assert_eq!(
            decision.top_k, expected.top_k_min,
            "top_k mismatch for {:?}",
            mode,
        );
    }
}

#[test]
fn test_router_tie_breaking_returns_first_trained() {
    let mut router = Router::new(0.5, 0.0);

    // Train two actions with identical reward
    let mut decision_a = router.decide(Mode::Debug, 0.5);
    decision_a.search_strategy = SearchStrategy::HighThreshold;
    router.update(Mode::Debug, &decision_a, 1.0);

    let mut decision_b = router.decide(Mode::Debug, 0.5);
    decision_b.search_strategy = SearchStrategy::LowThreshold;
    router.update(Mode::Debug, &decision_b, 1.0);

    // Both have Q=0.5 after one update with reward=1.0, alpha=0.5
    // pick_best_or_default iterates variants in order;
    // HighThreshold comes first => it wins the tie.
    let result = router.decide(Mode::Debug, 0.5);
    assert_eq!(result.search_strategy, SearchStrategy::HighThreshold);
}

#[test]
fn test_router_convergence() {
    let mut router = Router::new(0.1, 0.0);

    for _ in 0..100 {
        let mut decision = router.decide(Mode::Architecture, 0.5);
        decision.search_strategy = SearchStrategy::HighThreshold;
        decision.llm_selection = LlmSelection::Cheap;
        decision.contextualization = Contextualization::Raw;
        decision.proactivity = Proactivity::Passive;
        router.update(Mode::Architecture, &decision, 1.0);
    }

    let decision = router.decide(Mode::Architecture, 0.5);
    assert_eq!(decision.search_strategy, SearchStrategy::HighThreshold);
    assert_eq!(decision.llm_selection, LlmSelection::Cheap);
    assert_eq!(decision.contextualization, Contextualization::Raw);
    assert_eq!(decision.proactivity, Proactivity::Passive);
}
