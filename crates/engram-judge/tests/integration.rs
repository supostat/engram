use engram_judge::{CombinedJudge, HeuristicJudge, JudgeError, JudgeInput, LlmJudge};
use engram_llm_client::{ApiError, TextGenerator};

struct MockTextGenerator {
    response: String,
}

impl TextGenerator for MockTextGenerator {
    fn generate(&self, _prompt: &str) -> Result<String, ApiError> {
        Ok(self.response.clone())
    }

    fn model_name(&self) -> &str {
        "mock"
    }
}

struct FailingTextGenerator;

impl TextGenerator for FailingTextGenerator {
    fn generate(&self, _prompt: &str) -> Result<String, ApiError> {
        Err(ApiError::LlmApiUnavailable("mock fail".into()))
    }

    fn model_name(&self) -> &str {
        "failing-mock"
    }
}

fn make_input(context: &str, action: &str, result: &str) -> JudgeInput {
    JudgeInput {
        context: context.to_string(),
        action: action.to_string(),
        result: result.to_string(),
        days_since_update: 0.0,
        used_count: 0,
    }
}

#[test]
fn heuristic_score_full_keyword_match() {
    let input = make_input("rust memory safety", "allocate buffer", "success");
    let score = HeuristicJudge::score("rust memory", &input);

    assert!(score.score > 0.3, "full keyword match should yield high keyword component");
    assert!(!score.degraded);
}

#[test]
fn heuristic_score_no_keyword_match() {
    let input = make_input("python web framework", "deploy", "ok");
    let score = HeuristicJudge::score("rust memory", &input);

    assert!(score.reason.starts_with("keyword:0.00"));
}

#[test]
fn heuristic_score_partial_keyword_match() {
    let input = make_input("rust web server", "handle request", "200 ok");
    let score = HeuristicJudge::score("rust memory", &input);

    assert!(score.reason.contains("keyword:0.50"));
}

#[test]
fn heuristic_recency_recent() {
    let mut input = make_input("test", "test", "test");
    input.days_since_update = 0.0;
    let score = HeuristicJudge::score("test", &input);

    assert!(score.reason.contains("recency:1.00"));
}

#[test]
fn heuristic_recency_old() {
    let mut input = make_input("test", "test", "test");
    input.days_since_update = 90.0;
    let score = HeuristicJudge::score("test", &input);

    assert!(score.reason.contains("recency:0.05"));
}

#[test]
fn heuristic_frequency_unused() {
    let mut input = make_input("test", "test", "test");
    input.used_count = 0;
    let score = HeuristicJudge::score("test", &input);

    assert!(score.reason.contains("frequency:0.00"));
}

#[test]
fn heuristic_frequency_saturated() {
    let mut input = make_input("test", "test", "test");
    input.used_count = 20;
    let score = HeuristicJudge::score("test", &input);

    assert!(score.reason.contains("frequency:1.00"));
}

#[test]
fn heuristic_score_in_valid_range() {
    let mut input = make_input("a b c d e", "f g h", "i j k");
    input.days_since_update = 1000.0;
    input.used_count = 100;
    let score = HeuristicJudge::score("z y x", &input);

    assert!((0.0..=1.0).contains(&score.score));

    input.days_since_update = 0.0;
    let score = HeuristicJudge::score("a b c d e f g h i j k", &input);

    assert!((0.0..=1.0).contains(&score.score));
}

#[test]
fn heuristic_empty_query() {
    let mut input = make_input("rust memory", "allocate", "ok");
    input.days_since_update = 0.0;
    input.used_count = 5;
    let score = HeuristicJudge::score("", &input);

    assert!(score.reason.starts_with("keyword:0.00"));
}

#[test]
fn llm_judge_valid_json() {
    let generator = MockTextGenerator {
        response: r#"{"score": 0.85, "reason": "relevant"}"#.to_string(),
    };
    let judge = LlmJudge::new(&generator);
    let input = make_input("test", "test", "test");
    let result = judge.score("query", &input).unwrap();

    assert!((result.score - 0.85).abs() < 0.001);
    assert_eq!(result.reason, "relevant");
    assert!(!result.degraded);
}

#[test]
fn llm_judge_score_clamped() {
    let generator = MockTextGenerator {
        response: r#"{"score": 1.5, "reason": "too high"}"#.to_string(),
    };
    let judge = LlmJudge::new(&generator);
    let input = make_input("test", "test", "test");
    let result = judge.score("query", &input).unwrap();

    assert!((result.score - 1.0).abs() < 0.001);
}

#[test]
fn llm_judge_score_clamped_negative() {
    let generator = MockTextGenerator {
        response: r#"{"score": -0.5, "reason": "too low"}"#.to_string(),
    };
    let judge = LlmJudge::new(&generator);
    let input = make_input("test", "test", "test");
    let result = judge.score("query", &input).unwrap();

    assert!((result.score - 0.0).abs() < 0.001);
}

#[test]
fn llm_judge_missing_score_field() {
    let generator = MockTextGenerator {
        response: r#"{"reason": "ok"}"#.to_string(),
    };
    let judge = LlmJudge::new(&generator);
    let input = make_input("test", "test", "test");
    let result = judge.score("query", &input);

    assert!(matches!(result, Err(JudgeError::InvalidResponse(_))));
}

#[test]
fn llm_judge_invalid_json() {
    let generator = MockTextGenerator {
        response: "not json".to_string(),
    };
    let judge = LlmJudge::new(&generator);
    let input = make_input("test", "test", "test");
    let result = judge.score("query", &input);

    assert!(matches!(result, Err(JudgeError::InvalidResponse(_))));
}

#[test]
fn llm_judge_missing_reason_uses_empty() {
    let generator = MockTextGenerator {
        response: r#"{"score": 0.5}"#.to_string(),
    };
    let judge = LlmJudge::new(&generator);
    let input = make_input("test", "test", "test");
    let result = judge.score("query", &input).unwrap();

    assert_eq!(result.reason, "");
}

#[test]
fn llm_judge_unavailable() {
    let generator = FailingTextGenerator;
    let judge = LlmJudge::new(&generator);
    let input = make_input("test", "test", "test");
    let result = judge.score("query", &input);

    assert!(matches!(result, Err(JudgeError::LlmUnavailable(_))));
}

#[test]
fn combined_with_llm_success() {
    let generator = MockTextGenerator {
        response: r#"{"score": 0.9, "reason": "great match"}"#.to_string(),
    };
    let judge = CombinedJudge::with_llm(&generator);
    let input = make_input("test", "test", "test");
    let result = judge.score("query", &input);

    assert!((result.score - 0.9).abs() < 0.001);
    assert!(!result.degraded);
}

#[test]
fn combined_with_llm_failure_falls_back() {
    let generator = FailingTextGenerator;
    let judge = CombinedJudge::with_llm(&generator);
    let mut input = make_input("test query", "action", "result");
    input.days_since_update = 1.0;
    input.used_count = 5;
    let result = judge.score("test", &input);

    assert!(result.degraded);
    assert!((0.0..=1.0).contains(&result.score));
}

#[test]
fn combined_heuristic_only() {
    let judge = CombinedJudge::heuristic_only();
    let mut input = make_input("rust memory", "allocate", "ok");
    input.days_since_update = 5.0;
    input.used_count = 3;
    let result = judge.score("rust", &input);

    assert!(!result.degraded);
    assert!((0.0..=1.0).contains(&result.score));
}

#[test]
fn judge_error_display_format() {
    let llm_err = JudgeError::LlmUnavailable("timeout".into());
    assert_eq!(
        format!("{llm_err}"),
        "judge error: llm unavailable: timeout"
    );

    let parse_err = JudgeError::InvalidResponse("bad json".into());
    assert_eq!(
        format!("{parse_err}"),
        "judge error: invalid response: bad json"
    );
}

#[test]
fn judge_error_implements_std_error() {
    fn assert_error<T: std::error::Error>() {}
    assert_error::<JudgeError>();
}

#[test]
fn heuristic_normalize_handles_punctuation() {
    let input = make_input("rust's memory-safety", "", "");
    let score = HeuristicJudge::score("rust s memory safety", &input);

    assert!(score.reason.contains("keyword:1.00"));
}
