use crate::score::{JudgeInput, JudgeScore};

const KEYWORD_WEIGHT: f32 = 0.4;
const RECENCY_WEIGHT: f32 = 0.3;
const FREQUENCY_WEIGHT: f32 = 0.3;

pub struct HeuristicJudge;

impl HeuristicJudge {
    pub fn score(query: &str, input: &JudgeInput) -> JudgeScore {
        let keyword_score = Self::keyword_match(query, input);
        let recency_score = Self::recency(input.days_since_update);
        let frequency_score = Self::frequency(input.used_count);

        let combined = keyword_score * KEYWORD_WEIGHT
            + recency_score * RECENCY_WEIGHT
            + frequency_score * FREQUENCY_WEIGHT;

        JudgeScore {
            score: combined.clamp(0.0, 1.0),
            reason: format!(
                "keyword:{keyword_score:.2} recency:{recency_score:.2} frequency:{frequency_score:.2}"
            ),
            degraded: false,
        }
    }

    fn keyword_match(query: &str, input: &JudgeInput) -> f32 {
        let query_words = Self::normalize(query);
        if query_words.is_empty() {
            return 0.0;
        }

        let memory_text = format!("{} {} {}", input.context, input.action, input.result);
        let memory_words = Self::normalize(&memory_text);

        let matches = query_words
            .iter()
            .filter(|word| memory_words.contains(word))
            .count();

        matches as f32 / query_words.len() as f32
    }

    fn recency(days_since_update: f64) -> f32 {
        (-days_since_update / 30.0).exp() as f32
    }

    fn frequency(used_count: u64) -> f32 {
        (used_count as f32 / 10.0).min(1.0)
    }

    fn normalize(text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|character: char| !character.is_alphanumeric())
            .filter(|segment| !segment.is_empty())
            .map(String::from)
            .collect()
    }
}
