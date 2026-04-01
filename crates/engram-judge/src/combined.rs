use crate::heuristic::HeuristicJudge;
use crate::llm_judge::LlmJudge;
use crate::score::{JudgeInput, JudgeScore};
use engram_llm_client::TextGenerator;

pub struct CombinedJudge<'generator> {
    llm: Option<LlmJudge<'generator>>,
}

impl<'generator> CombinedJudge<'generator> {
    pub fn with_llm(text_generator: &'generator dyn TextGenerator) -> Self {
        Self {
            llm: Some(LlmJudge::new(text_generator)),
        }
    }

    pub fn heuristic_only() -> Self {
        Self { llm: None }
    }

    pub fn score(&self, query: &str, input: &JudgeInput) -> JudgeScore {
        if let Some(llm) = &self.llm {
            match llm.score(query, input) {
                Ok(llm_score) => return llm_score,
                Err(_) => {
                    let mut fallback_score = HeuristicJudge::score(query, input);
                    fallback_score.degraded = true;
                    return fallback_score;
                }
            }
        }

        HeuristicJudge::score(query, input)
    }
}
