use crate::error::JudgeError;
use crate::score::{JudgeInput, JudgeScore};
use engram_llm_client::TextGenerator;

const JUDGE_PROMPT_TEMPLATE: &str = "\
Rate the relevance of this memory to the query on a scale of 0.0 to 1.0.\n\
\n\
Query: {query}\n\
\n\
Memory:\n\
- Context: {context}\n\
- Action: {action}\n\
- Result: {result}\n\
\n\
Respond with ONLY a JSON object: {\"score\": <float>, \"reason\": \"<string>\"}";

pub struct LlmJudge<'generator> {
    text_generator: &'generator dyn TextGenerator,
}

impl<'generator> LlmJudge<'generator> {
    pub fn new(text_generator: &'generator dyn TextGenerator) -> Self {
        Self { text_generator }
    }

    pub fn score(&self, query: &str, input: &JudgeInput) -> Result<JudgeScore, JudgeError> {
        let prompt = JUDGE_PROMPT_TEMPLATE
            .replace("{query}", query)
            .replace("{context}", &input.context)
            .replace("{action}", &input.action)
            .replace("{result}", &input.result);

        let response = self
            .text_generator
            .generate(&prompt)
            .map_err(|api_error| JudgeError::LlmUnavailable(api_error.to_string()))?;

        Self::parse_response(&response)
    }

    fn parse_response(response: &str) -> Result<JudgeScore, JudgeError> {
        let parsed: serde_json::Value = serde_json::from_str(response)
            .map_err(|parse_error| JudgeError::InvalidResponse(parse_error.to_string()))?;

        let score = parsed["score"]
            .as_f64()
            .ok_or_else(|| JudgeError::InvalidResponse("missing score field".into()))?
            as f32;

        let reason = parsed["reason"].as_str().unwrap_or("").to_string();

        Ok(JudgeScore {
            score: score.clamp(0.0, 1.0),
            reason,
            degraded: false,
        })
    }
}
