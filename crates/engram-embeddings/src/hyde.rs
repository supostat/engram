use engram_llm_client::error::ApiError;
use engram_llm_client::provider::TextGenerator;

const HYDE_PROMPT_TEMPLATE: &str = "Given this short query, generate a detailed hypothetical document \
     that would be a perfect search result. Query: ";

pub fn should_use_hyde(text: &str, threshold: usize) -> bool {
    if threshold == 0 {
        return false;
    }
    if text.trim().is_empty() {
        return false;
    }
    text.split_whitespace().count() < threshold
}

pub fn generate_hypothesis(
    text: &str,
    text_generator: &dyn TextGenerator,
) -> Result<String, ApiError> {
    let prompt = format!("{HYDE_PROMPT_TEMPLATE}{text}");
    text_generator.generate(&prompt)
}
