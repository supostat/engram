use std::sync::Mutex;

use ndarray::Array2;
use ort::session::Session;
use ort::value::TensorRef;
use tokenizers::Tokenizer;

use crate::error::ApiError;
use crate::provider::TextGenerator;

const DEFAULT_MAX_NEW_TOKENS: usize = 64;

pub struct LocalTextGenerator {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    model_name: String,
}

impl LocalTextGenerator {
    pub fn new(model_path: &str, tokenizer_path: &str) -> Result<Self, ApiError> {
        let session = Session::builder()
            .map_err(|error| ApiError::LocalModelLoadFailed(error.to_string()))?
            .commit_from_file(model_path)
            .map_err(|error| ApiError::LocalModelLoadFailed(error.to_string()))?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|error| ApiError::LocalModelLoadFailed(error.to_string()))?;

        let model_name = extract_model_name(model_path);

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            model_name,
        })
    }
}

fn extract_model_name(model_path: &str) -> String {
    std::path::Path::new(model_path)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("unknown")
        .to_string()
}

fn tokenize(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>, ApiError> {
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|error| ApiError::LocalInferenceFailed(error.to_string()))?;
    Ok(encoding.get_ids().to_vec())
}

fn build_input_array(token_ids: &[u32]) -> Result<Array2<i64>, ApiError> {
    let ids_i64: Vec<i64> = token_ids.iter().map(|&id| id as i64).collect();
    Array2::from_shape_vec((1, ids_i64.len()), ids_i64)
        .map_err(|error| ApiError::LocalInferenceFailed(error.to_string()))
}

fn run_inference(session: &Mutex<Session>, input: &Array2<i64>) -> Result<u32, ApiError> {
    let tensor_ref = TensorRef::from_array_view(input.view())
        .map_err(|error| ApiError::LocalInferenceFailed(error.to_string()))?;

    let mut locked_session = session
        .lock()
        .map_err(|error| ApiError::LocalInferenceFailed(error.to_string()))?;

    let outputs = locked_session
        .run(ort::inputs![tensor_ref])
        .map_err(|error| ApiError::LocalInferenceFailed(error.to_string()))?;

    let logits = outputs[0]
        .try_extract_array::<f32>()
        .map_err(|error| ApiError::LocalInferenceFailed(error.to_string()))?;

    let last_token_logits = logits.slice(ndarray::s![0, -1_isize, ..]).to_owned();

    let next_token_id = last_token_logits
        .iter()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(index, _)| index as u32)
        .ok_or_else(|| {
            ApiError::LocalInferenceFailed("logits are empty or contain only NaN/Inf".into())
        })?;

    Ok(next_token_id)
}

fn decode_tokens(tokenizer: &Tokenizer, token_ids: &[u32]) -> Result<String, ApiError> {
    tokenizer
        .decode(token_ids, true)
        .map_err(|error| ApiError::LocalInferenceFailed(error.to_string()))
}

impl TextGenerator for LocalTextGenerator {
    fn generate(&self, prompt: &str) -> Result<String, ApiError> {
        let mut token_ids = tokenize(&self.tokenizer, prompt)?;
        let prompt_token_count = token_ids.len();

        for _ in 0..DEFAULT_MAX_NEW_TOKENS {
            let input = build_input_array(&token_ids)?;
            let next_token = run_inference(&self.session, &input)?;
            token_ids.push(next_token);
        }

        let generated_ids = &token_ids[prompt_token_count..];
        decode_tokens(&self.tokenizer, generated_ids)
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }
}
