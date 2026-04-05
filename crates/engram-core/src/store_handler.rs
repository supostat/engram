use std::sync::Arc;

use serde::Deserialize;
use serde_json::{Value, json};

use engram_storage::Memory;

use crate::error::CoreError;
use crate::server::ServerState;
use crate::timestamp::current_utc_timestamp;

const MAX_FIELD_LENGTH: usize = 10_000;

#[derive(Deserialize)]
struct StoreParams {
    memory_type: String,
    context: String,
    action: String,
    result: String,
    tags: Option<String>,
    project: Option<String>,
}

pub async fn handle(state: &Arc<ServerState>, params: Value) -> Result<Value, CoreError> {
    let parsed: StoreParams = serde_json::from_value(params)
        .map_err(|error| CoreError::DispatchError(error.to_string()))?;
    validate_field_length("context", &parsed.context)?;
    validate_field_length("action", &parsed.action)?;
    validate_field_length("result", &parsed.result)?;
    let memory_id = uuid::Uuid::new_v4().to_string();
    let timestamp = current_utc_timestamp();
    let embedding = compute_embedding(state, &parsed).await?;
    let memory = build_memory(&memory_id, &timestamp, &parsed, &embedding);
    persist_memory(state, &memory, &memory_id, &embedding).await?;
    Ok(json!({ "id": memory_id, "indexed": true }))
}

async fn compute_embedding(
    state: &Arc<ServerState>,
    params: &StoreParams,
) -> Result<engram_embeddings::ThreeFieldEmbedding, CoreError> {
    let config = state.config.clone();
    let context = params.context.clone();
    let action = params.action.clone();
    let result = params.result.clone();
    let state_clone = Arc::clone(state);
    tokio::task::spawn_blocking(move || {
        let provider = config.build_embedding_provider()?;
        let text_gen = config.build_text_generator().ok();
        let text_gen_ref = text_gen.as_deref();
        let mut embedder = state_clone.embedder.lock().unwrap();
        embedder
            .embed_fields(&context, &action, &result, provider.as_ref(), text_gen_ref)
            .map_err(|error| {
                CoreError::Api(engram_llm_client::ApiError::EmbeddingApiUnavailable(
                    error.to_string(),
                ))
            })
    })
    .await
    .map_err(|error| CoreError::SocketError(error.to_string()))?
}

fn build_memory(
    memory_id: &str,
    timestamp: &str,
    params: &StoreParams,
    embedding: &engram_embeddings::ThreeFieldEmbedding,
) -> Memory {
    Memory {
        id: memory_id.to_string(),
        memory_type: params.memory_type.clone(),
        context: params.context.clone(),
        action: params.action.clone(),
        result: params.result.clone(),
        score: 0.0,
        embedding_context: Some(f32_vec_to_bytes(&embedding.context)),
        embedding_action: Some(f32_vec_to_bytes(&embedding.action)),
        embedding_result: Some(f32_vec_to_bytes(&embedding.result)),
        indexed: true,
        tags: params.tags.clone(),
        project: params.project.clone(),
        parent_id: None,
        source_ids: None,
        insight_type: None,
        created_at: timestamp.to_string(),
        updated_at: timestamp.to_string(),
        used_count: 0,
        last_used_at: None,
        superseded_by: None,
    }
}

async fn persist_memory(
    state: &Arc<ServerState>,
    memory: &Memory,
    memory_id: &str,
    embedding: &engram_embeddings::ThreeFieldEmbedding,
) -> Result<(), CoreError> {
    let memory_owned = memory.clone();
    let state_db = Arc::clone(state);
    tokio::task::spawn_blocking(move || {
        let database = state_db.database.lock().unwrap();
        database.insert_memory(&memory_owned)?;
        Ok::<(), CoreError>(())
    })
    .await
    .map_err(|error| CoreError::SocketError(error.to_string()))??;

    let hashed_id = hash_string_to_u64(memory_id);
    let rng_value = deterministic_rng(hashed_id);
    let memory_id_owned = memory_id.to_string();
    let embedding_owned = engram_embeddings::ThreeFieldEmbedding {
        context: embedding.context.clone(),
        action: embedding.action.clone(),
        result: embedding.result.clone(),
    };
    let state_idx = Arc::clone(state);
    tokio::task::spawn_blocking(move || {
        let mut indexes = state_idx.indexes.lock().unwrap();
        indexes.insert(hashed_id, &memory_id_owned, &embedding_owned, rng_value)?;
        Ok::<(), CoreError>(())
    })
    .await
    .map_err(|error| CoreError::SocketError(error.to_string()))??;

    Ok(())
}

fn f32_vec_to_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0100_0000_01b3;
const DETERMINISTIC_RNG_MULTIPLIER: u64 = 0x9e37_79b9_7f4a_7c15;

fn hash_string_to_u64(value: &str) -> u64 {
    let mut hash: u64 = FNV_OFFSET_BASIS;
    for byte in value.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

fn deterministic_rng(id: u64) -> f64 {
    let mixed = id.wrapping_mul(DETERMINISTIC_RNG_MULTIPLIER);
    (mixed >> 11) as f64 / (1u64 << 53) as f64
}

fn validate_field_length(field_name: &str, value: &str) -> Result<(), CoreError> {
    if value.len() > MAX_FIELD_LENGTH {
        return Err(CoreError::DispatchError(format!(
            "{field_name} exceeds maximum length of {MAX_FIELD_LENGTH} bytes"
        )));
    }
    Ok(())
}
