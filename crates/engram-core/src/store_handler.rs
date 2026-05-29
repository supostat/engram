use std::sync::Arc;

use serde::Deserialize;
use serde_json::{Value, json};

use engram_storage::Memory;

use crate::error::CoreError;
use crate::lock_helpers;
use crate::persistence::f32_vec_to_bytes;
use crate::server::ServerState;
use crate::tags_normalize::{TagsInput, normalize_tags};
use crate::timestamp::current_utc_timestamp;

const MAX_FIELD_LENGTH: usize = 10_000;

#[derive(Deserialize)]
struct StoreParams {
    memory_type: String,
    context: String,
    action: String,
    result: String,
    tags: Option<TagsInput>,
    project: Option<String>,
}

pub async fn handle(state: &Arc<ServerState>, params: Value) -> Result<Value, CoreError> {
    let mut parsed: StoreParams = serde_json::from_value(params)
        .map_err(|error| CoreError::DispatchError(error.to_string()))?;
    validate_field_length("context", &parsed.context)?;
    validate_field_length("action", &parsed.action)?;
    validate_field_length("result", &parsed.result)?;
    let normalized_tags = normalize_tags(parsed.tags.take());
    let memory_id = uuid::Uuid::new_v4().to_string();
    let timestamp = current_utc_timestamp();
    let embedding = compute_embedding(state, &parsed).await?;
    let memory = build_memory(&memory_id, &timestamp, &parsed, normalized_tags, &embedding);
    let indexed = persist_memory(state, &memory, &memory_id, &embedding).await?;
    Ok(json!({ "id": memory_id, "indexed": indexed }))
}

async fn compute_embedding(
    state: &Arc<ServerState>,
    params: &StoreParams,
) -> Result<engram_embeddings::ThreeFieldEmbedding, CoreError> {
    let context = params.context.clone();
    let action = params.action.clone();
    let result = params.result.clone();
    let state_clone = Arc::clone(state);
    tokio::task::spawn_blocking(move || {
        let provider = state_clone.embedding_provider.as_ref();
        let text_gen_ref = state_clone
            .text_generator
            .as_deref()
            .map(|generator| generator as &dyn engram_llm_client::TextGenerator);
        state_clone
            .embedder
            .embed_fields(&context, &action, &result, provider, text_gen_ref)
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
    normalized_tags: Option<String>,
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
        indexed: false,
        tags: normalized_tags,
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

/// Persists the memory to SQLite (the source of truth, written with
/// `indexed=false`) and then attempts the HNSW index write. Returns the
/// truthful final `indexed` state: `true` only after the HNSW write is
/// confirmed and the row is marked indexed. A transient HNSW failure leaves
/// the row at `indexed=false` for background reindex to recover; a hash
/// collision is non-recoverable and propagates.
async fn persist_memory(
    state: &Arc<ServerState>,
    memory: &Memory,
    memory_id: &str,
    embedding: &engram_embeddings::ThreeFieldEmbedding,
) -> Result<bool, CoreError> {
    insert_row(state, memory).await?;
    match attempt_index(state, memory_id, embedding).await? {
        Ok(()) => {
            mark_indexed(state, memory_id).await?;
            Ok(true)
        }
        Err(collision @ CoreError::IndexHashCollision { .. }) => Err(collision),
        Err(transient) => {
            eprintln!(
                "warning: HNSW index write failed for {memory_id}, \
                 left unindexed for background reindex: {transient}"
            );
            Ok(false)
        }
    }
}

async fn insert_row(state: &Arc<ServerState>, memory: &Memory) -> Result<(), CoreError> {
    let memory_owned = memory.clone();
    let state_db = Arc::clone(state);
    tokio::task::spawn_blocking(move || {
        let database = lock_helpers::lock_db(&state_db);
        database.insert_memory(&memory_owned)?;
        Ok::<(), CoreError>(())
    })
    .await
    .map_err(|error| CoreError::SocketError(error.to_string()))?
}

async fn attempt_index(
    state: &Arc<ServerState>,
    memory_id: &str,
    embedding: &engram_embeddings::ThreeFieldEmbedding,
) -> Result<Result<(), CoreError>, CoreError> {
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
        let mut indexes = lock_helpers::write_indexes(&state_idx);
        indexes.insert_atomic(hashed_id, &memory_id_owned, &embedding_owned, rng_value)
    })
    .await
    .map_err(|error| CoreError::SocketError(error.to_string()))
}

async fn mark_indexed(state: &Arc<ServerState>, memory_id: &str) -> Result<(), CoreError> {
    let memory_id_owned = memory_id.to_string();
    let state_db = Arc::clone(state);
    tokio::task::spawn_blocking(move || {
        let database = lock_helpers::lock_db(&state_db);
        database.set_memory_indexed(&memory_id_owned, true)?;
        Ok::<(), CoreError>(())
    })
    .await
    .map_err(|error| CoreError::SocketError(error.to_string()))?
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
