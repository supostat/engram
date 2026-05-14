use std::sync::Arc;

use serde::Deserialize;
use serde_json::{Value, json};

use engram_embeddings::ThreeFieldEmbedding;
use engram_storage::Memory;

use crate::error::CoreError;
use crate::persistence::{deterministic_rng, f32_vec_to_bytes, hash_string_to_u64};
use crate::server::ServerState;

#[derive(Deserialize, Default)]
struct ReembedParams {
    /// Reserved for future safety thresholds (e.g., refuse if memory count
    /// exceeds a configured limit). Currently a no-op placeholder per
    /// ADR 2026-05-14-voyage-4-migration-via-reembed-cli §Decision step 5.
    #[serde(default)]
    force: bool,
}

pub async fn handle(state: &Arc<ServerState>, params: Value) -> Result<Value, CoreError> {
    let parsed: ReembedParams = serde_json::from_value(params).unwrap_or_default();
    let _ = parsed.force;

    state.embedder.clear_cache();

    let state_clone = Arc::clone(state);
    tokio::task::spawn_blocking(move || run_reembed(state_clone))
        .await
        .map_err(|error| CoreError::SocketError(error.to_string()))?
}

fn run_reembed(state: Arc<ServerState>) -> Result<Value, CoreError> {
    let memories = {
        let database = state.database.lock().unwrap();
        database.list_all_memories().map_err(CoreError::Storage)?
    };
    let total = memories.len();
    let model = state.embedding_provider.model_name().to_string();

    let mut succeeded = 0usize;
    let mut failed = 0usize;

    for memory in &memories {
        match reembed_one(&state, memory) {
            Ok(()) => succeeded += 1,
            Err(ReembedError::Recoverable) => {
                mark_for_retry(&state, &memory.id)?;
                failed += 1;
            }
            Err(ReembedError::Fatal(error)) => return Err(error),
        }
    }

    if failed == 0 {
        let database = state.database.lock().unwrap();
        crate::migrations::embedding_model_v1::record(&database, &model)?;
    }

    Ok(json!({
        "total": total,
        "succeeded": succeeded,
        "failed": failed,
        "model": model,
    }))
}

enum ReembedError {
    /// Provider or HNSW transient failure — memory stays in the DB with
    /// stale embeddings; `indexed=0` flags it for retry by background
    /// reindex (server mode) or a subsequent reembed invocation.
    Recoverable,
    /// Database write failure or invariant violation — propagate.
    Fatal(CoreError),
}

fn reembed_one(state: &Arc<ServerState>, memory: &Memory) -> Result<(), ReembedError> {
    let provider = state.embedding_provider.as_ref();
    let text_gen = state
        .text_generator
        .as_deref()
        .map(|generator| generator as &dyn engram_llm_client::TextGenerator);

    let embedding = state
        .embedder
        .embed_fields(
            &memory.context,
            &memory.action,
            &memory.result,
            provider,
            text_gen,
        )
        .map_err(|_| ReembedError::Recoverable)?;

    replace_in_hnsw(state, &memory.id, &embedding).map_err(|_| ReembedError::Recoverable)?;
    persist_embeddings(state, &memory.id, &embedding).map_err(ReembedError::Fatal)?;

    Ok(())
}

fn replace_in_hnsw(
    state: &Arc<ServerState>,
    memory_id: &str,
    embedding: &ThreeFieldEmbedding,
) -> Result<(), CoreError> {
    let mut indexes = state.indexes.write().unwrap();
    let hashed = hash_string_to_u64(memory_id);
    if indexes.contains(hashed) {
        indexes.delete(hashed).map_err(CoreError::Hnsw)?;
    }
    let rng_value = deterministic_rng(hashed);
    indexes
        .insert(hashed, memory_id, embedding, rng_value)
        .map_err(CoreError::Hnsw)
}

fn persist_embeddings(
    state: &Arc<ServerState>,
    memory_id: &str,
    embedding: &ThreeFieldEmbedding,
) -> Result<(), CoreError> {
    let database = state.database.lock().unwrap();
    database
        .set_memory_embeddings(
            memory_id,
            &f32_vec_to_bytes(&embedding.context),
            &f32_vec_to_bytes(&embedding.action),
            &f32_vec_to_bytes(&embedding.result),
        )
        .map_err(CoreError::Storage)?;
    database
        .set_memory_indexed(memory_id, true)
        .map_err(CoreError::Storage)
}

fn mark_for_retry(state: &Arc<ServerState>, memory_id: &str) -> Result<(), CoreError> {
    let database = state.database.lock().unwrap();
    database
        .set_memory_indexed(memory_id, false)
        .map_err(CoreError::Storage)
}
