use std::collections::HashMap;
use std::sync::Arc;

use serde::Deserialize;
use serde_json::{Value, json};

use engram_router::Mode;

use crate::error::CoreError;
use crate::server::ServerState;
use crate::timestamp::current_utc_timestamp;

const VECTOR_WEIGHT: f64 = 0.7;
const SPARSE_WEIGHT: f64 = 0.3;
const MAX_QUERY_LENGTH: usize = 5_000;

#[derive(Deserialize)]
#[allow(dead_code)]
struct SearchParams {
    query: String,
    limit: Option<usize>,
    mode: Option<String>,
    project: Option<String>,
}

pub async fn handle(state: &Arc<ServerState>, params: Value) -> Result<Value, CoreError> {
    let parsed: SearchParams = serde_json::from_value(params)
        .map_err(|error| CoreError::DispatchError(error.to_string()))?;
    if parsed.query.len() > MAX_QUERY_LENGTH {
        return Err(CoreError::DispatchError(format!(
            "query exceeds maximum length of {MAX_QUERY_LENGTH} bytes"
        )));
    }
    let detected_mode = resolve_mode(&parsed);
    let top_k = resolve_top_k(&parsed, state, detected_mode);
    let query_embedding = embed_query(state, &parsed.query).await?;
    let vector_results = search_vector_index(state, &query_embedding, top_k).await?;
    let sparse_results = search_fts(state, &parsed.query, top_k).await?;
    let merged = merge_results(&vector_results, &sparse_results);
    let limited = limit_results(merged, top_k);
    let memories = load_memories(state, &limited).await?;
    Ok(json!(memories))
}

fn resolve_mode(params: &SearchParams) -> Mode {
    match &params.mode {
        Some(mode_string) => {
            Mode::parse(mode_string).unwrap_or_else(|_| Mode::detect(&params.query))
        }
        None => Mode::detect(&params.query),
    }
}

fn resolve_top_k(params: &SearchParams, state: &Arc<ServerState>, mode: Mode) -> usize {
    if let Some(limit) = params.limit {
        return limit;
    }
    let router = state.router.lock().unwrap();
    let decision = router.decide(mode, 0.5);
    decision.top_k
}

async fn embed_query(state: &Arc<ServerState>, query: &str) -> Result<Vec<f32>, CoreError> {
    let config = state.config.clone();
    let query_owned = query.to_string();
    let state_clone = Arc::clone(state);
    tokio::task::spawn_blocking(move || {
        let provider = config.build_embedding_provider()?;
        let text_gen = config.build_text_generator().ok();
        let text_gen_ref = text_gen.as_deref();
        let mut embedder = state_clone.embedder.lock().unwrap();
        embedder
            .embed_query(&query_owned, provider.as_ref(), text_gen_ref)
            .map_err(|error| {
                CoreError::Api(engram_llm_client::ApiError::EmbeddingApiUnavailable(
                    error.to_string(),
                ))
            })
    })
    .await
    .map_err(|error| CoreError::SocketError(error.to_string()))?
}

async fn search_vector_index(
    state: &Arc<ServerState>,
    query_embedding: &[f32],
    top_k: usize,
) -> Result<Vec<(u64, f32)>, CoreError> {
    let embedding_owned = query_embedding.to_vec();
    let state_clone = Arc::clone(state);
    tokio::task::spawn_blocking(move || {
        let indexes = state_clone.indexes.lock().unwrap();
        indexes
            .search(&embedding_owned, top_k)
            .map_err(CoreError::Hnsw)
    })
    .await
    .map_err(|error| CoreError::SocketError(error.to_string()))?
}

async fn search_fts(
    state: &Arc<ServerState>,
    query: &str,
    top_k: usize,
) -> Result<Vec<(String, f64)>, CoreError> {
    let query_owned = query.to_string();
    let state_clone = Arc::clone(state);
    tokio::task::spawn_blocking(move || {
        let database = state_clone.database.lock().unwrap();
        let fts_results = database.search_fts(&query_owned, top_k)?;
        let scored: Vec<(String, f64)> = fts_results
            .into_iter()
            .map(|fts| (fts.memory.id, fts.rank.abs()))
            .collect();
        Ok::<_, CoreError>(scored)
    })
    .await
    .map_err(|error| CoreError::SocketError(error.to_string()))?
}

fn merge_results(
    vector_results: &[(u64, f32)],
    sparse_results: &[(String, f64)],
) -> Vec<(String, f64)> {
    let mut combined: HashMap<String, f64> = HashMap::new();
    let vector_max = vector_results
        .iter()
        .map(|(_, score)| *score as f64)
        .fold(0.0f64, f64::max)
        .max(1.0);
    for &(id_hash, score) in vector_results {
        let normalized = score as f64 / vector_max;
        let key = format!("hnsw:{id_hash}");
        combined
            .entry(key)
            .and_modify(|existing| *existing = existing.max(normalized * VECTOR_WEIGHT))
            .or_insert(normalized * VECTOR_WEIGHT);
    }
    let sparse_max = sparse_results
        .iter()
        .map(|(_, rank)| *rank)
        .fold(0.0f64, f64::max)
        .max(1.0);
    for (memory_id, rank) in sparse_results {
        let normalized = *rank / sparse_max;
        combined
            .entry(memory_id.clone())
            .and_modify(|existing| *existing += normalized * SPARSE_WEIGHT)
            .or_insert(normalized * SPARSE_WEIGHT);
    }
    let mut results: Vec<(String, f64)> = combined.into_iter().collect();
    results.sort_by(|a, b| b.1.total_cmp(&a.1));
    results
}

fn limit_results(results: Vec<(String, f64)>, top_k: usize) -> Vec<(String, f64)> {
    results.into_iter().take(top_k).collect()
}

async fn load_memories(
    state: &Arc<ServerState>,
    scored_results: &[(String, f64)],
) -> Result<Vec<Value>, CoreError> {
    let memory_ids: Vec<String> = scored_results
        .iter()
        .filter(|(key, _)| !key.starts_with("hnsw:"))
        .map(|(id, _)| id.clone())
        .collect();
    let scores: HashMap<String, f64> = scored_results.iter().cloned().collect();
    let state_clone = Arc::clone(state);
    tokio::task::spawn_blocking(move || {
        let database = state_clone.database.lock().unwrap();
        let timestamp = current_utc_timestamp();
        let mut results = Vec::new();
        for memory_id in &memory_ids {
            let memory = match database.get_memory(memory_id) {
                Ok(m) => m,
                Err(_) => continue,
            };
            let _ = database.track_search(memory_id, &timestamp);
            let score = scores.get(memory_id.as_str()).copied().unwrap_or(0.0);
            results.push(json!({
                "id": memory.id,
                "score": score,
                "context": memory.context,
                "action": memory.action,
                "result": memory.result,
                "memory_type": memory.memory_type,
            }));
        }
        Ok::<Vec<Value>, CoreError>(results)
    })
    .await
    .map_err(|error| CoreError::SocketError(error.to_string()))?
}
