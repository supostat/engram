use std::collections::HashMap;
use std::sync::Arc;

use serde::Deserialize;
use serde_json::{Value, json};

use engram_llm_client::ApiError;
use engram_router::{Mode, RouterDecision};
use engram_storage::routing_log::RoutingLogEntry;

use crate::error::CoreError;
use crate::indexes::instrumentation::ReaderTracker;
use crate::lock_helpers;
use crate::rank_fusion::{SHADOW_K_SET, limit_results, merge_results, shadow_rewards_for_k_set};
use crate::server::ServerState;
use crate::timestamp::current_utc_timestamp;

const MAX_QUERY_LENGTH: usize = 5_000;

#[derive(Deserialize)]
#[allow(dead_code)]
struct SearchParams {
    query: String,
    limit: Option<usize>,
    mode: Option<String>,
    project: Option<String>,
    tags: Option<Vec<String>>,
}

pub async fn handle(state: &Arc<ServerState>, params: Value) -> Result<Value, CoreError> {
    let parsed: SearchParams = serde_json::from_value(params)
        .map_err(|error| CoreError::DispatchError(error.to_string()))?;
    if parsed.query.len() > MAX_QUERY_LENGTH {
        return Err(CoreError::DispatchError(format!(
            "query exceeds maximum length of {MAX_QUERY_LENGTH} bytes"
        )));
    }
    let query_id = uuid::Uuid::new_v4().to_string();
    let detected_mode = resolve_mode(&parsed);
    let decision = resolve_decision(&parsed, state, detected_mode);
    let top_k = decision.top_k;
    match embed_query(state, &parsed.query).await {
        Ok(query_embedding) => {
            let vector_results = search_vector_index(state, &query_embedding, top_k).await?;
            let filtered =
                rank_and_load(state, &vector_results, &parsed, &decision, &query_id).await?;
            Ok(search_response(filtered, false))
        }
        Err(CoreError::Api(ApiError::EmbeddingApiUnavailable(_))) => {
            let filtered = rank_and_load(state, &[], &parsed, &decision, &query_id).await?;
            Ok(search_response(filtered, true))
        }
        Err(other) => Err(other),
    }
}

async fn rank_and_load(
    state: &Arc<ServerState>,
    vector_results: &[(String, f32)],
    params: &SearchParams,
    decision: &RouterDecision,
    query_id: &str,
) -> Result<Vec<Value>, CoreError> {
    let top_k = decision.top_k;
    let sparse_results = search_fts(state, &params.query, top_k).await?;
    let merged = merge_results(vector_results, &sparse_results, &state.config.search);
    log_routing(state, decision, query_id, &merged);
    let limited = limit_results(merged, top_k);
    let memories = load_memories(state, &limited, query_id).await?;
    Ok(filter_by_tags(memories, &params.tags))
}

/// Side-effect-only instrumentation: records the served router decision plus the
/// counterfactual shadow rewards for offline analysis. Best-effort — a logging
/// failure never affects the served `{results, degraded}` response.
fn log_routing(
    state: &Arc<ServerState>,
    decision: &RouterDecision,
    query_id: &str,
    merged: &[(String, f64)],
) {
    let shadow_rewards = shadow_rewards_for_k_set(merged, &SHADOW_K_SET);
    let shadow_rewards_json = serde_json::to_string(
        &shadow_rewards
            .iter()
            .map(|(k, reward)| json!({ "k": k, "reward": reward }))
            .collect::<Vec<Value>>(),
    )
    .ok();
    let created_at = current_utc_timestamp();
    let entry = RoutingLogEntry {
        query_id,
        mode: decision.mode.as_str(),
        search_strategy: decision.search_strategy.as_str(),
        llm_selection: decision.llm_selection.as_str(),
        contextualization: decision.contextualization.as_str(),
        proactivity: decision.proactivity.as_str(),
        top_k: decision.top_k,
        shadow_rewards_json: shadow_rewards_json.as_deref(),
        created_at: &created_at,
    };
    let _ = {
        let database = lock_helpers::lock_db(state);
        database.log_routing_decision(&entry)
    };
}

fn search_response(results: Vec<Value>, degraded: bool) -> Value {
    json!({ "results": results, "degraded": degraded })
}

fn resolve_mode(params: &SearchParams) -> Mode {
    match &params.mode {
        Some(mode_string) => {
            Mode::parse(mode_string).unwrap_or_else(|_| Mode::detect(&params.query))
        }
        None => Mode::detect(&params.query),
    }
}

/// Resolves the full router decision for this search. The router lock is dropped
/// before returning (the owned `RouterDecision` is moved out). An explicit
/// `params.limit` overrides only the served `top_k`; every other level is taken
/// straight from `router.decide` so serving stays byte-identical to the prior
/// `resolve_top_k` behaviour.
fn resolve_decision(params: &SearchParams, state: &Arc<ServerState>, mode: Mode) -> RouterDecision {
    let mut decision = {
        let router = lock_helpers::lock_router(state);
        router.decide(mode, 0.5)
    };
    if let Some(limit) = params.limit {
        decision.top_k = limit;
    }
    decision
}

async fn embed_query(state: &Arc<ServerState>, query: &str) -> Result<Vec<f32>, CoreError> {
    let query_owned = query.to_string();
    let state_clone = Arc::clone(state);
    tokio::task::spawn_blocking(move || {
        let provider = state_clone.embedding_provider.as_ref();
        let text_gen_ref = state_clone
            .text_generator
            .as_deref()
            .map(|generator| generator as &dyn engram_llm_client::TextGenerator);
        state_clone
            .embedder
            .embed_query(&query_owned, provider, text_gen_ref)
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
) -> Result<Vec<(String, f32)>, CoreError> {
    let embedding_owned = query_embedding.to_vec();
    let state_clone = Arc::clone(state);
    tokio::task::spawn_blocking(move || {
        let _tracker = ReaderTracker::new();
        let indexes = lock_helpers::read_indexes(&state_clone);
        let raw_results = indexes
            .search(&embedding_owned, top_k)
            .map_err(CoreError::Hnsw)?;
        let resolved: Vec<(String, f32)> = raw_results
            .into_iter()
            .filter_map(|(node_id, score)| {
                indexes
                    .resolve_node_id(node_id)
                    .map(|memory_id| (memory_id.to_string(), score))
            })
            .collect();
        Ok(resolved)
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
        let database = lock_helpers::lock_db(&state_clone);
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

fn filter_by_tags(memories: Vec<Value>, required_tags: &Option<Vec<String>>) -> Vec<Value> {
    let required = match required_tags {
        Some(tags) if !tags.is_empty() => tags,
        _ => return memories,
    };
    let required_lower: Vec<String> = required.iter().map(|tag| tag.to_lowercase()).collect();
    memories
        .into_iter()
        .filter(|memory| memory_has_all_tags(memory, &required_lower))
        .collect()
}

fn memory_has_all_tags(memory: &Value, required_lower: &[String]) -> bool {
    let raw = match memory.get("tags").and_then(Value::as_str) {
        Some(value) if !value.is_empty() => value,
        _ => return false,
    };
    let stored_tags: Vec<String> = match serde_json::from_str(raw) {
        Ok(parsed) => parsed,
        Err(_) => return false,
    };
    let stored_lower: Vec<String> = stored_tags.iter().map(|tag| tag.to_lowercase()).collect();
    required_lower
        .iter()
        .all(|required_tag| stored_lower.contains(required_tag))
}

async fn load_memories(
    state: &Arc<ServerState>,
    scored_results: &[(String, f64)],
    query_id: &str,
) -> Result<Vec<Value>, CoreError> {
    let memory_ids: Vec<String> = scored_results.iter().map(|(id, _)| id.clone()).collect();
    let scores: HashMap<String, f64> = scored_results.iter().cloned().collect();
    let query_id = query_id.to_string();
    let state_clone = Arc::clone(state);
    tokio::task::spawn_blocking(move || {
        let database = lock_helpers::lock_db(&state_clone);
        let timestamp = current_utc_timestamp();
        let mut results = Vec::new();
        for memory_id in &memory_ids {
            let memory = match database.get_memory(memory_id) {
                Ok(m) => m,
                Err(_) => continue,
            };
            if memory.superseded_by.is_some() {
                continue;
            }
            let _ = database.track_search(memory_id, &query_id, &timestamp);
            let score = scores.get(memory_id.as_str()).copied().unwrap_or(0.0);
            results.push(json!({
                "id": memory.id,
                "score": score,
                "context": memory.context,
                "action": memory.action,
                "result": memory.result,
                "memory_type": memory.memory_type,
                "tags": memory.tags,
            }));
        }
        Ok::<Vec<Value>, CoreError>(results)
    })
    .await
    .map_err(|error| CoreError::SocketError(error.to_string()))?
}
