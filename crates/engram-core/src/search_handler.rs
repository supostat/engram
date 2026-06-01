use std::collections::HashMap;
use std::sync::Arc;

use serde::Deserialize;
use serde_json::{Value, json};

use engram_router::Mode;

use crate::config::SearchConfig;
use crate::error::CoreError;
use crate::indexes::instrumentation::ReaderTracker;
use crate::lock_helpers;
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
    let detected_mode = resolve_mode(&parsed);
    let top_k = resolve_top_k(&parsed, state, detected_mode);
    let query_embedding = embed_query(state, &parsed.query).await?;
    let vector_results = search_vector_index(state, &query_embedding, top_k).await?;
    let sparse_results = search_fts(state, &parsed.query, top_k).await?;
    let merged = merge_results(&vector_results, &sparse_results, &state.config.search);
    let limited = limit_results(merged, top_k);
    let memories = load_memories(state, &limited).await?;
    let filtered = filter_by_tags(memories, &parsed.tags);
    Ok(json!(filtered))
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
    let router = lock_helpers::lock_router(state);
    let decision = router.decide(mode, 0.5);
    decision.top_k
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

/// Fuses dense-vector and full-text hits via weighted Reciprocal Rank Fusion.
/// Both input slices arrive sorted best-first (HNSW by descending similarity,
/// FTS5 by ascending rank), so only each hit's rank position contributes —
/// the raw similarity/rank scores are intentionally ignored. A memory present
/// in both lists accumulates both weighted reciprocal-rank terms.
fn merge_results(
    vector_results: &[(String, f32)],
    sparse_results: &[(String, f64)],
    search: &SearchConfig,
) -> Vec<(String, f64)> {
    let k = search.rrf_k as f64;
    let mut combined: HashMap<String, f64> = HashMap::new();
    for (rank, (memory_id, _)) in vector_results.iter().enumerate() {
        *combined.entry(memory_id.clone()).or_insert(0.0) +=
            search.vector_weight / (k + (rank + 1) as f64);
    }
    for (rank, (memory_id, _)) in sparse_results.iter().enumerate() {
        *combined.entry(memory_id.clone()).or_insert(0.0) +=
            search.sparse_weight / (k + (rank + 1) as f64);
    }
    let mut results: Vec<(String, f64)> = combined.into_iter().collect();
    results.sort_by(|a, b| b.1.total_cmp(&a.1));
    results
}

fn limit_results(results: Vec<(String, f64)>, top_k: usize) -> Vec<(String, f64)> {
    results.into_iter().take(top_k).collect()
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
) -> Result<Vec<Value>, CoreError> {
    let memory_ids: Vec<String> = scored_results.iter().map(|(id, _)| id.clone()).collect();
    let scores: HashMap<String, f64> = scored_results.iter().cloned().collect();
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
            let _ = database.track_search(memory_id, &timestamp);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn vector_hit(id: &str) -> (String, f32) {
        (id.to_string(), 0.0)
    }

    fn sparse_hit(id: &str) -> (String, f64) {
        (id.to_string(), 0.0)
    }

    #[test]
    fn merge_ranks_by_position_not_score() {
        let vector = vec![vector_hit("a"), vector_hit("b")];
        let merged = merge_results(&vector, &[], &SearchConfig::default());
        assert_eq!(merged[0].0, "a");
        assert_eq!(merged[1].0, "b");
        assert!(merged[0].1 > merged[1].1);
    }

    #[test]
    fn merge_sums_contributions_for_shared_id() {
        let search = SearchConfig::default();
        let vector = vec![vector_hit("a"), vector_hit("b")];
        let sparse = vec![sparse_hit("a"), sparse_hit("c")];
        let merged = merge_results(&vector, &sparse, &search);

        let k = search.rrf_k as f64;
        let score_a = merged.iter().find(|(id, _)| id == "a").unwrap().1;
        let expected_a = search.vector_weight / (k + 1.0) + search.sparse_weight / (k + 1.0);
        assert!((score_a - expected_a).abs() < 1e-12);

        let score_b = merged.iter().find(|(id, _)| id == "b").unwrap().1;
        assert!(score_a > score_b);
    }

    // Asymmetric weights must drive the final ranking, not just rank position.
    // Construction: X is rank-1 in the dense-vector list and absent from sparse;
    // Y is rank-1 in sparse and far down the vector list (rank-9). With heavy
    // vector weighting (0.9 / 0.1) the dense source dominates and X leads; the
    // symmetric default (0.7 / 0.3) lets Y's strong sparse hit win instead, so
    // the two configs produce opposite orderings.
    //
    // Exact RRF (k = 60), heavy-vector 0.9 / 0.1:
    //   X = 0.9 / (60 + 1)              = 0.014754098360655738
    //   Y = 0.9 / (60 + 9) + 0.1/(60+1) = 0.014682822523164649  → X > Y
    // Symmetric default 0.7 / 0.3:
    //   X = 0.7 / (60 + 1)              = 0.011475409836065573
    //   Y = 0.7 / (60 + 9) + 0.3/(60+1) = 0.015062960323117129  → Y > X
    //
    // Y sits at vector rank-9 because, sharing both lists, it accrues the heavy
    // vector term too; only past rank-8 does X's single rank-1 vector hit
    // overtake Y's rank-9 vector hit plus its rank-1 sparse hit at 0.9 / 0.1.
    #[test]
    fn asymmetric_weights_drive_final_ranking() {
        let vector = vec![
            vector_hit("x"),
            vector_hit("v2"),
            vector_hit("v3"),
            vector_hit("v4"),
            vector_hit("v5"),
            vector_hit("v6"),
            vector_hit("v7"),
            vector_hit("v8"),
            vector_hit("y"),
        ];
        let sparse = vec![sparse_hit("y")];

        let heavy_vector = SearchConfig {
            rrf_k: 60,
            vector_weight: 0.9,
            sparse_weight: 0.1,
        };
        let merged = merge_results(&vector, &sparse, &heavy_vector);

        let score_x = merged.iter().find(|(id, _)| id == "x").unwrap().1;
        let score_y = merged.iter().find(|(id, _)| id == "y").unwrap().1;
        let expected_x = 0.9 / 61.0;
        let expected_y = 0.9 / 69.0 + 0.1 / 61.0;
        assert!((score_x - expected_x).abs() < 1e-12);
        assert!((score_y - expected_y).abs() < 1e-12);
        assert!(
            score_x > score_y,
            "heavy vector weighting must let the vector rank-1 id win: x={score_x}, y={score_y}"
        );
        let x_position = merged.iter().position(|(id, _)| id == "x").unwrap();
        let y_position = merged.iter().position(|(id, _)| id == "y").unwrap();
        assert!(x_position < y_position, "x must outrank y under 0.9 / 0.1");

        let symmetric = SearchConfig::default();
        let merged_symmetric = merge_results(&vector, &sparse, &symmetric);
        let symmetric_x = merged_symmetric.iter().find(|(id, _)| id == "x").unwrap().1;
        let symmetric_y = merged_symmetric.iter().find(|(id, _)| id == "y").unwrap().1;
        assert!(
            symmetric_y > symmetric_x,
            "the symmetric default must produce the opposite order: x={symmetric_x}, y={symmetric_y}"
        );
    }

    #[test]
    fn merge_empty_inputs_yield_empty() {
        let merged = merge_results(&[], &[], &SearchConfig::default());
        assert!(merged.is_empty());
    }

    #[test]
    fn merge_scores_sparse_only_doc_absent_from_vector() {
        let search = SearchConfig::default();
        let sparse = vec![sparse_hit("only_sparse")];
        let merged = merge_results(&[], &sparse, &search);

        let k = search.rrf_k as f64;
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].0, "only_sparse");
        let expected = search.sparse_weight / (k + 1.0);
        assert!((merged[0].1 - expected).abs() < 1e-12);
    }
}
