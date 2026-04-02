use std::sync::Arc;

use serde::Deserialize;
use serde_json::{Value, json};

use engram_judge::{CombinedJudge, JudgeInput};

use crate::error::CoreError;
use crate::server::ServerState;
use crate::timestamp::{current_utc_timestamp, parse_timestamp_to_epoch};

const MAX_MEMORY_ID_LENGTH: usize = 100;

#[derive(Deserialize)]
struct JudgeParams {
    memory_id: String,
    query: Option<String>,
    score: Option<f32>,
}

pub async fn handle(state: &Arc<ServerState>, params: Value) -> Result<Value, CoreError> {
    let parsed: JudgeParams = serde_json::from_value(params)
        .map_err(|error| CoreError::DispatchError(error.to_string()))?;
    if parsed.memory_id.is_empty() {
        return Err(CoreError::DispatchError(
            "memory_id must not be empty".into(),
        ));
    }
    if parsed.memory_id.len() > MAX_MEMORY_ID_LENGTH {
        return Err(CoreError::DispatchError(format!(
            "memory_id exceeds maximum length of {MAX_MEMORY_ID_LENGTH} chars"
        )));
    }
    if let Some(explicit_score) = parsed.score {
        return apply_explicit_score(state, &parsed.memory_id, explicit_score).await;
    }
    let query = parsed.query.unwrap_or_default();
    compute_and_apply_score(state, &parsed.memory_id, &query).await
}

async fn apply_explicit_score(
    state: &Arc<ServerState>,
    memory_id: &str,
    score: f32,
) -> Result<Value, CoreError> {
    let id_owned = memory_id.to_string();
    let state_clone = Arc::clone(state);
    tokio::task::spawn_blocking(move || {
        let database = state_clone.database.lock().unwrap();
        database.set_memory_score(&id_owned, score)?;
        let timestamp = current_utc_timestamp();
        database.mark_judged(&id_owned, &timestamp)?;
        Ok::<(), CoreError>(())
    })
    .await
    .map_err(|error| CoreError::SocketError(error.to_string()))??;
    Ok(json!({
        "score": score,
        "reason": "explicit score",
        "degraded": false,
    }))
}

async fn compute_and_apply_score(
    state: &Arc<ServerState>,
    memory_id: &str,
    query: &str,
) -> Result<Value, CoreError> {
    let id_owned = memory_id.to_string();
    let query_owned = query.to_string();
    let config = state.config.clone();
    let state_clone = Arc::clone(state);
    let result = tokio::task::spawn_blocking(move || {
        let database = state_clone.database.lock().unwrap();
        let memory = database.get_memory(&id_owned)?;
        let judge_input = build_judge_input(&memory);
        let text_gen = config.build_text_generator().ok();
        let judge = match text_gen.as_deref() {
            Some(generator) => CombinedJudge::with_llm(generator),
            None => CombinedJudge::heuristic_only(),
        };
        let judge_score = judge.score(&query_owned, &judge_input);
        database.set_memory_score(&id_owned, judge_score.score)?;
        let timestamp = current_utc_timestamp();
        database.mark_judged(&id_owned, &timestamp)?;
        Ok::<_, CoreError>(json!({
            "score": judge_score.score,
            "reason": judge_score.reason,
            "degraded": judge_score.degraded,
        }))
    })
    .await
    .map_err(|error| CoreError::SocketError(error.to_string()))??;
    Ok(result)
}

fn build_judge_input(memory: &engram_storage::Memory) -> JudgeInput {
    let days_since_update = compute_days_since_update(&memory.updated_at);
    JudgeInput {
        context: memory.context.clone(),
        action: memory.action.clone(),
        result: memory.result.clone(),
        days_since_update,
        used_count: memory.used_count.max(0) as u64,
    }
}

fn compute_days_since_update(updated_at: &str) -> f64 {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let parsed = parse_timestamp_to_epoch(updated_at).unwrap_or(now);
    let elapsed_seconds = now.saturating_sub(parsed);
    elapsed_seconds as f64 / 86400.0
}
