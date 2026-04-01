use std::sync::Arc;

use serde::Deserialize;
use serde_json::{Value, json};

use crate::error::CoreError;
use crate::server::ServerState;

const MAX_STALE_DAYS: u32 = 3650;

#[derive(Deserialize)]
struct PreviewParams {
    stale_days: Option<u32>,
    min_score: Option<f64>,
}

pub async fn handle_preview(
    state: &Arc<ServerState>,
    params: Value,
) -> Result<Value, CoreError> {
    let parsed: PreviewParams = serde_json::from_value(params)
        .map_err(|error| CoreError::DispatchError(error.to_string()))?;
    let stale_days = parsed.stale_days.unwrap_or(state.config.consolidation.stale_days);
    let min_score = parsed.min_score.unwrap_or(state.config.consolidation.min_score);
    validate_consolidation_params(stale_days, min_score)?;
    let state_clone = Arc::clone(state);
    let result = tokio::task::spawn_blocking(move || {
        let database = state_clone.database.lock().unwrap();
        let preview = engram_consolidate::preview(&database, stale_days, min_score)?;
        Ok::<_, CoreError>(json!({
            "duplicates": preview.duplicates.len(),
            "stale": preview.stale.len(),
            "garbage": preview.garbage.len(),
            "duplicate_groups": serialize_duplicate_groups(&preview.duplicates),
            "stale_ids": preview.stale,
            "garbage_ids": preview.garbage,
        }))
    })
    .await
    .map_err(|error| CoreError::SocketError(error.to_string()))??;
    Ok(result)
}

pub async fn handle_analyze(
    state: &Arc<ServerState>,
    params: Value,
) -> Result<Value, CoreError> {
    let parsed: PreviewParams = serde_json::from_value(params)
        .map_err(|error| CoreError::DispatchError(error.to_string()))?;
    let stale_days = parsed.stale_days.unwrap_or(state.config.consolidation.stale_days);
    let min_score = parsed.min_score.unwrap_or(state.config.consolidation.min_score);
    validate_consolidation_params(stale_days, min_score)?;
    let config = state.config.clone();
    let state_clone = Arc::clone(state);
    let result = tokio::task::spawn_blocking(move || {
        let database = state_clone.database.lock().unwrap();
        let preview = engram_consolidate::preview(&database, stale_days, min_score)?;
        let text_gen = config.build_text_generator().ok();
        let text_gen_ref = text_gen.as_deref();
        let analysis = engram_consolidate::analyze(&database, &preview, text_gen_ref)?;
        Ok::<_, CoreError>(json!({
            "analyzed_count": analysis.analyzed_count,
            "recommendations": serialize_recommendations(&analysis.recommendations),
        }))
    })
    .await
    .map_err(|error| CoreError::SocketError(error.to_string()))??;
    Ok(result)
}

#[derive(Deserialize)]
struct ApplyParams {
    stale_days: Option<u32>,
    min_score: Option<f64>,
}

pub async fn handle_apply(
    state: &Arc<ServerState>,
    params: Value,
) -> Result<Value, CoreError> {
    let parsed: ApplyParams = serde_json::from_value(params)
        .map_err(|error| CoreError::DispatchError(error.to_string()))?;
    let stale_days = parsed.stale_days.unwrap_or(state.config.consolidation.stale_days);
    let min_score = parsed.min_score.unwrap_or(state.config.consolidation.min_score);
    validate_consolidation_params(stale_days, min_score)?;
    let config = state.config.clone();
    let state_clone = Arc::clone(state);
    let result = tokio::task::spawn_blocking(move || {
        let database = state_clone.database.lock().unwrap();
        let preview = engram_consolidate::preview(&database, stale_days, min_score)?;
        let text_gen = config.build_text_generator().ok();
        let text_gen_ref = text_gen.as_deref();
        let analysis = engram_consolidate::analyze(&database, &preview, text_gen_ref)?;
        let apply_result =
            engram_consolidate::apply(&database, &analysis.recommendations, "server")?;
        Ok::<_, CoreError>(json!({
            "merged": apply_result.merged,
            "deleted": apply_result.deleted,
            "archived": apply_result.archived,
            "kept": apply_result.kept,
            "errors": apply_result.errors,
        }))
    })
    .await
    .map_err(|error| CoreError::SocketError(error.to_string()))??;
    Ok(result)
}

fn serialize_duplicate_groups(
    groups: &[engram_consolidate::DuplicateGroup],
) -> Vec<Value> {
    groups
        .iter()
        .map(|group| {
            json!({
                "primary_id": group.primary_id,
                "duplicate_ids": group.duplicate_ids,
                "similarity": group.similarity,
            })
        })
        .collect()
}

fn serialize_recommendations(
    recommendations: &[engram_consolidate::Recommendation],
) -> Vec<Value> {
    recommendations
        .iter()
        .map(|recommendation| {
            json!({
                "action": format_action(&recommendation.action),
                "confidence": recommendation.confidence,
                "reasoning": recommendation.reasoning,
            })
        })
        .collect()
}

fn format_action(action: &engram_consolidate::RecommendedAction) -> Value {
    match action {
        engram_consolidate::RecommendedAction::Merge {
            source_id,
            target_id,
        } => json!({ "type": "merge", "source_id": source_id, "target_id": target_id }),
        engram_consolidate::RecommendedAction::Delete { memory_id } => {
            json!({ "type": "delete", "memory_id": memory_id })
        }
        engram_consolidate::RecommendedAction::Archive { memory_id } => {
            json!({ "type": "archive", "memory_id": memory_id })
        }
        engram_consolidate::RecommendedAction::Keep { memory_id } => {
            json!({ "type": "keep", "memory_id": memory_id })
        }
    }
}

fn validate_consolidation_params(stale_days: u32, min_score: f64) -> Result<(), CoreError> {
    if stale_days > MAX_STALE_DAYS {
        return Err(CoreError::DispatchError(format!(
            "stale_days exceeds maximum of {MAX_STALE_DAYS}"
        )));
    }
    if !(0.0..=1.0).contains(&min_score) {
        return Err(CoreError::DispatchError(
            "min_score must be between 0.0 and 1.0".into(),
        ));
    }
    Ok(())
}
