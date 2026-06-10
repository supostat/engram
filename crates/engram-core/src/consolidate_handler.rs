use std::sync::Arc;

use serde::Deserialize;
use serde_json::{Value, json};

use engram_storage::Database;

use crate::error::CoreError;
use crate::lock_helpers;
use crate::persistence::hash_string_to_u64;
use crate::server::ServerState;

const MAX_STALE_DAYS: u32 = 3650;
const MIN_CONFIDENCE_DEFAULT: f32 = 0.0;

#[derive(Deserialize)]
struct PreviewParams {
    stale_days: Option<u32>,
    min_score: Option<f64>,
}

pub async fn handle_preview(state: &Arc<ServerState>, params: Value) -> Result<Value, CoreError> {
    let parsed: PreviewParams = serde_json::from_value(params)
        .map_err(|error| CoreError::DispatchError(error.to_string()))?;
    let stale_days = parsed
        .stale_days
        .unwrap_or(state.config.consolidation.stale_days);
    let min_score = parsed
        .min_score
        .unwrap_or(state.config.consolidation.min_score);
    validate_consolidation_params(stale_days, min_score)?;
    let state_clone = Arc::clone(state);
    let result = tokio::task::spawn_blocking(move || {
        let database = lock_helpers::lock_db(&state_clone);
        let preview = run_preview(&state_clone, &database, stale_days, min_score)?;
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

pub async fn handle_analyze(state: &Arc<ServerState>, params: Value) -> Result<Value, CoreError> {
    let parsed: PreviewParams = serde_json::from_value(params)
        .map_err(|error| CoreError::DispatchError(error.to_string()))?;
    let stale_days = parsed
        .stale_days
        .unwrap_or(state.config.consolidation.stale_days);
    let min_score = parsed
        .min_score
        .unwrap_or(state.config.consolidation.min_score);
    validate_consolidation_params(stale_days, min_score)?;
    let state_clone = Arc::clone(state);
    let result = tokio::task::spawn_blocking(move || {
        let database = lock_helpers::lock_db(&state_clone);
        let preview = run_preview(&state_clone, &database, stale_days, min_score)?;
        let text_gen_ref = state_clone
            .text_generator
            .as_deref()
            .map(|generator| generator as &dyn engram_llm_client::TextGenerator);
        let analysis = engram_consolidate::analyze(&database, &preview, text_gen_ref)?;
        let recommendations = serialize_recommendations(&analysis.recommendations);
        Ok::<_, CoreError>(json!({
            "analyzed_count": analysis.analyzed_count,
            "recommendations": recommendations,
            "errors": analysis.errors,
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
    min_confidence: Option<f32>,
}

pub async fn handle_apply(state: &Arc<ServerState>, params: Value) -> Result<Value, CoreError> {
    let parsed: ApplyParams = serde_json::from_value(params)
        .map_err(|error| CoreError::DispatchError(error.to_string()))?;
    let stale_days = parsed
        .stale_days
        .unwrap_or(state.config.consolidation.stale_days);
    let min_score = parsed
        .min_score
        .unwrap_or(state.config.consolidation.min_score);
    let min_confidence = parsed.min_confidence.unwrap_or(MIN_CONFIDENCE_DEFAULT);
    validate_consolidation_params(stale_days, min_score)?;
    validate_min_confidence(min_confidence)?;
    let state_clone = Arc::clone(state);
    let result = tokio::task::spawn_blocking(move || {
        let (apply_result, mut errors) = {
            let database = lock_helpers::lock_db(&state_clone);
            let preview = run_preview(&state_clone, &database, stale_days, min_score)?;
            let text_gen_ref = state_clone
                .text_generator
                .as_deref()
                .map(|generator| generator as &dyn engram_llm_client::TextGenerator);
            let analysis = engram_consolidate::analyze(&database, &preview, text_gen_ref)?;
            let apply_result = engram_consolidate::apply(
                &database,
                &analysis.recommendations,
                "server",
                min_confidence,
            )?;
            (apply_result, analysis.errors)
        };
        let mut indexes = lock_helpers::write_indexes(&state_clone);
        for id in &apply_result.pruned_ids {
            let hashed = hash_string_to_u64(id);
            if indexes.contains(hashed) {
                indexes.delete(hashed).map_err(CoreError::Hnsw)?;
            }
        }
        errors.extend(apply_result.errors.iter().cloned());
        Ok::<_, CoreError>(json!({
            "merged": apply_result.merged,
            "deleted": apply_result.deleted,
            "archived": apply_result.archived,
            "kept": apply_result.kept,
            "errors": errors,
        }))
    })
    .await
    .map_err(|error| CoreError::SocketError(error.to_string()))??;
    Ok(result)
}

fn run_preview(
    state: &ServerState,
    database: &Database,
    stale_days: u32,
    min_score: f64,
) -> Result<engram_consolidate::PreviewResult, CoreError> {
    let preview = engram_consolidate::preview(
        database,
        stale_days,
        min_score,
        state.config.consolidation.fts_similarity_floor,
    )?;
    Ok(preview)
}

fn serialize_duplicate_groups(groups: &[engram_consolidate::DuplicateGroup]) -> Vec<Value> {
    groups
        .iter()
        .map(|group| {
            json!({
                "primary_id": group.primary_id,
                "duplicate_ids": group.duplicate_ids,
                "similarity": group.similarity,
                "match_type": group.match_type.as_str(),
            })
        })
        .collect()
}

fn serialize_recommendations(recommendations: &[engram_consolidate::Recommendation]) -> Vec<Value> {
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

fn validate_min_confidence(min_confidence: f32) -> Result<(), CoreError> {
    if !(0.0..=1.0).contains(&min_confidence) {
        return Err(CoreError::DispatchError(
            "min_confidence must be between 0.0 and 1.0".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn min_confidence_in_range_is_accepted() {
        assert!(validate_min_confidence(0.0).is_ok());
        assert!(validate_min_confidence(0.5).is_ok());
        assert!(validate_min_confidence(1.0).is_ok());
    }

    #[test]
    fn min_confidence_out_of_range_is_rejected() {
        assert!(matches!(
            validate_min_confidence(-0.1),
            Err(CoreError::DispatchError(_))
        ));
        assert!(matches!(
            validate_min_confidence(1.1),
            Err(CoreError::DispatchError(_))
        ));
    }

    #[test]
    fn min_confidence_nan_is_rejected() {
        assert!(matches!(
            validate_min_confidence(f32::NAN),
            Err(CoreError::DispatchError(_))
        ));
    }
}
