use std::sync::Arc;

use serde::Deserialize;
use serde_json::{Value, json};

use crate::error::CoreError;
use crate::server::ServerState;

#[derive(Deserialize)]
struct ConfigParams {
    action: String,
}

pub async fn handle(
    state: &Arc<ServerState>,
    params: Value,
) -> Result<Value, CoreError> {
    let parsed: ConfigParams = serde_json::from_value(params)
        .map_err(|error| CoreError::DispatchError(error.to_string()))?;
    match parsed.action.as_str() {
        "get" => Ok(sanitized_config(&state.config)),
        "set" => Err(CoreError::ConfigReadOnly),
        other => Err(CoreError::DispatchError(format!(
            "invalid config action: {other}"
        ))),
    }
}

fn sanitized_config(config: &crate::config::Config) -> Value {
    json!({
        "database": { "path": config.database.path },
        "embedding": {
            "provider": config.embedding.provider,
            "model": config.embedding.model,
            "dimension": config.embedding.dimension,
        },
        "llm": {
            "provider": config.llm.provider,
            "model": config.llm.model,
        },
        "server": {
            "socket_path": config.server.socket_path,
            "reindex_interval_secs": config.server.reindex_interval_secs,
        },
        "hnsw": {
            "max_connections": config.hnsw.max_connections,
            "ef_construction": config.hnsw.ef_construction,
            "ef_search": config.hnsw.ef_search,
            "dimension": config.hnsw.dimension,
        },
        "consolidation": {
            "stale_days": config.consolidation.stale_days,
            "min_score": config.consolidation.min_score,
        },
    })
}
