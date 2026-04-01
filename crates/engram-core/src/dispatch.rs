use std::sync::Arc;

use serde_json::Value;

use crate::consolidate_handler;
use crate::error::CoreError;
use crate::judge_handler;
use crate::search_handler;
use crate::server::ServerState;
use crate::status_handler;
use crate::store_handler;

pub async fn route(
    method: &str,
    state: &Arc<ServerState>,
    params: Value,
) -> Result<Value, CoreError> {
    match method {
        "memory_store" => store_handler::handle(state, params).await,
        "memory_search" => search_handler::handle(state, params).await,
        "memory_judge" => judge_handler::handle(state, params).await,
        "memory_status" => status_handler::handle(state, params).await,
        "memory_consolidate_preview" => {
            consolidate_handler::handle_preview(state, params).await
        }
        "memory_consolidate" => {
            consolidate_handler::handle_analyze(state, params).await
        }
        "memory_consolidate_apply" => {
            consolidate_handler::handle_apply(state, params).await
        }
        unknown => Err(CoreError::DispatchError(format!(
            "unknown method: {unknown}"
        ))),
    }
}
