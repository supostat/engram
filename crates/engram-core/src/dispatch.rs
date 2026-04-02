use std::sync::Arc;

use serde_json::Value;

use crate::config_handler;
use crate::consolidate_handler;
use crate::error::CoreError;
use crate::export_handler;
use crate::import_handler;
use crate::insights_handler;
use crate::judge_handler;
use crate::search_handler;
use crate::server::ServerState;
use crate::status_handler;
use crate::store_handler;
use crate::train_handler;

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
        "memory_consolidate_preview" => consolidate_handler::handle_preview(state, params).await,
        "memory_consolidate" => consolidate_handler::handle_analyze(state, params).await,
        "memory_consolidate_apply" => consolidate_handler::handle_apply(state, params).await,
        "memory_config" => config_handler::handle(state, params).await,
        "memory_export" => export_handler::handle(state, params).await,
        "memory_import" => import_handler::handle(state, params).await,
        "memory_insights" => insights_handler::handle(state, params).await,
        "memory_train_generate" => train_handler::handle_generate(state, params).await,
        "memory_train_list" => train_handler::handle_list(state, params).await,
        "memory_train_delete" => train_handler::handle_delete(state, params).await,
        unknown => Err(CoreError::DispatchError(format!(
            "unknown method: {unknown}"
        ))),
    }
}
