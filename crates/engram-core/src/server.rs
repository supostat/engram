use std::sync::{Arc, Mutex};

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixListener;
use tokio::signal::unix::{signal, SignalKind};

use engram_embeddings::Embedder;
use engram_router::Router;
use engram_storage::Database;

use crate::config::Config;
use crate::dispatch;
use crate::error::CoreError;
use crate::indexes::IndexSet;
use crate::protocol::{JsonRequest, JsonResponse};

const ROUTER_DEFAULT_ALPHA: f32 = 0.1;
const ROUTER_DEFAULT_EPSILON: f32 = 0.15;

pub struct ServerState {
    pub database: Mutex<Database>,
    pub indexes: Mutex<IndexSet>,
    pub embedder: Mutex<Embedder>,
    pub router: Mutex<Router>,
    pub config: Config,
}

pub async fn run(config: Config) -> Result<(), CoreError> {
    let state = initialize_state(&config)?;
    let shared_state = Arc::new(state);
    let socket_path = expand_tilde(&config.server.socket_path);
    cleanup_stale_socket(&socket_path);
    let listener = bind_listener(&socket_path)?;
    accept_loop(listener, shared_state).await
}

pub(crate) fn initialize_state(config: &Config) -> Result<ServerState, CoreError> {
    let database_path = config.resolve_database_path();
    let database = Database::open(&database_path)?;
    let hnsw_config = config.clone();
    let indexes = crate::persistence::load_or_rebuild(
        &resolve_index_directory(&database_path),
        &database,
        || hnsw_config.build_hnsw_params(),
    )?;
    let embedder = Embedder::new();
    let router = Router::new(ROUTER_DEFAULT_ALPHA, ROUTER_DEFAULT_EPSILON);
    Ok(ServerState {
        database: Mutex::new(database),
        indexes: Mutex::new(indexes),
        embedder: Mutex::new(embedder),
        router: Mutex::new(router),
        config: config.clone(),
    })
}

pub(crate) fn resolve_index_directory(database_path: &str) -> String {
    std::path::Path::new(database_path)
        .parent()
        .map(|parent| parent.to_string_lossy().to_string())
        .unwrap_or_else(|| ".".to_string())
}

fn cleanup_stale_socket(socket_path: &str) {
    let _ = std::fs::remove_file(socket_path);
}

fn bind_listener(socket_path: &str) -> Result<UnixListener, CoreError> {
    if let Some(parent) = std::path::Path::new(socket_path).parent() {
        std::fs::create_dir_all(parent)
            .map_err(|error| CoreError::SocketError(error.to_string()))?;
    }
    UnixListener::bind(socket_path)
        .map_err(|error| CoreError::SocketError(error.to_string()))
}

async fn accept_loop(
    listener: UnixListener,
    state: Arc<ServerState>,
) -> Result<(), CoreError> {
    let mut sigterm =
        signal(SignalKind::terminate()).map_err(|e| CoreError::SocketError(e.to_string()))?;
    spawn_background_reindex(Arc::clone(&state));
    loop {
        tokio::select! {
            accept_result = listener.accept() => {
                let (stream, _address) = accept_result
                    .map_err(|error| CoreError::SocketError(error.to_string()))?;
                let client_state = Arc::clone(&state);
                tokio::spawn(async move {
                    if let Err(error) = handle_client(stream, client_state).await {
                        eprintln!("client error: {error}");
                    }
                });
            }
            _ = sigterm.recv() => {
                return save_indexes_on_shutdown(&state);
            }
        }
    }
}

fn spawn_background_reindex(state: Arc<ServerState>) {
    let interval_secs = state.config.server.reindex_interval_secs;
    if interval_secs == 0 {
        return;
    }
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(
            std::time::Duration::from_secs(interval_secs),
        );
        interval.tick().await; // skip first immediate tick
        loop {
            interval.tick().await;
            let reindex_state = Arc::clone(&state);
            let _ = tokio::task::spawn_blocking(move || {
                reindex_unindexed_memories(&reindex_state);
            })
            .await;
        }
    });
}

fn reindex_unindexed_memories(state: &ServerState) {
    let database = state.database.lock().unwrap();
    let unindexed = match query_unindexed_ids(&database) {
        Ok(ids) => ids,
        Err(_) => return,
    };
    if unindexed.is_empty() {
        return;
    }
    let mut indexes = state.indexes.lock().unwrap();
    for memory_id in &unindexed {
        let _ = reindex_single_memory(&database, &mut indexes, memory_id);
    }
}

fn query_unindexed_ids(database: &Database) -> Result<Vec<String>, CoreError> {
    let mut statement = database
        .connection()
        .prepare(
            "SELECT id FROM memories WHERE indexed = 0 \
             AND embedding_context IS NOT NULL LIMIT 100",
        )
        .map_err(|e| CoreError::Storage(engram_storage::StorageError::from(e)))?;
    let rows = statement
        .query_map([], |row| row.get::<_, String>(0))
        .map_err(|e| CoreError::Storage(engram_storage::StorageError::from(e)))?;
    let mut ids = Vec::new();
    for row in rows {
        if let Ok(id) = row {
            ids.push(id);
        }
    }
    Ok(ids)
}

fn reindex_single_memory(
    database: &Database,
    indexes: &mut IndexSet,
    memory_id: &str,
) -> Result<(), CoreError> {
    let memory = database.get_memory(memory_id)?;
    let embedding = crate::persistence::extract_embeddings_from_memory(&memory)?;
    let id_hash = crate::persistence::hash_string_to_u64(memory_id);
    let rng_value = crate::persistence::deterministic_rng(id_hash);
    if !indexes.contains(id_hash) {
        indexes.insert(id_hash, &embedding, rng_value)?;
    }
    database.set_memory_indexed(memory_id, true)?;
    Ok(())
}

async fn handle_client(
    stream: tokio::net::UnixStream,
    state: Arc<ServerState>,
) -> Result<(), CoreError> {
    let (reader, mut writer) = stream.into_split();
    let mut lines = BufReader::new(reader).lines();
    while let Some(line) = lines
        .next_line()
        .await
        .map_err(|error| CoreError::SocketError(error.to_string()))?
    {
        let response = process_request_line(&line, &state).await;
        let serialized = serde_json::to_string(&response)
            .map_err(|error| CoreError::SocketError(error.to_string()))?;
        writer
            .write_all(serialized.as_bytes())
            .await
            .map_err(|error| CoreError::SocketError(error.to_string()))?;
        writer
            .write_all(b"\n")
            .await
            .map_err(|error| CoreError::SocketError(error.to_string()))?;
    }
    Ok(())
}

async fn process_request_line(
    line: &str,
    state: &Arc<ServerState>,
) -> JsonResponse {
    let request: JsonRequest = match serde_json::from_str(line) {
        Ok(parsed) => parsed,
        Err(error) => {
            return JsonResponse::error(
                String::new(),
                4000,
                format!("invalid json: {error}"),
            );
        }
    };
    let request_id = request.id.clone();
    match dispatch::route(&request.method, state, request.params).await {
        Ok(data) => JsonResponse::success(request_id, data),
        Err(error) => JsonResponse::error(request_id, error_code(&error), error.to_string()),
    }
}

fn error_code(error: &CoreError) -> u32 {
    match error {
        CoreError::ConfigNotFound => 6001,
        CoreError::ConfigParseError(_) => 6002,
        CoreError::InvalidProvider(_) => 6003,
        CoreError::IndexCorrupted(_) => 6004,
        CoreError::RebuildFailed(_) => 6005,
        CoreError::SocketError(_) => 6006,
        CoreError::DispatchError(_) => 6007,
        CoreError::ConfigReadOnly => 6008,
        CoreError::ExportFailed(_) => 6009,
        CoreError::ImportVersionMismatch(_) => 6010,
        CoreError::ImportFailed(_) => 6011,
        CoreError::InitFailed(_) => 6012,
        CoreError::TrainerFailed(_) => 6013,
        CoreError::TrainerTimeout => 6014,
        CoreError::TrainerMalformedOutput(_) => 6015,
        CoreError::Storage(_) => 1000,
        CoreError::Hnsw(_) => 3000,
        CoreError::Api(_) => 2000,
        CoreError::Consolidation(_) => 5000,
    }
}

fn save_indexes_on_shutdown(state: &Arc<ServerState>) -> Result<(), CoreError> {
    let database_path = state.config.resolve_database_path();
    let index_directory = resolve_index_directory(&database_path);
    let indexes = state.indexes.lock().unwrap();
    crate::persistence::save_to_disk(&index_directory, &indexes)
}

fn expand_tilde(path: &str) -> String {
    if let Some(rest) = path.strip_prefix("~/")
        && let Ok(home) = std::env::var("HOME") {
            return format!("{home}/{rest}");
        }
    path.to_string()
}
