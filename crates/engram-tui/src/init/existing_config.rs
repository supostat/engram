use std::fs;
use std::path::{Path, PathBuf};

use toml::Value;

pub struct ExistingConfig {
    pub embedding_provider: String,
    pub embedding_model: String,
    pub embedding_api_key: Option<String>,
    pub llm_provider: String,
    pub llm_model: String,
    pub llm_api_key: Option<String>,
    pub database_path: String,
    pub socket_path: String,
}

pub struct EngineStats {
    pub memory_count: usize,
    pub indexed_count: usize,
    pub average_score: f64,
    pub model_count: usize,
    pub models_size_bytes: u64,
}

pub struct HealthStatus {
    pub embedding_key_present: bool,
    pub llm_key_present: bool,
    pub database_found: bool,
    pub database_memory_count: usize,
    pub database_size_bytes: u64,
    pub hnsw_found: bool,
    pub hnsw_size_bytes: u64,
    pub socket_exists: bool,
    pub model_count: usize,
    pub models_size_bytes: u64,
}

impl ExistingConfig {
    pub fn load() -> Option<Self> {
        let config_path = engram_config_path()?;
        let content = fs::read_to_string(&config_path).ok()?;
        let table: Value = content.parse().ok()?;

        let embedding_provider = table_string(&table, &["embedding", "provider"])?;
        let embedding_model =
            table_string(&table, &["embedding", "model"]).unwrap_or_else(|| "unknown".into());
        let llm_provider = table_string(&table, &["llm", "provider"])?;
        let llm_model = table_string(&table, &["llm", "model"]).unwrap_or_else(|| "unknown".into());
        let database_path = table_string(&table, &["database", "path"])
            .unwrap_or_else(|| "~/.engram/memories.db".into());
        let socket_path = table_string(&table, &["server", "socket_path"])
            .unwrap_or_else(|| "~/.engram/engram.sock".into());

        let embedding_api_key = table_string(&table, &["embedding", "api_key"])
            .or_else(|| std::env::var("ENGRAM_VOYAGE_API_KEY").ok());
        let llm_api_key = table_string(&table, &["llm", "api_key"])
            .or_else(|| std::env::var("ENGRAM_OPENAI_API_KEY").ok());

        Some(Self {
            embedding_provider,
            embedding_model,
            embedding_api_key,
            llm_provider,
            llm_model,
            llm_api_key,
            database_path,
            socket_path,
        })
    }

    pub fn collect_stats(&self) -> EngineStats {
        let expanded_database = expand_tilde(&self.database_path);
        let (memory_count, indexed_count, average_score) = read_database_stats(&expanded_database);
        let (model_count, models_size_bytes) = read_models_stats();
        EngineStats {
            memory_count,
            indexed_count,
            average_score,
            model_count,
            models_size_bytes,
        }
    }

    pub fn run_health_check(&self) -> HealthStatus {
        let expanded_database = expand_tilde(&self.database_path);
        let expanded_socket = expand_tilde(&self.socket_path);

        let (database_found, database_memory_count, database_size_bytes) =
            check_database(&expanded_database);
        let (hnsw_found, hnsw_size_bytes) = check_hnsw();
        let (model_count, models_size_bytes) = read_models_stats();

        HealthStatus {
            embedding_key_present: self
                .embedding_api_key
                .as_ref()
                .is_some_and(|k| !k.is_empty()),
            llm_key_present: self.llm_api_key.as_ref().is_some_and(|k| !k.is_empty()),
            database_found,
            database_memory_count,
            database_size_bytes,
            hnsw_found,
            hnsw_size_bytes,
            socket_exists: Path::new(&expanded_socket).exists(),
            model_count,
            models_size_bytes,
        }
    }
}

pub fn mask_api_key(key: &str) -> String {
    if key.len() > 7 {
        let prefix: String = key.chars().take(3).collect();
        let suffix: String = key
            .chars()
            .rev()
            .take(4)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        format!("{prefix}****{suffix}")
    } else {
        "****".into()
    }
}

pub fn format_size(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.1} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{} MB", bytes / 1_048_576)
    } else if bytes >= 1024 {
        format!("{} KB", bytes / 1024)
    } else {
        format!("{bytes} B")
    }
}

fn engram_config_path() -> Option<PathBuf> {
    let home = dirs::home_dir()?;
    let path = home.join(".engram/engram.toml");
    if path.exists() { Some(path) } else { None }
}

fn table_string(table: &Value, keys: &[&str]) -> Option<String> {
    let mut current = table;
    for key in keys {
        current = current.get(key)?;
    }
    current.as_str().map(|s| s.to_string())
}

fn expand_tilde(path: &str) -> String {
    if !path.starts_with('~') {
        return path.to_string();
    }
    let Some(home) = dirs::home_dir() else {
        return path.to_string();
    };
    home.join(&path[2..]).to_string_lossy().into_owned()
}

fn read_database_stats(database_path: &str) -> (usize, usize, f64) {
    let path = Path::new(database_path);
    if !path.exists() {
        return (0, 0, 0.0);
    }
    let Ok(connection) = rusqlite::Connection::open_with_flags(
        database_path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    ) else {
        return (0, 0, 0.0);
    };
    let memory_count = connection
        .query_row("SELECT COUNT(*) FROM memories", [], |row| {
            row.get::<_, i64>(0)
        })
        .unwrap_or(0) as usize;
    let indexed_count = connection
        .query_row(
            "SELECT COUNT(*) FROM memories WHERE indexed = TRUE",
            [],
            |row| row.get::<_, i64>(0),
        )
        .unwrap_or(0) as usize;
    let average_score = connection
        .query_row(
            "SELECT COALESCE(AVG(score), 0.0) FROM memories",
            [],
            |row| row.get::<_, f64>(0),
        )
        .unwrap_or(0.0);
    (memory_count, indexed_count, average_score)
}

fn check_database(database_path: &str) -> (bool, usize, u64) {
    let path = Path::new(database_path);
    if !path.exists() {
        return (false, 0, 0);
    }
    let size = fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let Ok(connection) = rusqlite::Connection::open_with_flags(
        database_path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    ) else {
        return (true, 0, size);
    };
    let count = connection
        .query_row("SELECT COUNT(*) FROM memories", [], |row| {
            row.get::<_, i64>(0)
        })
        .unwrap_or(0) as usize;
    (true, count, size)
}

fn check_hnsw() -> (bool, u64) {
    let Some(home) = dirs::home_dir() else {
        return (false, 0);
    };
    let hnsw_path = home.join(".engram/indexes.hnsw");
    if !hnsw_path.exists() {
        return (false, 0);
    }
    let size = fs::metadata(&hnsw_path).map(|m| m.len()).unwrap_or(0);
    (true, size)
}

fn read_models_stats() -> (usize, u64) {
    let Some(home) = dirs::home_dir() else {
        return (0, 0);
    };
    let models_dir = home.join(".engram/models");
    let Ok(entries) = fs::read_dir(&models_dir) else {
        return (0, 0);
    };
    let mut count = 0usize;
    let mut total_size = 0u64;
    for entry in entries.flatten() {
        let is_model = entry
            .path()
            .extension()
            .is_some_and(|ext| ext == "onnx" || ext == "json" || ext == "data" || ext == "txt");
        if is_model {
            count += 1;
            total_size += entry.metadata().map(|m| m.len()).unwrap_or(0);
        }
    }
    (count, total_size)
}
