use std::fs;
use std::path::Path;

use serde::Deserialize;

use engram_hnsw::HnswParams;
use engram_llm_client::{
    EmbeddingProvider, OpenAITextGenerator, TextGenerator, VoyageEmbeddingProvider,
};

use crate::error::CoreError;

const CONFIG_LOCAL_PATH: &str = "engram.toml";
const CONFIG_HOME_SUBDIR: &str = ".engram/engram.toml";

const DEFAULT_DB_PATH: &str = "~/.engram/memories.db";
const DEFAULT_SOCKET_PATH: &str = "~/.engram/engram.sock";
const DEFAULT_EMBEDDING_PROVIDER: &str = "voyage";
const DEFAULT_EMBEDDING_MODEL: &str = "voyage-code-3";
const DEFAULT_EMBEDDING_DIMENSION: usize = 1024;
const DEFAULT_LLM_PROVIDER: &str = "openai";
const DEFAULT_LLM_MODEL: &str = "gpt-4o-mini";
const DEFAULT_REINDEX_INTERVAL_SECS: u64 = 3600;
const DEFAULT_HNSW_MAX_CONNECTIONS: usize = 16;
const DEFAULT_HNSW_EF_CONSTRUCTION: usize = 200;
const DEFAULT_HNSW_EF_SEARCH: usize = 40;
const DEFAULT_CONSOLIDATION_STALE_DAYS: u32 = 90;
const DEFAULT_CONSOLIDATION_MIN_SCORE: f64 = 0.3;

#[derive(Deserialize, Clone)]
pub struct Config {
    pub database: DatabaseConfig,
    pub embedding: EmbeddingConfig,
    pub llm: LlmConfig,
    pub server: ServerConfig,
    pub hnsw: HnswConfig,
    #[serde(default)]
    pub consolidation: ConsolidationConfig,
}

#[derive(Deserialize, Clone)]
pub struct DatabaseConfig {
    pub path: String,
}

#[derive(Deserialize, Clone)]
pub struct EmbeddingConfig {
    pub provider: String,
    pub api_key: Option<String>,
    pub model: Option<String>,
    pub dimension: Option<usize>,
}

#[derive(Deserialize, Clone)]
pub struct LlmConfig {
    pub provider: String,
    pub api_key: Option<String>,
    pub model: Option<String>,
}

#[derive(Deserialize, Clone)]
pub struct ServerConfig {
    pub socket_path: String,
    pub reindex_interval_secs: u64,
}

#[derive(Deserialize, Clone)]
pub struct HnswConfig {
    pub max_connections: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub dimension: usize,
}

#[derive(Deserialize, Clone)]
pub struct ConsolidationConfig {
    pub stale_days: u32,
    pub min_score: f64,
}

impl Default for ConsolidationConfig {
    fn default() -> Self {
        Self {
            stale_days: DEFAULT_CONSOLIDATION_STALE_DAYS,
            min_score: DEFAULT_CONSOLIDATION_MIN_SCORE,
        }
    }
}

impl Config {
    pub fn load() -> Result<Self, CoreError> {
        let local_path = Path::new(CONFIG_LOCAL_PATH);
        if local_path.exists() {
            return Self::load_from_path(CONFIG_LOCAL_PATH);
        }

        if let Some(home) = home_directory() {
            let home_config = Path::new(&home).join(CONFIG_HOME_SUBDIR);
            if home_config.exists() {
                return Self::load_from_path(home_config.to_str().unwrap_or_default());
            }
        }

        let mut config = Self::default();
        config.apply_env_overrides();
        Ok(config)
    }

    pub fn load_from_path(path: &str) -> Result<Self, CoreError> {
        let content = fs::read_to_string(path).map_err(|_| CoreError::ConfigNotFound)?;
        let mut config: Config =
            toml::from_str(&content).map_err(|error| CoreError::ConfigParseError(error.to_string()))?;
        config.apply_env_overrides();
        Ok(config)
    }

    pub fn build_embedding_provider(&self) -> Result<Box<dyn EmbeddingProvider>, CoreError> {
        let dimension = self.embedding.dimension.unwrap_or(DEFAULT_EMBEDDING_DIMENSION);
        match self.embedding.provider.as_str() {
            "voyage" => {
                let api_key = self
                    .embedding
                    .api_key
                    .as_deref()
                    .filter(|key| !key.is_empty())
                    .ok_or_else(|| CoreError::InvalidProvider("voyage requires api_key".into()))?;
                let provider = VoyageEmbeddingProvider::new(api_key.to_owned())?;
                Ok(Box::new(provider))
            }
            "deterministic" => Ok(Box::new(DeterministicEmbeddingProvider { dimension })),
            other => Err(CoreError::InvalidProvider(format!(
                "{other} embedding not supported"
            ))),
        }
    }

    pub fn build_text_generator(&self) -> Result<Box<dyn TextGenerator>, CoreError> {
        match self.llm.provider.as_str() {
            "openai" => {
                let api_key = self
                    .llm
                    .api_key
                    .as_deref()
                    .filter(|key| !key.is_empty())
                    .ok_or_else(|| CoreError::InvalidProvider("openai requires api_key".into()))?;
                let generator = OpenAITextGenerator::new(api_key.to_owned())?;
                Ok(Box::new(generator))
            }
            other => Err(CoreError::InvalidProvider(format!(
                "{other} llm not supported"
            ))),
        }
    }

    pub fn build_hnsw_params(&self) -> Result<HnswParams, CoreError> {
        HnswParams::new(self.hnsw.dimension)?
            .with_max_connections(self.hnsw.max_connections)?
            .with_ef_construction(self.hnsw.ef_construction)?
            .with_ef_search(self.hnsw.ef_search)
            .map_err(CoreError::Hnsw)
    }

    pub fn resolve_database_path(&self) -> String {
        expand_tilde(&self.database.path)
    }

    fn apply_env_overrides(&mut self) {
        if let Ok(value) = std::env::var("ENGRAM_DB_PATH") {
            self.database.path = value;
        }
        if let Ok(value) = std::env::var("ENGRAM_SOCKET_PATH") {
            self.server.socket_path = value;
        }
        if let Ok(value) = std::env::var("ENGRAM_EMBEDDING_MODEL") {
            self.embedding.model = Some(value);
        }
        if let Ok(value) = std::env::var("ENGRAM_LLM_MODEL") {
            self.llm.model = Some(value);
        }
        self.apply_provider_api_key_overrides();
    }

    fn apply_provider_api_key_overrides(&mut self) {
        if self.embedding.provider == "voyage"
            && let Ok(value) = std::env::var("ENGRAM_VOYAGE_API_KEY") {
                self.embedding.api_key = Some(value);
            }
        if self.llm.provider == "openai"
            && let Ok(value) = std::env::var("ENGRAM_OPENAI_API_KEY") {
                self.llm.api_key = Some(value);
            }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            database: DatabaseConfig {
                path: DEFAULT_DB_PATH.into(),
            },
            embedding: EmbeddingConfig {
                provider: DEFAULT_EMBEDDING_PROVIDER.into(),
                api_key: None,
                model: Some(DEFAULT_EMBEDDING_MODEL.into()),
                dimension: Some(DEFAULT_EMBEDDING_DIMENSION),
            },
            llm: LlmConfig {
                provider: DEFAULT_LLM_PROVIDER.into(),
                api_key: None,
                model: Some(DEFAULT_LLM_MODEL.into()),
            },
            server: ServerConfig {
                socket_path: DEFAULT_SOCKET_PATH.into(),
                reindex_interval_secs: DEFAULT_REINDEX_INTERVAL_SECS,
            },
            hnsw: HnswConfig {
                max_connections: DEFAULT_HNSW_MAX_CONNECTIONS,
                ef_construction: DEFAULT_HNSW_EF_CONSTRUCTION,
                ef_search: DEFAULT_HNSW_EF_SEARCH,
                dimension: DEFAULT_EMBEDDING_DIMENSION,
            },
            consolidation: ConsolidationConfig::default(),
        }
    }
}

pub fn expand_tilde(path: &str) -> String {
    if let Some(rest) = path.strip_prefix("~/")
        && let Some(home) = home_directory() {
            return format!("{home}/{rest}");
        }
    path.to_string()
}

pub fn home_directory() -> Option<String> {
    std::env::var("HOME").ok()
}

struct DeterministicEmbeddingProvider {
    dimension: usize,
}

impl EmbeddingProvider for DeterministicEmbeddingProvider {
    fn embed(&self, text: &str) -> Result<Vec<f32>, engram_llm_client::ApiError> {
        let mut embedding = vec![0.0_f32; self.dimension];
        for (index, byte) in text.bytes().enumerate() {
            embedding[index % self.dimension] += byte as f32 / 255.0;
        }
        let norm: f32 = embedding.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-10);
        for value in &mut embedding {
            *value /= norm;
        }
        Ok(embedding)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn model_name(&self) -> &str {
        "deterministic"
    }
}
