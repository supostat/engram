use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use serde::{Deserialize, Serialize};

use engram_hnsw::HnswParams;
use engram_llm_client::{
    EmbeddingProvider, LocalTextGenerator, OllamaEmbeddingProvider, OpenAITextGenerator,
    TextGenerator, VoyageEmbeddingProvider,
};

use crate::config_loader::{
    deep_merge, load_global_config_tree, load_project_config_tree, restore_secret, secret_at,
};
use crate::error::CoreError;

/// Path of an `engram.toml` relative to its `.engram/` parent. Used for both
/// the global config under `$HOME` and project-local configs under a
/// discovered project root.
pub(crate) const ENGRAM_CONFIG_SUBPATH: &str = ".engram/engram.toml";
const PROJECT_DIR_MARKER: &str = ".engram";

const EMBEDDING_SECTION: &str = "embedding";
const LLM_SECTION: &str = "llm";

const DEFAULT_EMBEDDING_PROVIDER: &str = "voyage";
pub const DEFAULT_EMBEDDING_MODEL: &str = "voyage-4";
const DEFAULT_EMBEDDING_DIMENSION: usize = 1024;
pub const DEFAULT_EMBEDDING_HOST: &str = "http://localhost:11434";
const DEFAULT_HYDE_THRESHOLD: usize = 0;
const DEFAULT_LLM_PROVIDER: &str = "openai";
const DEFAULT_LLM_MODEL: &str = "gpt-4o-mini";
const DEFAULT_REINDEX_INTERVAL_SECS: u64 = 3600;
const DEFAULT_HNSW_MAX_CONNECTIONS: usize = 16;
const DEFAULT_HNSW_EF_CONSTRUCTION: usize = 200;
const DEFAULT_HNSW_EF_SEARCH: usize = 40;
const DEFAULT_CONSOLIDATION_STALE_DAYS: u32 = 90;
const DEFAULT_CONSOLIDATION_MIN_SCORE: f64 = 0.3;
const DEFAULT_DEDUP_THRESHOLD: f32 = 0.95;
const DEFAULT_TRAINER_BINARY: &str = "engram-trainer";
const DEFAULT_TRAINER_TIMEOUT_SECS: u64 = 300;
const DEFAULT_MODELS_PATH: &str = "~/.engram/models";
const DEFAULT_RRF_K: usize = 60;
const DEFAULT_VECTOR_WEIGHT: f64 = 0.7;
const DEFAULT_SPARSE_WEIGHT: f64 = 0.3;

#[derive(Serialize, Deserialize, Clone)]
pub struct Config {
    #[serde(default)]
    pub database: DatabaseConfig,
    pub embedding: EmbeddingConfig,
    pub llm: LlmConfig,
    pub server: ServerConfig,
    pub hnsw: HnswConfig,
    #[serde(default)]
    pub consolidation: ConsolidationConfig,
    #[serde(default)]
    pub deduplication: DeduplicationConfig,
    #[serde(default)]
    pub trainer: TrainerConfig,
    #[serde(default)]
    pub search: SearchConfig,
}

#[derive(Serialize, Deserialize, Clone, Default)]
pub struct DatabaseConfig {
    /// Legacy fallback used only when no project `.engram/` marker is found and
    /// `ENGRAM_DB_PATH` is not set. Runtime resolution always prefers the
    /// per-project layout (`<project>/.engram/engram.db`), so for normal use
    /// this can be `None`. Kept for backward-compat TOML parsing.
    #[serde(default)]
    pub path: Option<String>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct EmbeddingConfig {
    pub provider: String,
    pub api_key: Option<String>,
    pub model: Option<String>,
    pub dimension: Option<usize>,
    /// Voyage-4 output dimension (256/512/1024/2048 via Matryoshka). Omit to
    /// let the API choose its default (1024 for voyage-4). Ignored by
    /// non-Voyage providers. Must match `[hnsw].dimension` when set —
    /// mismatch fails fast at HNSW insert with `[3002] DimensionMismatch`.
    #[serde(default)]
    pub output_dimension: Option<usize>,
    /// Base URL of the Ollama daemon for the `ollama` provider (e.g.
    /// `http://localhost:11434`). Ignored by non-Ollama providers. Omit to use
    /// `DEFAULT_EMBEDDING_HOST`; override at runtime with `ENGRAM_OLLAMA_HOST`.
    #[serde(default)]
    pub host: Option<String>,
    #[serde(default)]
    pub hyde_threshold: usize,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct LlmConfig {
    pub provider: String,
    pub api_key: Option<String>,
    pub model: Option<String>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ServerConfig {
    /// Legacy fallback socket path. Used only when no project `.engram/`
    /// marker is found and `ENGRAM_SOCKET_PATH` is not set. Runtime prefers
    /// the per-project socket at `<project>/.engram/engram.sock`.
    #[serde(default)]
    pub socket_path: Option<String>,
    pub reindex_interval_secs: u64,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct HnswConfig {
    pub max_connections: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub dimension: usize,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ConsolidationConfig {
    pub stale_days: u32,
    pub min_score: f64,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct DeduplicationConfig {
    pub threshold: f32,
}

/// Hybrid-search blend parameters. `rrf_k` is the Reciprocal Rank Fusion
/// smoothing constant (larger values flatten the rank-position weighting);
/// `vector_weight` and `sparse_weight` scale the dense-vector and full-text
/// contributions before fusion.
#[derive(Serialize, Deserialize, Clone)]
pub struct SearchConfig {
    pub rrf_k: usize,
    pub vector_weight: f64,
    pub sparse_weight: f64,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct TrainerConfig {
    pub trainer_binary: String,
    pub trainer_timeout_secs: u64,
    pub models_path: String,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            trainer_binary: DEFAULT_TRAINER_BINARY.into(),
            trainer_timeout_secs: DEFAULT_TRAINER_TIMEOUT_SECS,
            models_path: DEFAULT_MODELS_PATH.into(),
        }
    }
}

impl Default for ConsolidationConfig {
    fn default() -> Self {
        Self {
            stale_days: DEFAULT_CONSOLIDATION_STALE_DAYS,
            min_score: DEFAULT_CONSOLIDATION_MIN_SCORE,
        }
    }
}

impl Default for DeduplicationConfig {
    fn default() -> Self {
        Self {
            threshold: DEFAULT_DEDUP_THRESHOLD,
        }
    }
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            rrf_k: DEFAULT_RRF_K,
            vector_weight: DEFAULT_VECTOR_WEIGHT,
            sparse_weight: DEFAULT_SPARSE_WEIGHT,
        }
    }
}

/// Validates a configured deduplication threshold lies in the half-open
/// cosine-similarity range `(0.0, 1.0]`. A threshold of `0.0` or below would
/// treat unrelated memories as duplicates; above `1.0` is unreachable for
/// normalized embeddings and signals a misconfiguration.
pub fn validate_dedup_threshold(threshold: f32) -> Result<(), CoreError> {
    if threshold > 0.0 && threshold <= 1.0 {
        return Ok(());
    }
    Err(CoreError::ConfigValidation(format!(
        "deduplication.threshold must be in (0.0, 1.0], got {threshold}"
    )))
}

/// Validates the hybrid-search blend parameters. `rrf_k` must be non-zero
/// (it appears in the Reciprocal Rank Fusion denominator `k + rank`), both
/// weights must be finite and non-negative, and at least one weight must be
/// positive — two zero weights would cancel every contribution and yield an
/// empty ranking regardless of how many results each index returns. NaN/Inf
/// weights are rejected outright (NaN passes a naive `< 0.0` range check).
pub fn validate_search_config(config: &SearchConfig) -> Result<(), CoreError> {
    if config.rrf_k == 0 {
        return Err(CoreError::ConfigValidation(
            "search.rrf_k must be greater than 0".to_string(),
        ));
    }
    if !config.vector_weight.is_finite() || !config.sparse_weight.is_finite() {
        return Err(CoreError::ConfigValidation(
            "search weights must be finite".to_string(),
        ));
    }
    if config.vector_weight < 0.0 || config.sparse_weight < 0.0 {
        return Err(CoreError::ConfigValidation(format!(
            "search weights must be non-negative, got vector_weight={}, sparse_weight={}",
            config.vector_weight, config.sparse_weight
        )));
    }
    if config.vector_weight == 0.0 && config.sparse_weight == 0.0 {
        return Err(CoreError::ConfigValidation(
            "search.vector_weight and search.sparse_weight cannot both be 0.0".to_string(),
        ));
    }
    Ok(())
}

impl Config {
    /// Loads the effective config by layering project-local `engram.toml`
    /// over the global `~/.engram/engram.toml`, then applying `ENGRAM_*` env
    /// overrides. Final priority is `env > project-local > global`.
    ///
    /// When neither config file exists the built-in defaults are used. The
    /// `api_key` of each provider is always taken from the global layer — a
    /// project-local config can never set or change a secret.
    pub fn load() -> Result<Self, CoreError> {
        let global_tree = load_global_config_tree()?;
        let project_tree = load_project_config_tree()?;

        if global_tree.is_none() && project_tree.is_none() {
            let mut config = Self::default();
            config.apply_env_overrides();
            return Ok(config);
        }

        let mut merged_tree = match global_tree {
            Some(tree) => tree,
            None => toml::Value::try_from(Self::default())
                .map_err(|error| CoreError::ConfigParseError(error.to_string()))?,
        };

        let global_embedding_secret = secret_at(&merged_tree, EMBEDDING_SECTION);
        let global_llm_secret = secret_at(&merged_tree, LLM_SECTION);

        if let Some(project_tree) = project_tree {
            deep_merge(&mut merged_tree, project_tree);
        }

        restore_secret(&mut merged_tree, EMBEDDING_SECTION, global_embedding_secret);
        restore_secret(&mut merged_tree, LLM_SECTION, global_llm_secret);

        let mut config: Config = merged_tree
            .try_into()
            .map_err(|error: toml::de::Error| CoreError::ConfigParseError(error.to_string()))?;
        config.apply_env_overrides();
        Ok(config)
    }

    pub fn load_from_path(path: &str) -> Result<Self, CoreError> {
        let content = fs::read_to_string(path).map_err(|_| CoreError::ConfigNotFound)?;
        let mut config: Config = toml::from_str(&content)
            .map_err(|error| CoreError::ConfigParseError(error.to_string()))?;
        config.apply_env_overrides();
        Ok(config)
    }

    pub fn build_embedding_provider(
        &self,
    ) -> Result<Box<dyn EmbeddingProvider + Send + Sync>, CoreError> {
        let dimension = self
            .embedding
            .dimension
            .unwrap_or(DEFAULT_EMBEDDING_DIMENSION);
        match self.embedding.provider.as_str() {
            "voyage" => {
                let api_key = self
                    .embedding
                    .api_key
                    .as_deref()
                    .filter(|key| !key.is_empty())
                    .ok_or_else(|| CoreError::InvalidProvider("voyage requires api_key".into()))?;
                let model = self
                    .embedding
                    .model
                    .clone()
                    .unwrap_or_else(|| DEFAULT_EMBEDDING_MODEL.into());
                let provider = VoyageEmbeddingProvider::with_config(
                    api_key.to_owned(),
                    model,
                    dimension,
                    self.embedding.output_dimension,
                    engram_llm_client::RetryConfig::default(),
                    "https://api.voyageai.com".into(),
                )?;
                Ok(Box::new(provider))
            }
            "ollama" => {
                let host = self
                    .embedding
                    .host
                    .clone()
                    .unwrap_or_else(|| DEFAULT_EMBEDDING_HOST.into());
                let model = self
                    .embedding
                    .model
                    .clone()
                    .unwrap_or_else(|| engram_llm_client::ollama::DEFAULT_OLLAMA_MODEL.into());
                let provider = OllamaEmbeddingProvider::new(host, model, dimension)?;
                Ok(Box::new(provider))
            }
            "deterministic" => Ok(Box::new(DeterministicEmbeddingProvider { dimension })),
            other => Err(CoreError::InvalidProvider(format!(
                "{other} embedding not supported"
            ))),
        }
    }

    pub fn build_text_generator(&self) -> Result<Box<dyn TextGenerator + Send + Sync>, CoreError> {
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
            "local" => {
                let models_path = expand_tilde(&self.trainer.models_path);
                let model_path = format!("{models_path}/text_generator.onnx");
                let tokenizer_path = format!("{models_path}/tokenizer.json");
                let generator = LocalTextGenerator::new(&model_path, &tokenizer_path)
                    .map_err(|error| CoreError::InvalidProvider(error.to_string()))?;
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

    fn apply_env_overrides(&mut self) {
        // ENGRAM_DB_PATH / ENGRAM_SOCKET_PATH are honored directly by server.rs
        // (resolve_database_path / resolve_socket_path). Copying them onto
        // `self.database.path` / `self.server.socket_path` here would be redundant
        // and could mask the per-project layout derived from `project_dir`.
        if let Ok(value) = std::env::var("ENGRAM_EMBEDDING_MODEL") {
            self.embedding.model = Some(value);
        }
        if let Ok(value) = std::env::var("ENGRAM_LLM_MODEL") {
            self.llm.model = Some(value);
        }
        if let Ok(value) = std::env::var("ENGRAM_TRAINER_BINARY") {
            self.trainer.trainer_binary = value;
        }
        if let Ok(value) = std::env::var("ENGRAM_TRAINER_TIMEOUT")
            && let Ok(secs) = value.parse::<u64>()
        {
            self.trainer.trainer_timeout_secs = secs;
        }
        if let Ok(value) = std::env::var("ENGRAM_MODELS_PATH") {
            self.trainer.models_path = value;
        }
        self.apply_provider_api_key_overrides();
    }

    fn apply_provider_api_key_overrides(&mut self) {
        if self.embedding.provider == "voyage"
            && let Ok(value) = std::env::var("ENGRAM_VOYAGE_API_KEY")
        {
            self.embedding.api_key = Some(value);
        }
        if self.embedding.provider == "ollama"
            && let Ok(value) = std::env::var("ENGRAM_OLLAMA_HOST")
        {
            self.embedding.host = Some(value);
        }
        if self.llm.provider == "openai"
            && let Ok(value) = std::env::var("ENGRAM_OPENAI_API_KEY")
        {
            self.llm.api_key = Some(value);
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            database: DatabaseConfig { path: None },
            embedding: EmbeddingConfig {
                provider: DEFAULT_EMBEDDING_PROVIDER.into(),
                api_key: None,
                model: Some(DEFAULT_EMBEDDING_MODEL.into()),
                dimension: Some(DEFAULT_EMBEDDING_DIMENSION),
                output_dimension: None,
                host: None,
                hyde_threshold: DEFAULT_HYDE_THRESHOLD,
            },
            llm: LlmConfig {
                provider: DEFAULT_LLM_PROVIDER.into(),
                api_key: None,
                model: Some(DEFAULT_LLM_MODEL.into()),
            },
            server: ServerConfig {
                socket_path: None,
                reindex_interval_secs: DEFAULT_REINDEX_INTERVAL_SECS,
            },
            hnsw: HnswConfig {
                max_connections: DEFAULT_HNSW_MAX_CONNECTIONS,
                ef_construction: DEFAULT_HNSW_EF_CONSTRUCTION,
                ef_search: DEFAULT_HNSW_EF_SEARCH,
                dimension: DEFAULT_EMBEDDING_DIMENSION,
            },
            consolidation: ConsolidationConfig::default(),
            deduplication: DeduplicationConfig::default(),
            trainer: TrainerConfig::default(),
            search: SearchConfig::default(),
        }
    }
}

pub fn resolve_project_dir(
    start: &Path,
    explicit_override: Option<&Path>,
) -> Result<PathBuf, CoreError> {
    if let Some(path) = explicit_override {
        return Ok(path.to_path_buf());
    }
    if let Ok(env_path) = std::env::var("ENGRAM_PROJECT_DIR") {
        let candidate = PathBuf::from(env_path);
        if candidate.is_absolute() {
            return Ok(candidate);
        }
    }
    let mut current = start.to_path_buf();
    loop {
        if current.join(PROJECT_DIR_MARKER).is_dir() {
            return Ok(current);
        }
        // `PathBuf::pop()` returns false at filesystem root, which terminates this loop safely.
        if !current.pop() {
            break;
        }
    }
    Err(CoreError::ProjectDirNotFound)
}

pub fn expand_tilde(path: &str) -> String {
    if let Some(rest) = path.strip_prefix("~/")
        && let Some(home) = home_directory()
    {
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

// Test instrumentation for the deterministic provider. Gated behind
// `DETERMINISTIC_PROVIDER_INSTRUMENTATION`, which defaults to OFF so
// production users (TUI demo mode, local setups without API keys) pay
// zero overhead. Tests call `enable_deterministic_provider_instrumentation`
// before parallel assertions and `disable_deterministic_provider_instrumentation`
// in teardown (typically via a drop-guard).
static DETERMINISTIC_PROVIDER_INSTRUMENTATION: AtomicBool = AtomicBool::new(false);
static DETERMINISTIC_PROVIDER_ENTRIES: AtomicUsize = AtomicUsize::new(0);
static DETERMINISTIC_PROVIDER_MAX_CONCURRENT: AtomicUsize = AtomicUsize::new(0);

/// Enable test-only instrumentation (counter + 20ms sleep) on the deterministic
/// embedding provider. Off in production by default; tests enable it before
/// reading concurrency counters.
pub fn enable_deterministic_provider_instrumentation() {
    DETERMINISTIC_PROVIDER_INSTRUMENTATION.store(true, Ordering::Relaxed);
}

/// Disable instrumentation. Tests should call this in teardown to avoid leaking
/// state across serially-run suites.
pub fn disable_deterministic_provider_instrumentation() {
    DETERMINISTIC_PROVIDER_INSTRUMENTATION.store(false, Ordering::Relaxed);
}

pub fn deterministic_provider_max_concurrent() -> usize {
    DETERMINISTIC_PROVIDER_MAX_CONCURRENT.load(Ordering::Relaxed)
}

pub fn reset_deterministic_provider_counters() {
    DETERMINISTIC_PROVIDER_ENTRIES.store(0, Ordering::Relaxed);
    DETERMINISTIC_PROVIDER_MAX_CONCURRENT.store(0, Ordering::Relaxed);
}

impl EmbeddingProvider for DeterministicEmbeddingProvider {
    fn embed(
        &self,
        text: &str,
        _input_type: Option<&str>,
    ) -> Result<Vec<f32>, engram_llm_client::ApiError> {
        // Test-only instrumentation: when enabled via
        // `enable_deterministic_provider_instrumentation`, each embed call
        // records concurrent entries and sleeps 20ms to widen the overlap
        // window for parallelism assertions. Production users never enable
        // this — a relaxed atomic load is ~1ns, so overhead is effectively
        // zero otherwise.
        let instrumented = DETERMINISTIC_PROVIDER_INSTRUMENTATION.load(Ordering::Relaxed);
        if instrumented {
            let entries = DETERMINISTIC_PROVIDER_ENTRIES.fetch_add(1, Ordering::SeqCst) + 1;
            DETERMINISTIC_PROVIDER_MAX_CONCURRENT.fetch_max(entries, Ordering::SeqCst);
            std::thread::sleep(std::time::Duration::from_millis(20));
        }

        let mut embedding = vec![0.0_f32; self.dimension];
        for (index, byte) in text.bytes().enumerate() {
            embedding[index % self.dimension] += byte as f32 / 255.0;
        }
        let norm: f32 = embedding
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt()
            .max(1e-10);
        for value in &mut embedding {
            *value /= norm;
        }

        if instrumented {
            DETERMINISTIC_PROVIDER_ENTRIES.fetch_sub(1, Ordering::SeqCst);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_provider_ignores_input_type() {
        let provider = DeterministicEmbeddingProvider { dimension: 8 };
        let text = "rust ownership rules govern compile-time memory access";

        let document = provider.embed(text, Some("document")).unwrap();
        let query = provider.embed(text, Some("query")).unwrap();
        let none = provider.embed(text, None).unwrap();

        // Deterministic provider must yield byte-identical vectors regardless
        // of input_type — test fixtures rely on this for assertion stability.
        assert_eq!(document, query);
        assert_eq!(document, none);
    }

    #[test]
    fn embedding_config_deserializes_without_output_dimension() {
        let toml_input = r#"
            provider = "voyage"
            model = "voyage-code-3"
            dimension = 1024
        "#;
        let config: EmbeddingConfig = toml::from_str(toml_input).unwrap();
        assert_eq!(config.provider, "voyage");
        assert_eq!(config.output_dimension, None);
    }

    #[test]
    fn embedding_config_deserializes_with_output_dimension() {
        let toml_input = r#"
            provider = "voyage"
            model = "voyage-4"
            dimension = 1024
            output_dimension = 1024
        "#;
        let config: EmbeddingConfig = toml::from_str(toml_input).unwrap();
        assert_eq!(config.output_dimension, Some(1024));
    }

    #[test]
    fn embedding_config_deserializes_without_host() {
        let toml_input = r#"
            provider = "ollama"
            model = "qwen3-embedding:0.6b"
            dimension = 1024
        "#;
        let config: EmbeddingConfig = toml::from_str(toml_input).unwrap();
        assert_eq!(config.host, None);
    }

    #[test]
    fn embedding_config_deserializes_with_host() {
        let toml_input = r#"
            provider = "ollama"
            model = "qwen3-embedding:0.6b"
            dimension = 1024
            host = "http://ollama.internal:11434"
        "#;
        let config: EmbeddingConfig = toml::from_str(toml_input).unwrap();
        assert_eq!(config.host.as_deref(), Some("http://ollama.internal:11434"));
    }

    #[test]
    fn build_embedding_provider_ollama_uses_prefixed_model_name() {
        let mut config = Config::default();
        config.embedding.provider = "ollama".into();
        config.embedding.model = Some("qwen3-embedding:0.6b".into());
        config.embedding.dimension = Some(1024);
        config.embedding.host = Some("http://127.0.0.1:1".into());

        let provider = config
            .build_embedding_provider()
            .expect("ollama provider builds without a live daemon");
        assert_eq!(provider.model_name(), "ollama:qwen3-embedding:0.6b");
        assert_eq!(provider.dimension(), 1024);
    }

    #[test]
    fn default_dedup_threshold_is_documented_default() {
        assert_eq!(Config::default().deduplication.threshold, 0.95);
    }

    #[test]
    fn validate_dedup_threshold_rejects_zero() {
        let error = validate_dedup_threshold(0.0).expect_err("zero is out of range");
        assert!(matches!(error, CoreError::ConfigValidation(_)));
        assert!(error.to_string().contains("[6022]"));
        assert!(error.to_string().contains("0"));
    }

    #[test]
    fn validate_dedup_threshold_rejects_negative() {
        let error = validate_dedup_threshold(-0.5).expect_err("negative is out of range");
        assert!(matches!(error, CoreError::ConfigValidation(_)));
        assert!(error.to_string().contains("-0.5"));
    }

    #[test]
    fn validate_dedup_threshold_rejects_above_one() {
        let error = validate_dedup_threshold(1.0001).expect_err("above 1.0 is out of range");
        assert!(matches!(error, CoreError::ConfigValidation(_)));
    }

    #[test]
    fn validate_dedup_threshold_accepts_one() {
        validate_dedup_threshold(1.0).expect("1.0 is the inclusive upper bound");
    }

    #[test]
    fn validate_dedup_threshold_accepts_tiny_positive() {
        validate_dedup_threshold(0.0001).expect("any positive within range is accepted");
    }

    #[test]
    fn default_search_config_matches_documented_values() {
        let search = SearchConfig::default();
        assert_eq!(search.rrf_k, 60);
        assert_eq!(search.vector_weight, 0.7);
        assert_eq!(search.sparse_weight, 0.3);
    }

    #[test]
    fn config_loads_without_search_section() {
        // The [search] table is serde-defaulted, so a TOML predating Phase 11
        // must still deserialize and fall back to the documented defaults.
        let toml_input = r#"
            [embedding]
            provider = "deterministic"

            [llm]
            provider = "openai"

            [server]
            reindex_interval_secs = 3600

            [hnsw]
            max_connections = 16
            ef_construction = 200
            ef_search = 40
            dimension = 1024
        "#;
        let config: Config = toml::from_str(toml_input).unwrap();
        assert_eq!(config.search.rrf_k, 60);
        assert_eq!(config.search.vector_weight, 0.7);
        assert_eq!(config.search.sparse_weight, 0.3);
    }

    #[test]
    fn search_config_round_trips_through_toml() {
        let toml_input = r#"
            rrf_k = 30
            vector_weight = 0.6
            sparse_weight = 0.4
        "#;
        let search: SearchConfig = toml::from_str(toml_input).unwrap();
        assert_eq!(search.rrf_k, 30);
        assert_eq!(search.vector_weight, 0.6);
        assert_eq!(search.sparse_weight, 0.4);
    }

    #[test]
    fn validate_search_config_accepts_defaults() {
        validate_search_config(&SearchConfig::default()).expect("documented defaults are valid");
    }

    #[test]
    fn validate_search_config_accepts_one_zero_weight() {
        let search = SearchConfig {
            rrf_k: 60,
            vector_weight: 0.0,
            sparse_weight: 0.3,
        };
        validate_search_config(&search).expect("a single zero weight still leaves one live term");
    }

    #[test]
    fn validate_search_config_rejects_zero_rrf_k() {
        let search = SearchConfig {
            rrf_k: 0,
            ..SearchConfig::default()
        };
        let error = validate_search_config(&search).expect_err("rrf_k of 0 divides by k+rank base");
        assert!(matches!(error, CoreError::ConfigValidation(_)));
        assert!(error.to_string().contains("rrf_k"));
    }

    #[test]
    fn validate_search_config_rejects_negative_weight() {
        let search = SearchConfig {
            rrf_k: 60,
            vector_weight: -0.1,
            sparse_weight: 0.3,
        };
        let error = validate_search_config(&search).expect_err("negative weight is invalid");
        assert!(matches!(error, CoreError::ConfigValidation(_)));
        assert!(error.to_string().contains("non-negative"));
    }

    #[test]
    fn validate_search_config_rejects_non_finite_weight() {
        let search = SearchConfig {
            rrf_k: 60,
            vector_weight: f64::NAN,
            sparse_weight: 0.3,
        };
        let error = validate_search_config(&search).expect_err("NaN weight is invalid");
        assert!(matches!(error, CoreError::ConfigValidation(_)));
        assert!(error.to_string().contains("finite"));
    }

    #[test]
    fn validate_search_config_rejects_all_zero_weights() {
        let search = SearchConfig {
            rrf_k: 60,
            vector_weight: 0.0,
            sparse_weight: 0.0,
        };
        let error = validate_search_config(&search).expect_err("both weights zero cancels output");
        assert!(matches!(error, CoreError::ConfigValidation(_)));
    }
}
