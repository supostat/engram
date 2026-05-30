use std::fmt;

use engram_consolidate::ConsolidateError;
use engram_hnsw::HnswError;
use engram_llm_client::ApiError;
use engram_storage::StorageError;

#[derive(Debug)]
pub enum CoreError {
    ConfigNotFound,
    ConfigParseError(String),
    InvalidProvider(String),
    IndexCorrupted(String),
    RebuildFailed(String),
    SocketError(String),
    DispatchError(String),
    ConfigReadOnly,
    ExportFailed(String),
    ImportVersionMismatch(u64),
    ImportFailed(String),
    InitFailed(String),
    Storage(StorageError),
    Hnsw(HnswError),
    Api(ApiError),
    TrainerFailed(String),
    TrainerTimeout,
    TrainerMalformedOutput(String),
    ProjectDirNotFound,
    LegacyDatabaseDetected {
        legacy_path: String,
        project_path: String,
    },
    MigrationSourceNotFound,
    MigrationFailed(String),
    EmbeddingModelMismatch {
        stored: String,
        configured: String,
    },
    IndexHashCollision {
        hash: u64,
        existing_id: String,
        conflicting_id: String,
    },
    ConfigValidation(String),
    Consolidation(ConsolidateError),
}

impl fmt::Display for CoreError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ConfigNotFound => {
                write!(formatter, "[6001] config not found")
            }
            Self::ConfigParseError(message) => {
                write!(formatter, "[6002] config parse error: {message}")
            }
            Self::InvalidProvider(message) => {
                write!(formatter, "[6003] invalid provider: {message}")
            }
            Self::IndexCorrupted(message) => {
                write!(formatter, "[6004] index corrupted: {message}")
            }
            Self::RebuildFailed(message) => {
                write!(formatter, "[6005] rebuild failed: {message}")
            }
            Self::SocketError(message) => {
                write!(formatter, "[6006] socket error: {message}")
            }
            Self::DispatchError(message) => {
                write!(formatter, "[6007] dispatch error: {message}")
            }
            Self::ConfigReadOnly => {
                write!(formatter, "[6008] config is read-only")
            }
            Self::ExportFailed(message) => {
                write!(formatter, "[6009] export failed: {message}")
            }
            Self::ImportVersionMismatch(version) => {
                write!(
                    formatter,
                    "[6010] import version mismatch: expected 1, got {version}"
                )
            }
            Self::ImportFailed(message) => {
                write!(formatter, "[6011] import failed: {message}")
            }
            Self::InitFailed(message) => {
                write!(formatter, "[6012] init failed: {message}")
            }
            Self::TrainerFailed(message) => {
                write!(formatter, "[6013] trainer failed: {message}")
            }
            Self::TrainerTimeout => {
                write!(formatter, "[6014] trainer timeout")
            }
            Self::TrainerMalformedOutput(message) => {
                write!(formatter, "[6015] trainer malformed output: {message}")
            }
            Self::ProjectDirNotFound => {
                write!(
                    formatter,
                    "[6016] project directory not found: no .engram/ in cwd or ancestors (run 'engram init')"
                )
            }
            Self::LegacyDatabaseDetected {
                legacy_path,
                project_path,
            } => {
                write!(
                    formatter,
                    "[6017] legacy global database detected at {legacy_path}. Run `engram migrate` to import into project {project_path}, or `engram init` to start fresh."
                )
            }
            Self::MigrationSourceNotFound => {
                write!(
                    formatter,
                    "[6018] migration source not found: no legacy database at ~/.engram/engram.db (nothing to migrate)"
                )
            }
            Self::MigrationFailed(message) => {
                write!(formatter, "[6019] migration failed: {message}")
            }
            Self::EmbeddingModelMismatch { stored, configured } => {
                write!(
                    formatter,
                    "[6020] embedding model mismatch: database was last embedded with `{stored}`, but config specifies `{configured}`. Run `engram reembed` to re-compute embeddings, then restart the daemon."
                )
            }
            Self::IndexHashCollision {
                hash,
                existing_id,
                conflicting_id,
            } => {
                write!(
                    formatter,
                    "[6021] index hash collision: hash {hash:#x} already mapped to '{existing_id}', refusing '{conflicting_id}'"
                )
            }
            Self::ConfigValidation(message) => {
                write!(formatter, "[6022] invalid configuration: {message}")
            }
            Self::Storage(error) => error.fmt(formatter),
            Self::Hnsw(error) => error.fmt(formatter),
            Self::Api(error) => error.fmt(formatter),
            Self::Consolidation(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for CoreError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Storage(error) => Some(error),
            Self::Hnsw(error) => Some(error),
            Self::Api(error) => Some(error),
            Self::Consolidation(error) => Some(error),
            _ => None,
        }
    }
}

impl From<StorageError> for CoreError {
    fn from(error: StorageError) -> Self {
        Self::Storage(error)
    }
}

impl From<HnswError> for CoreError {
    fn from(error: HnswError) -> Self {
        Self::Hnsw(error)
    }
}

impl From<ApiError> for CoreError {
    fn from(error: ApiError) -> Self {
        Self::Api(error)
    }
}

impl From<ConsolidateError> for CoreError {
    fn from(error: ConsolidateError) -> Self {
        Self::Consolidation(error)
    }
}
