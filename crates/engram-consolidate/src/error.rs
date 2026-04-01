use std::fmt;

use engram_storage::StorageError;

#[derive(Debug)]
pub enum ConsolidateError {
    NoCandidates,
    IndexStale,
    InvalidMergeParams(String),
    AnalysisFailed(String),
    ApplyFailed(String),
    Storage(StorageError),
}

impl fmt::Display for ConsolidateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoCandidates => {
                write!(formatter, "[5001] no consolidation candidates found")
            }
            Self::IndexStale => {
                write!(formatter, "[5002] index is stale, rebuild required")
            }
            Self::InvalidMergeParams(message) => {
                write!(formatter, "[5003] invalid merge parameters: {message}")
            }
            Self::AnalysisFailed(message) => {
                write!(formatter, "[5004] analysis failed: {message}")
            }
            Self::ApplyFailed(message) => {
                write!(formatter, "[5005] apply failed: {message}")
            }
            Self::Storage(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for ConsolidateError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Storage(error) => Some(error),
            _ => None,
        }
    }
}

impl From<StorageError> for ConsolidateError {
    fn from(error: StorageError) -> Self {
        Self::Storage(error)
    }
}
