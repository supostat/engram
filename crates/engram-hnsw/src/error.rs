use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum HnswError {
    /// 3001: Index data is corrupted
    IndexCorrupted(String),
    /// 3002: Vector dimension does not match index dimension
    DimensionMismatch { expected: usize, got: usize },
    /// 3003: Index needs to be rebuilt
    RebuildRequired,
    /// Node not found in the graph
    NodeNotFound(u64),
    /// Duplicate node ID
    DuplicateNode(u64),
    /// Empty vector provided
    EmptyVector,
    /// Invalid parameter value
    InvalidParameter(String),
}

impl fmt::Display for HnswError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IndexCorrupted(detail) => {
                write!(formatter, "[3001] index corrupted: {detail}")
            }
            Self::DimensionMismatch { expected, got } => {
                write!(
                    formatter,
                    "[3002] dimension mismatch: expected {expected}, got {got}"
                )
            }
            Self::RebuildRequired => {
                write!(formatter, "[3003] index rebuild required")
            }
            Self::NodeNotFound(id) => {
                write!(formatter, "node not found: {id}")
            }
            Self::DuplicateNode(id) => {
                write!(formatter, "duplicate node: {id}")
            }
            Self::EmptyVector => {
                write!(formatter, "empty vector provided")
            }
            Self::InvalidParameter(detail) => {
                write!(formatter, "invalid parameter: {detail}")
            }
        }
    }
}

impl std::error::Error for HnswError {}
