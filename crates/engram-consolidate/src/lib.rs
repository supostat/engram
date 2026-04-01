pub mod analyze;
pub mod apply;
pub mod error;
pub mod preview;

pub use analyze::{analyze, AnalysisResult, Recommendation, RecommendedAction};
pub use apply::{apply, ApplyResult};
pub use error::ConsolidateError;
pub use preview::{preview, DuplicateGroup, PreviewResult};
