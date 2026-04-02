pub mod analyze;
pub mod apply;
pub mod error;
pub mod preview;

pub use analyze::{AnalysisResult, Recommendation, RecommendedAction, analyze};
pub use apply::{ApplyResult, apply};
pub use error::ConsolidateError;
pub use preview::{DuplicateGroup, PreviewResult, preview};
