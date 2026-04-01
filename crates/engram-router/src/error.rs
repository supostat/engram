use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum RouterError {
    /// 4001: Unknown mode string
    UnknownMode(String),
    /// 4002: Could not detect mode from text
    ModeDetectionFailed,
    /// 4003: Unknown action string
    UnknownAction(String),
}

impl fmt::Display for RouterError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownMode(mode) => {
                write!(formatter, "unknown mode: {mode}")
            }
            Self::ModeDetectionFailed => {
                write!(formatter, "could not detect mode from text")
            }
            Self::UnknownAction(action) => {
                write!(formatter, "unknown action: {action}")
            }
        }
    }
}

impl std::error::Error for RouterError {}
