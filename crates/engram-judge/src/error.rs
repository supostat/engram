use std::fmt;

#[derive(Debug)]
pub enum JudgeError {
    LlmUnavailable(String),
    InvalidResponse(String),
}

impl fmt::Display for JudgeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LlmUnavailable(message) => {
                write!(formatter, "judge error: llm unavailable: {message}")
            }
            Self::InvalidResponse(message) => {
                write!(formatter, "judge error: invalid response: {message}")
            }
        }
    }
}

impl std::error::Error for JudgeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}
