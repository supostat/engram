use crate::error::RouterError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Mode {
    Debug,
    Architecture,
    Coding,
    Review,
    Plan,
    Routine,
}

const ALL_VARIANTS: [Mode; 6] = [
    Mode::Debug,
    Mode::Architecture,
    Mode::Coding,
    Mode::Review,
    Mode::Plan,
    Mode::Routine,
];

const DEBUG_KEYWORDS: &[&str] = &[
    "bug",
    "error",
    "stack",
    "trace",
    "crash",
    "fix",
    "issue",
    "debug",
    "exception",
    "panic",
];

const ARCHITECTURE_KEYWORDS: &[&str] = &[
    "design",
    "choose",
    "structure",
    "technology",
    "framework",
    "library",
    "pattern",
    "architect",
    "component",
];

const CODING_KEYWORDS: &[&str] = &[
    "implement",
    "code",
    "function",
    "method",
    "class",
    "feature",
    "module",
    "write",
    "create",
    "add",
];

const REVIEW_KEYWORDS: &[&str] = &[
    "review", "refactor", "improve", "clean", "lint", "optimize", "quality",
];

const PLAN_KEYWORDS: &[&str] = &[
    "plan", "estimate", "risk", "assess", "schedule", "timeline", "scope", "roadmap",
];

const ROUTINE_KEYWORDS: &[&str] = &[
    "update",
    "version",
    "config",
    "dependency",
    "setup",
    "init",
    "bump",
];

/// Priority order: debug > plan > architecture > review > coding > routine
const DETECTION_ORDER: [(Mode, &[&str]); 6] = [
    (Mode::Debug, DEBUG_KEYWORDS),
    (Mode::Plan, PLAN_KEYWORDS),
    (Mode::Architecture, ARCHITECTURE_KEYWORDS),
    (Mode::Review, REVIEW_KEYWORDS),
    (Mode::Coding, CODING_KEYWORDS),
    (Mode::Routine, ROUTINE_KEYWORDS),
];

impl Mode {
    pub fn parse(source: &str) -> Result<Self, RouterError> {
        match source.to_ascii_lowercase().as_str() {
            "debug" => Ok(Self::Debug),
            "architecture" => Ok(Self::Architecture),
            "coding" => Ok(Self::Coding),
            "review" => Ok(Self::Review),
            "plan" => Ok(Self::Plan),
            "routine" => Ok(Self::Routine),
            _ => Err(RouterError::UnknownMode(source.to_string())),
        }
    }

    pub fn detect(text: &str) -> Self {
        let lowercase = text.to_ascii_lowercase();
        let words: Vec<&str> = lowercase.split_whitespace().collect();

        for (mode, keywords) in &DETECTION_ORDER {
            if has_keyword_match(&words, keywords) {
                return *mode;
            }
        }

        Self::Routine
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Debug => "debug",
            Self::Architecture => "architecture",
            Self::Coding => "coding",
            Self::Review => "review",
            Self::Plan => "plan",
            Self::Routine => "routine",
        }
    }

    pub fn all_variants() -> &'static [Mode] {
        &ALL_VARIANTS
    }
}

fn has_keyword_match(words: &[&str], keywords: &[&str]) -> bool {
    words.iter().any(|word| keywords.contains(word))
}
