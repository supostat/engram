use crate::error::RouterError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SearchStrategy {
    HighThreshold,
    MediumThreshold,
    LowThreshold,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LlmSelection {
    Cheap,
    Balanced,
    Expensive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Contextualization {
    Raw,
    Summarize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Proactivity {
    Passive,
    Proactive,
}

macro_rules! impl_action {
    ($type:ident, $all:ident, [$( ($variant:ident, $str:expr) ),+ $(,)?]) => {
        const $all: [$type; { 0 $( + { let _ = $type::$variant; 1 } )+ }] = [
            $( $type::$variant ),+
        ];

        impl $type {
            pub fn all_variants() -> &'static [Self] {
                &$all
            }

            pub fn as_str(&self) -> &'static str {
                match self {
                    $( Self::$variant => $str ),+
                }
            }

            pub fn parse(source: &str) -> Result<Self, RouterError> {
                match source {
                    $( $str => Ok(Self::$variant), )+
                    _ => Err(RouterError::UnknownAction(source.to_string())),
                }
            }
        }
    };
}

impl_action!(
    SearchStrategy,
    ALL_SEARCH_STRATEGIES,
    [
        (HighThreshold, "high_threshold"),
        (MediumThreshold, "medium_threshold"),
        (LowThreshold, "low_threshold"),
    ]
);

impl_action!(
    LlmSelection,
    ALL_LLM_SELECTIONS,
    [
        (Cheap, "cheap"),
        (Balanced, "balanced"),
        (Expensive, "expensive"),
    ]
);

impl_action!(
    Contextualization,
    ALL_CONTEXTUALIZATIONS,
    [(Raw, "raw"), (Summarize, "summarize"),]
);

impl_action!(
    Proactivity,
    ALL_PROACTIVITIES,
    [(Passive, "passive"), (Proactive, "proactive"),]
);
