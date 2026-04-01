use crate::action::{Contextualization, LlmSelection, Proactivity, SearchStrategy};
use crate::mode::Mode;

pub struct ModeDefaults {
    pub search_strategy: SearchStrategy,
    pub llm_selection: LlmSelection,
    pub contextualization: Contextualization,
    pub proactivity: Proactivity,
    pub memory_type_priority: &'static [&'static str],
    pub similarity_threshold: f32,
    pub top_k_min: usize,
    pub top_k_max: usize,
}

pub fn defaults_for_mode(mode: Mode) -> ModeDefaults {
    match mode {
        Mode::Debug => ModeDefaults {
            search_strategy: SearchStrategy::HighThreshold,
            llm_selection: LlmSelection::Cheap,
            contextualization: Contextualization::Raw,
            proactivity: Proactivity::Passive,
            memory_type_priority: &["bugfix", "pattern", "context"],
            similarity_threshold: 0.8,
            top_k_min: 3,
            top_k_max: 5,
        },
        Mode::Architecture => ModeDefaults {
            search_strategy: SearchStrategy::LowThreshold,
            llm_selection: LlmSelection::Expensive,
            contextualization: Contextualization::Summarize,
            proactivity: Proactivity::Proactive,
            memory_type_priority: &["decision", "context", "pattern"],
            similarity_threshold: 0.5,
            top_k_min: 5,
            top_k_max: 10,
        },
        Mode::Coding => ModeDefaults {
            search_strategy: SearchStrategy::MediumThreshold,
            llm_selection: LlmSelection::Cheap,
            contextualization: Contextualization::Raw,
            proactivity: Proactivity::Passive,
            memory_type_priority: &["pattern", "context", "bugfix"],
            similarity_threshold: 0.7,
            top_k_min: 3,
            top_k_max: 5,
        },
        Mode::Review => ModeDefaults {
            search_strategy: SearchStrategy::MediumThreshold,
            llm_selection: LlmSelection::Expensive,
            contextualization: Contextualization::Raw,
            proactivity: Proactivity::Passive,
            memory_type_priority: &["pattern", "decision", "bugfix"],
            similarity_threshold: 0.7,
            top_k_min: 5,
            top_k_max: 7,
        },
        Mode::Plan => ModeDefaults {
            search_strategy: SearchStrategy::LowThreshold,
            llm_selection: LlmSelection::Expensive,
            contextualization: Contextualization::Summarize,
            proactivity: Proactivity::Proactive,
            memory_type_priority: &["decision", "bugfix", "context", "pattern"],
            similarity_threshold: 0.5,
            top_k_min: 7,
            top_k_max: 10,
        },
        Mode::Routine => ModeDefaults {
            search_strategy: SearchStrategy::HighThreshold,
            llm_selection: LlmSelection::Cheap,
            contextualization: Contextualization::Raw,
            proactivity: Proactivity::Passive,
            memory_type_priority: &["context"],
            similarity_threshold: 0.8,
            top_k_min: 1,
            top_k_max: 3,
        },
    }
}
