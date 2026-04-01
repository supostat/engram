use crate::action::{Contextualization, LlmSelection, Proactivity, SearchStrategy};
use crate::defaults::{ModeDefaults, defaults_for_mode};
use crate::mode::Mode;
use crate::q_table::QTable;

pub struct RouterDecision {
    pub mode: Mode,
    pub search_strategy: SearchStrategy,
    pub llm_selection: LlmSelection,
    pub contextualization: Contextualization,
    pub proactivity: Proactivity,
    pub memory_type_priority: Vec<String>,
    pub similarity_threshold: f32,
    pub top_k: usize,
}

pub struct Router {
    search_table: QTable,
    llm_table: QTable,
    context_table: QTable,
    proactivity_table: QTable,
    alpha: f32,
    epsilon: f32,
}

impl Router {
    pub fn new(alpha: f32, epsilon: f32) -> Self {
        Self {
            search_table: QTable::new(),
            llm_table: QTable::new(),
            context_table: QTable::new(),
            proactivity_table: QTable::new(),
            alpha,
            epsilon,
        }
    }

    pub fn decide(&self, mode: Mode, rng_value: f32) -> RouterDecision {
        let defaults = defaults_for_mode(mode);
        let state = mode.as_str();
        let actions = self.choose_all_levels(state, &defaults, rng_value);
        build_decision(mode, actions, &defaults)
    }

    pub fn update(&mut self, mode: Mode, decision: &RouterDecision, reward: f32) {
        let state = mode.as_str();
        let alpha = self.alpha;
        let actions: [&str; 4] = [
            decision.search_strategy.as_str(),
            decision.llm_selection.as_str(),
            decision.contextualization.as_str(),
            decision.proactivity.as_str(),
        ];
        let tables = [
            &mut self.search_table,
            &mut self.llm_table,
            &mut self.context_table,
            &mut self.proactivity_table,
        ];
        for (table, action) in tables.into_iter().zip(actions) {
            update_one_table(table, state, action, reward, alpha);
        }
    }

    fn choose_all_levels(
        &self,
        state: &str,
        defaults: &ModeDefaults,
        rng_value: f32,
    ) -> ChosenActions {
        ChosenActions {
            search_strategy: self.choose_for_level(
                &self.search_table,
                state,
                SearchStrategy::all_variants(),
                defaults.search_strategy,
                rng_value,
            ),
            llm_selection: self.choose_for_level(
                &self.llm_table,
                state,
                LlmSelection::all_variants(),
                defaults.llm_selection,
                rng_value,
            ),
            contextualization: self.choose_for_level(
                &self.context_table,
                state,
                Contextualization::all_variants(),
                defaults.contextualization,
                rng_value,
            ),
            proactivity: self.choose_for_level(
                &self.proactivity_table,
                state,
                Proactivity::all_variants(),
                defaults.proactivity,
                rng_value,
            ),
        }
    }

    fn choose_for_level<A: ActionVariant>(
        &self,
        table: &QTable,
        state: &str,
        variants: &[A],
        default: A,
        rng_value: f32,
    ) -> A {
        if rng_value < self.epsilon {
            return pick_random(variants, rng_value, self.epsilon);
        }
        pick_best_or_default(table, state, variants, default)
    }
}

struct ChosenActions {
    search_strategy: SearchStrategy,
    llm_selection: LlmSelection,
    contextualization: Contextualization,
    proactivity: Proactivity,
}

fn build_decision(mode: Mode, actions: ChosenActions, defaults: &ModeDefaults) -> RouterDecision {
    let memory_type_priority = defaults
        .memory_type_priority
        .iter()
        .map(|s| (*s).to_string())
        .collect();
    RouterDecision {
        mode,
        search_strategy: actions.search_strategy,
        llm_selection: actions.llm_selection,
        contextualization: actions.contextualization,
        proactivity: actions.proactivity,
        memory_type_priority,
        similarity_threshold: defaults.similarity_threshold,
        top_k: defaults.top_k_min,
    }
}

fn update_one_table(table: &mut QTable, state: &str, action: &str, reward: f32, alpha: f32) {
    table.update(state, action, reward, alpha);
}

trait ActionVariant: Copy {
    fn as_str(&self) -> &'static str;
}

impl ActionVariant for SearchStrategy {
    fn as_str(&self) -> &'static str {
        SearchStrategy::as_str(self)
    }
}

impl ActionVariant for LlmSelection {
    fn as_str(&self) -> &'static str {
        LlmSelection::as_str(self)
    }
}

impl ActionVariant for Contextualization {
    fn as_str(&self) -> &'static str {
        Contextualization::as_str(self)
    }
}

impl ActionVariant for Proactivity {
    fn as_str(&self) -> &'static str {
        Proactivity::as_str(self)
    }
}

fn pick_random<A: ActionVariant>(variants: &[A], rng_value: f32, epsilon: f32) -> A {
    let normalized = rng_value / epsilon;
    let index = (normalized * variants.len() as f32) as usize;
    let clamped = index.min(variants.len() - 1);
    variants.get(clamped).copied().unwrap_or(variants[0])
}

fn pick_best_or_default<A: ActionVariant>(
    table: &QTable,
    state: &str,
    variants: &[A],
    default: A,
) -> A {
    let mut best_action = None;
    let mut best_value = f32::NEG_INFINITY;
    let mut has_any_entry = false;

    for variant in variants {
        let count = table.update_count(state, variant.as_str());
        if count > 0 {
            has_any_entry = true;
            let q_value = table.get(state, variant.as_str());
            if q_value > best_value {
                best_value = q_value;
                best_action = Some(*variant);
            }
        }
    }

    if has_any_entry {
        best_action.unwrap_or(default)
    } else {
        default
    }
}
