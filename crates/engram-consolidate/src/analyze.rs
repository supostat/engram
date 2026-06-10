use engram_llm_client::TextGenerator;
use engram_storage::Database;

use crate::error::ConsolidateError;
use crate::preview::PreviewResult;

const LLM_MERGE_CONFIDENCE: f32 = 0.8;
const LLM_KEEP_CONFIDENCE: f32 = 0.7;
const HEURISTIC_MERGE_CONFIDENCE: f32 = 0.6;
const STALE_ARCHIVE_CONFIDENCE: f32 = 0.9;
const GARBAGE_DELETE_CONFIDENCE: f32 = 0.95;

#[derive(Debug, Clone)]
pub enum RecommendedAction {
    Merge {
        source_id: String,
        target_id: String,
    },
    Delete {
        memory_id: String,
    },
    Archive {
        memory_id: String,
    },
    Keep {
        memory_id: String,
    },
}

#[derive(Debug, Clone)]
pub struct Recommendation {
    pub action: RecommendedAction,
    pub confidence: f32,
    pub reasoning: String,
}

/// `errors` collects per-member analysis failures (LLM call failures,
/// unloadable groups) without aborting the batch; `analyzed_count` counts
/// only memories whose analysis actually completed.
#[derive(Debug, Clone)]
pub struct AnalysisResult {
    pub recommendations: Vec<Recommendation>,
    pub analyzed_count: usize,
    pub errors: Vec<String>,
}

struct GroupAnalysis {
    recommendations: Vec<Recommendation>,
    errors: Vec<String>,
    analyzed_members: usize,
}

pub fn analyze(
    database: &Database,
    preview_result: &PreviewResult,
    text_generator: Option<&dyn TextGenerator>,
) -> Result<AnalysisResult, ConsolidateError> {
    let mut recommendations = Vec::new();
    let mut errors = Vec::new();
    let mut analyzed_count: usize = 0;

    for group in &preview_result.duplicates {
        let group_analysis = analyze_duplicate_group(database, group, text_generator);
        analyzed_count += group_analysis.analyzed_members;
        recommendations.extend(group_analysis.recommendations);
        errors.extend(group_analysis.errors);
    }
    for stale_id in &preview_result.stale {
        recommendations.push(build_archive_recommendation(stale_id));
        analyzed_count += 1;
    }
    for garbage_id in &preview_result.garbage {
        recommendations.push(build_delete_recommendation(garbage_id));
        analyzed_count += 1;
    }
    Ok(AnalysisResult {
        recommendations,
        analyzed_count,
        errors,
    })
}

fn analyze_duplicate_group(
    database: &Database,
    group: &crate::preview::DuplicateGroup,
    text_generator: Option<&dyn TextGenerator>,
) -> GroupAnalysis {
    let members = match load_group_members(database, group) {
        Ok(members) => members,
        Err(error) => {
            return GroupAnalysis {
                recommendations: Vec::new(),
                errors: vec![format!("analyze group {}: {error}", group.primary_id)],
                analyzed_members: 0,
            };
        }
    };
    match text_generator {
        Some(generator) => analyze_members_with_llm(&members, generator),
        None => analyze_members_with_heuristic(&members),
    }
}

fn load_group_members(
    database: &Database,
    group: &crate::preview::DuplicateGroup,
) -> Result<Vec<engram_storage::Memory>, ConsolidateError> {
    let mut members = Vec::with_capacity(1 + group.duplicate_ids.len());
    members.push(database.get_memory(&group.primary_id)?);
    for duplicate_id in &group.duplicate_ids {
        members.push(database.get_memory(duplicate_id)?);
    }
    Ok(members)
}

// Picks the single memory the rest of the group folds into. Total order, applied in
// priority sequence: a non-insight beats an insight, then higher score, then higher
// usage, then the earlier creation, then the smaller id as a deterministic final
// tie-break. An insight is therefore never chosen while any non-insight member exists.
fn choose_survivor(members: &[engram_storage::Memory]) -> &engram_storage::Memory {
    members
        .iter()
        .max_by(|left, right| {
            let left_non_insight = u8::from(left.memory_type != "insight");
            let right_non_insight = u8::from(right.memory_type != "insight");
            left_non_insight
                .cmp(&right_non_insight)
                .then_with(|| left.score.total_cmp(&right.score))
                .then_with(|| left.used_count.cmp(&right.used_count))
                .then_with(|| right.created_at.cmp(&left.created_at))
                .then_with(|| right.id.cmp(&left.id))
        })
        .expect("a duplicate group always has at least one member")
}

fn analyze_members_with_llm(
    members: &[engram_storage::Memory],
    text_generator: &dyn TextGenerator,
) -> GroupAnalysis {
    let survivor = choose_survivor(members);
    let mut recommendations = Vec::new();
    let mut errors = Vec::new();
    // The survivor needs no LLM verdict, so it always counts as analyzed.
    let mut analyzed_members: usize = 1;

    for member in members {
        if member.id == survivor.id {
            continue;
        }
        let prompt = build_merge_prompt(survivor, member);
        match text_generator.generate(&prompt) {
            Ok(response) => {
                recommendations.push(parse_llm_response(&survivor.id, &member.id, &response));
                analyzed_members += 1;
            }
            Err(error) => {
                errors.push(format!("analyze {}->{}: {error}", member.id, survivor.id));
            }
        }
    }
    GroupAnalysis {
        recommendations,
        errors,
        analyzed_members,
    }
}

fn build_merge_prompt(
    primary: &engram_storage::Memory,
    duplicate: &engram_storage::Memory,
) -> String {
    format!(
        "Compare these two memories and decide: MERGE or KEEP_BOTH.\n\n\
         Memory A (id={}):\n  context: {}\n  action: {}\n  result: {}\n\n\
         Memory B (id={}):\n  context: {}\n  action: {}\n  result: {}\n\n\
         Respond with exactly one word: MERGE or KEEP_BOTH",
        primary.id,
        primary.context,
        primary.action,
        primary.result,
        duplicate.id,
        duplicate.context,
        duplicate.action,
        duplicate.result,
    )
}

fn parse_llm_response(survivor_id: &str, member_id: &str, response: &str) -> Recommendation {
    let normalized = response.trim().to_uppercase();
    let normalized_first_word = normalized.split_whitespace().next().unwrap_or("");
    if normalized_first_word == "MERGE" {
        Recommendation {
            action: RecommendedAction::Merge {
                source_id: survivor_id.to_string(),
                target_id: member_id.to_string(),
            },
            confidence: LLM_MERGE_CONFIDENCE,
            reasoning: "LLM recommended merge".to_string(),
        }
    } else {
        Recommendation {
            action: RecommendedAction::Keep {
                memory_id: member_id.to_string(),
            },
            confidence: LLM_KEEP_CONFIDENCE,
            reasoning: "LLM recommended keeping both".to_string(),
        }
    }
}

fn analyze_members_with_heuristic(members: &[engram_storage::Memory]) -> GroupAnalysis {
    let survivor = choose_survivor(members);
    let recommendations = members
        .iter()
        .filter(|member| member.id != survivor.id)
        .map(|member| Recommendation {
            action: RecommendedAction::Merge {
                source_id: survivor.id.clone(),
                target_id: member.id.clone(),
            },
            confidence: HEURISTIC_MERGE_CONFIDENCE,
            reasoning: "heuristic: survivor has higher score or usage".to_string(),
        })
        .collect();
    GroupAnalysis {
        recommendations,
        errors: Vec::new(),
        analyzed_members: members.len(),
    }
}

fn build_archive_recommendation(memory_id: &str) -> Recommendation {
    Recommendation {
        action: RecommendedAction::Archive {
            memory_id: memory_id.to_string(),
        },
        confidence: STALE_ARCHIVE_CONFIDENCE,
        reasoning: "stale: low score with no usage".to_string(),
    }
}

fn build_delete_recommendation(memory_id: &str) -> Recommendation {
    Recommendation {
        action: RecommendedAction::Delete {
            memory_id: memory_id.to_string(),
        },
        confidence: GARBAGE_DELETE_CONFIDENCE,
        reasoning: "garbage: broken parent reference".to_string(),
    }
}
