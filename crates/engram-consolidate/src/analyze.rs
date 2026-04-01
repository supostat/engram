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

#[derive(Debug, Clone)]
pub struct AnalysisResult {
    pub recommendations: Vec<Recommendation>,
    pub analyzed_count: usize,
}

pub fn analyze(
    database: &Database,
    preview_result: &PreviewResult,
    text_generator: Option<&dyn TextGenerator>,
) -> Result<AnalysisResult, ConsolidateError> {
    let mut recommendations = Vec::new();
    let mut analyzed_count: usize = 0;

    for group in &preview_result.duplicates {
        let group_recommendations =
            analyze_duplicate_group(database, group, text_generator)?;
        analyzed_count += 1 + group.duplicate_ids.len();
        recommendations.extend(group_recommendations);
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
    })
}

fn analyze_duplicate_group(
    database: &Database,
    group: &crate::preview::DuplicateGroup,
    text_generator: Option<&dyn TextGenerator>,
) -> Result<Vec<Recommendation>, ConsolidateError> {
    match text_generator {
        Some(generator) => analyze_with_llm(database, group, generator),
        None => Ok(analyze_with_heuristic(database, group)),
    }
}

fn analyze_with_llm(
    database: &Database,
    group: &crate::preview::DuplicateGroup,
    text_generator: &dyn TextGenerator,
) -> Result<Vec<Recommendation>, ConsolidateError> {
    let primary = database.get_memory(&group.primary_id)?;
    let mut recommendations = Vec::new();

    for duplicate_id in &group.duplicate_ids {
        let duplicate = database.get_memory(duplicate_id)?;
        let prompt = build_merge_prompt(&primary, &duplicate);
        let response = text_generator
            .generate(&prompt)
            .map_err(|error| ConsolidateError::AnalysisFailed(error.to_string()))?;
        let recommendation = parse_llm_response(
            &group.primary_id,
            duplicate_id,
            &response,
        );
        recommendations.push(recommendation);
    }
    Ok(recommendations)
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
        primary.id, primary.context, primary.action, primary.result,
        duplicate.id, duplicate.context, duplicate.action, duplicate.result,
    )
}

fn parse_llm_response(
    primary_id: &str,
    duplicate_id: &str,
    response: &str,
) -> Recommendation {
    let normalized = response.trim().to_uppercase();
    let normalized_first_word = normalized.split_whitespace().next().unwrap_or("");
    if normalized_first_word == "MERGE" {
        Recommendation {
            action: RecommendedAction::Merge {
                source_id: primary_id.to_string(),
                target_id: duplicate_id.to_string(),
            },
            confidence: LLM_MERGE_CONFIDENCE,
            reasoning: "LLM recommended merge".to_string(),
        }
    } else {
        Recommendation {
            action: RecommendedAction::Keep {
                memory_id: duplicate_id.to_string(),
            },
            confidence: LLM_KEEP_CONFIDENCE,
            reasoning: "LLM recommended keeping both".to_string(),
        }
    }
}

fn analyze_with_heuristic(
    database: &Database,
    group: &crate::preview::DuplicateGroup,
) -> Vec<Recommendation> {
    let primary = match database.get_memory(&group.primary_id) {
        Ok(memory) => memory,
        Err(_) => return Vec::new(),
    };
    let mut recommendations = Vec::new();

    for duplicate_id in &group.duplicate_ids {
        let duplicate = match database.get_memory(duplicate_id) {
            Ok(memory) => memory,
            Err(_) => continue,
        };
        let recommendation =
            heuristic_merge_decision(&primary, &duplicate);
        recommendations.push(recommendation);
    }
    recommendations
}

fn heuristic_merge_decision(
    primary: &engram_storage::Memory,
    duplicate: &engram_storage::Memory,
) -> Recommendation {
    let primary_wins = primary.score > duplicate.score
        || (primary.score == duplicate.score
            && primary.used_count >= duplicate.used_count);

    if primary_wins {
        Recommendation {
            action: RecommendedAction::Merge {
                source_id: primary.id.clone(),
                target_id: duplicate.id.clone(),
            },
            confidence: HEURISTIC_MERGE_CONFIDENCE,
            reasoning: "heuristic: primary has higher score or usage".to_string(),
        }
    } else {
        Recommendation {
            action: RecommendedAction::Merge {
                source_id: duplicate.id.clone(),
                target_id: primary.id.clone(),
            },
            confidence: HEURISTIC_MERGE_CONFIDENCE,
            reasoning: "heuristic: duplicate has higher score or usage".to_string(),
        }
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
