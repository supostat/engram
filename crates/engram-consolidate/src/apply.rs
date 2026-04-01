use engram_storage::Database;

use crate::analyze::{Recommendation, RecommendedAction};
use crate::error::ConsolidateError;

#[derive(Debug, Clone)]
pub struct ApplyResult {
    pub merged: usize,
    pub deleted: usize,
    pub archived: usize,
    pub kept: usize,
    pub errors: Vec<String>,
}

pub fn apply(
    database: &Database,
    recommendations: &[Recommendation],
    performed_by: &str,
) -> Result<ApplyResult, ConsolidateError> {
    let mut result = ApplyResult {
        merged: 0,
        deleted: 0,
        archived: 0,
        kept: 0,
        errors: Vec::new(),
    };
    let timestamp = current_utc_timestamp();
    for recommendation in recommendations {
        apply_single(database, recommendation, performed_by, &timestamp, &mut result);
    }
    Ok(result)
}

fn apply_single(
    database: &Database,
    recommendation: &Recommendation,
    performed_by: &str,
    timestamp: &str,
    result: &mut ApplyResult,
) {
    match &recommendation.action {
        RecommendedAction::Merge {
            source_id,
            target_id,
        } => apply_merge(database, source_id, target_id, performed_by, timestamp, result),
        RecommendedAction::Delete { memory_id } => {
            apply_delete(database, memory_id, performed_by, timestamp, result);
        }
        RecommendedAction::Archive { memory_id } => {
            apply_archive(database, memory_id, performed_by, timestamp, result);
        }
        RecommendedAction::Keep { .. } => {
            result.kept += 1;
        }
    }
}

fn apply_merge(
    database: &Database,
    source_id: &str,
    target_id: &str,
    performed_by: &str,
    timestamp: &str,
    result: &mut ApplyResult,
) {
    if let Err(error) = database.set_superseded_by(target_id, source_id) {
        result
            .errors
            .push(format!("merge {target_id}->{source_id}: {error}"));
        return;
    }
    if let Err(error) = database.log_consolidation(
        &generate_log_id(timestamp, result.merged),
        "merge",
        &[source_id.to_string(), target_id.to_string()],
        Some("merged duplicate"),
        performed_by,
        timestamp,
    ) {
        result.errors.push(format!("log merge: {error}"));
        return;
    }
    result.merged += 1;
}

fn apply_delete(
    database: &Database,
    memory_id: &str,
    performed_by: &str,
    timestamp: &str,
    result: &mut ApplyResult,
) {
    let log_result = database.log_consolidation(
        &generate_log_id(timestamp, result.deleted + result.merged),
        "delete",
        &[memory_id.to_string()],
        Some("garbage: broken reference"),
        performed_by,
        timestamp,
    );
    if let Err(error) = log_result {
        result.errors.push(format!("log delete: {error}"));
        return;
    }
    if let Err(error) = database.delete_memory(memory_id) {
        result
            .errors
            .push(format!("delete {memory_id}: {error}"));
        return;
    }
    result.deleted += 1;
}

fn apply_archive(
    database: &Database,
    memory_id: &str,
    performed_by: &str,
    timestamp: &str,
    result: &mut ApplyResult,
) {
    if let Err(error) = database.set_memory_indexed(memory_id, false) {
        result
            .errors
            .push(format!("archive {memory_id}: {error}"));
        return;
    }
    if let Err(error) = database.log_consolidation(
        &generate_log_id(
            timestamp,
            result.archived + result.deleted + result.merged,
        ),
        "archive",
        &[memory_id.to_string()],
        Some("stale: low score with no usage"),
        performed_by,
        timestamp,
    ) {
        result.errors.push(format!("log archive: {error}"));
        return;
    }
    result.archived += 1;
}

fn current_utc_timestamp() -> String {
    let duration = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let seconds = duration.as_secs();
    let (year, month, day, hour, minute, second) = unix_to_utc_components(seconds);
    format!(
        "{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z"
    )
}

fn unix_to_utc_components(timestamp: u64) -> (u64, u64, u64, u64, u64, u64) {
    let second = timestamp % 60;
    let minute = (timestamp / 60) % 60;
    let hour = (timestamp / 3600) % 24;
    let mut days = timestamp / 86400;
    let mut year = 1970;
    loop {
        let days_in_year = if is_leap_year(year) { 366 } else { 365 };
        if days < days_in_year {
            break;
        }
        days -= days_in_year;
        year += 1;
    }
    let month_days = month_lengths(is_leap_year(year));
    let mut month = 0;
    for (index, &length) in month_days.iter().enumerate() {
        if days < length {
            month = index as u64 + 1;
            break;
        }
        days -= length;
    }
    let day = days + 1;
    (year, month, day, hour, minute, second)
}

fn is_leap_year(year: u64) -> bool {
    (year.is_multiple_of(4) && !year.is_multiple_of(100)) || year.is_multiple_of(400)
}

fn month_lengths(leap: bool) -> [u64; 12] {
    let feb = if leap { 29 } else { 28 };
    [31, feb, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
}

fn generate_log_id(timestamp: &str, sequence: usize) -> String {
    format!("consol-{timestamp}-{sequence}")
}
