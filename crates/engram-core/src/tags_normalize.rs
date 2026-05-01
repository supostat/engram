//! Wire→storage normalization for memories.tags.
//! Canonical on-disk form: JSON-array string (per ADR 2026-05-01).

use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum TagsInput {
    Array(Vec<String>),
    Encoded(String),
}

pub fn normalize_tags(input: Option<TagsInput>) -> Option<String> {
    match input {
        None => None,
        Some(TagsInput::Array(values)) if values.is_empty() => None,
        Some(TagsInput::Array(values)) => {
            let filtered: Vec<String> = values
                .into_iter()
                .filter(|tag| !tag.trim().is_empty())
                .collect();
            if filtered.is_empty() {
                return None;
            }
            Some(serialize_tags(&filtered))
        }
        Some(TagsInput::Encoded(raw)) if raw.trim().is_empty() => None,
        Some(TagsInput::Encoded(raw)) => normalize_encoded(&raw),
    }
}

/// Normalizes encoded tag strings (JSON-array, CSV, or naked single-token).
///
/// JSON-array is the canonical on-disk form (ADR 2026-05-01). CSV and naked
/// forms are accepted for backward compatibility but are deprecated and will
/// be rejected at the next release; both branches emit a stderr warning
/// (grep marker: `tags_normalize: deprecated`).
///
/// Returns `None` when the input contains no non-empty tokens after
/// normalization, so the empty case is consistent regardless of wire shape.
fn normalize_encoded(raw: &str) -> Option<String> {
    if let Ok(parsed) = serde_json::from_str::<Vec<String>>(raw) {
        let filtered: Vec<String> = parsed
            .into_iter()
            .filter(|tag| !tag.trim().is_empty())
            .collect();
        if filtered.is_empty() {
            return None;
        }
        return Some(serialize_tags(&filtered));
    }
    if raw.contains(',') {
        eprintln!("tags_normalize: deprecated CSV form on wire (will be rejected at next release)");
        let parts: Vec<String> = raw
            .split(',')
            .map(|tag| tag.trim().to_string())
            .filter(|tag| !tag.is_empty())
            .collect();
        if parts.is_empty() {
            return None;
        }
        return Some(serialize_tags(&parts));
    }
    let single = raw.trim();
    if single.is_empty() {
        return None;
    }
    eprintln!("tags_normalize: deprecated naked form on wire (will be rejected at next release)");
    Some(serialize_tags(&[single.to_string()]))
}

fn serialize_tags(tags: &[String]) -> String {
    serde_json::to_string(tags).expect("Vec<String> serializes to JSON")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn none_input_returns_none() {
        assert_eq!(normalize_tags(None), None);
    }

    #[test]
    fn empty_array_returns_none() {
        assert_eq!(normalize_tags(Some(TagsInput::Array(vec![]))), None);
    }

    #[test]
    fn array_with_values_serializes_to_json() {
        let input = Some(TagsInput::Array(vec![
            "rust".to_string(),
            "bugfix".to_string(),
        ]));
        assert_eq!(
            normalize_tags(input).as_deref(),
            Some(r#"["rust","bugfix"]"#)
        );
    }

    #[test]
    fn empty_encoded_string_returns_none() {
        assert_eq!(
            normalize_tags(Some(TagsInput::Encoded(String::new()))),
            None
        );
    }

    #[test]
    fn whitespace_only_encoded_returns_none() {
        assert_eq!(
            normalize_tags(Some(TagsInput::Encoded("   \t  ".into()))),
            None
        );
    }

    #[test]
    fn encoded_json_array_passes_through() {
        let input = Some(TagsInput::Encoded(r#"["rust","bugfix"]"#.into()));
        assert_eq!(
            normalize_tags(input).as_deref(),
            Some(r#"["rust","bugfix"]"#)
        );
    }

    #[test]
    fn encoded_json_array_filters_empty_elements() {
        let input = Some(TagsInput::Encoded(r#"["rust","","bugfix"]"#.into()));
        assert_eq!(
            normalize_tags(input).as_deref(),
            Some(r#"["rust","bugfix"]"#)
        );
    }

    #[test]
    fn encoded_csv_is_split_into_array() {
        let input = Some(TagsInput::Encoded("rust,bugfix".into()));
        assert_eq!(
            normalize_tags(input).as_deref(),
            Some(r#"["rust","bugfix"]"#)
        );
    }

    #[test]
    fn encoded_csv_trims_whitespace_and_skips_empty() {
        let input = Some(TagsInput::Encoded(" rust , , bugfix ,".into()));
        assert_eq!(
            normalize_tags(input).as_deref(),
            Some(r#"["rust","bugfix"]"#)
        );
    }

    #[test]
    fn encoded_naked_single_token_wraps_into_array() {
        let input = Some(TagsInput::Encoded("rust".into()));
        assert_eq!(normalize_tags(input).as_deref(), Some(r#"["rust"]"#));
    }

    #[test]
    fn encoded_naked_token_is_trimmed() {
        let input = Some(TagsInput::Encoded("  rust  ".into()));
        assert_eq!(normalize_tags(input).as_deref(), Some(r#"["rust"]"#));
    }

    #[test]
    fn encoded_empty_json_array_returns_none() {
        assert_eq!(normalize_tags(Some(TagsInput::Encoded("[]".into()))), None);
    }

    #[test]
    fn encoded_json_array_with_only_empty_strings_returns_none() {
        assert_eq!(
            normalize_tags(Some(TagsInput::Encoded(r#"["", "  "]"#.into()))),
            None
        );
    }

    #[test]
    fn array_with_only_empty_strings_returns_none() {
        assert_eq!(
            normalize_tags(Some(TagsInput::Array(vec!["".into(), "  ".into()]))),
            None
        );
    }

    #[test]
    fn encoded_csv_with_only_empty_segments_returns_none() {
        assert_eq!(normalize_tags(Some(TagsInput::Encoded(",,,".into()))), None);
    }
}
