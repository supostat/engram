use serde_json::Value;

#[derive(Clone, clap::ValueEnum)]
pub enum OutputFormat {
    Json,
    Text,
    Jsonl,
}

pub fn format_output(value: &Value, format: &OutputFormat) -> String {
    match format {
        OutputFormat::Json => format_as_json(value),
        OutputFormat::Text => format_as_text(value),
        OutputFormat::Jsonl => format_as_jsonl(value),
    }
}

fn format_as_json(value: &Value) -> String {
    serde_json::to_string_pretty(value).unwrap_or_default()
}

fn format_as_jsonl(value: &Value) -> String {
    if let Some((results, degraded)) = search_envelope(value) {
        let mut lines: Vec<String> = results
            .iter()
            .map(|item| serde_json::to_string(item).unwrap_or_default())
            .collect();
        lines.push(serde_json::to_string(&json_degraded(degraded)).unwrap_or_default());
        return lines.join("\n");
    }
    match value {
        Value::Array(items) => items
            .iter()
            .map(|item| serde_json::to_string(item).unwrap_or_default())
            .collect::<Vec<_>>()
            .join("\n"),
        _ => serde_json::to_string(value).unwrap_or_default(),
    }
}

fn format_as_text(value: &Value) -> String {
    if let Some((results, degraded)) = search_envelope(value) {
        let degraded_line = format!("degraded: {degraded}");
        if results.is_empty() {
            return degraded_line;
        }
        return format!("{}\n{degraded_line}", format_array_as_text(results));
    }
    match value {
        Value::Object(map) => format_object_as_text(map),
        Value::Array(items) => format_array_as_text(items),
        Value::String(string) => string.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

/// A `memory_search` response is an object carrying a `results` array alongside
/// a `degraded` flag. The jsonl/text formatters fan out over the inner array so
/// each hit renders as its own line/block; other object responses (status,
/// judge, config) lack the `results` array and fall through to the generic
/// object rendering unchanged.
fn search_envelope(value: &Value) -> Option<(&[Value], bool)> {
    let map = value.as_object()?;
    let results = map.get("results")?.as_array()?;
    let degraded = map.get("degraded")?.as_bool()?;
    Some((results, degraded))
}

fn json_degraded(degraded: bool) -> Value {
    serde_json::json!({ "degraded": degraded })
}

fn format_object_as_text(map: &serde_json::Map<String, Value>) -> String {
    map.iter()
        .map(|(key, value)| format!("{key}: {}", scalar_to_text(value)))
        .collect::<Vec<_>>()
        .join("\n")
}

fn format_array_as_text(items: &[Value]) -> String {
    items
        .iter()
        .map(format_as_text)
        .collect::<Vec<_>>()
        .join("\n---\n")
}

fn scalar_to_text(value: &Value) -> String {
    match value {
        Value::String(string) => string.clone(),
        Value::Null => String::new(),
        Value::Array(_) | Value::Object(_) => serde_json::to_string(value).unwrap_or_default(),
        other => other.to_string(),
    }
}
