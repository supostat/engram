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
    match value {
        Value::Object(map) => format_object_as_text(map),
        Value::Array(items) => format_array_as_text(items),
        Value::String(string) => string.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
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
