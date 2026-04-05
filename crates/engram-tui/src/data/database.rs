use std::io;
use std::path::Path;

use rusqlite::{Connection, OpenFlags};

pub struct MemorySummary {
    pub memory_type: String,
    pub context: String,
    pub action: String,
    pub result: String,
    pub score: f64,
    pub project: Option<String>,
    pub created_at: String,
}

impl MemorySummary {
    pub fn project_display(&self) -> String {
        self.project
            .clone()
            .unwrap_or_else(|| "(none)".to_string())
    }
}

pub struct QTableEntry {
    pub router_level: i32,
    pub state: String,
    pub action: String,
    pub value: f64,
    pub update_count: i64,
}

pub struct ModelInfo {
    pub filename: String,
    pub size_bytes: u64,
    pub modified: String,
}

pub struct DatabaseReader {
    connection: Connection,
}

impl DatabaseReader {
    pub fn new(path: &str) -> io::Result<Self> {
        let connection = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)
            .map_err(io::Error::other)?;
        Ok(Self { connection })
    }

    pub fn memory_count(&self) -> usize {
        self.query_count("SELECT COUNT(*) FROM memories")
    }

    pub fn indexed_count(&self) -> usize {
        self.query_count("SELECT COUNT(*) FROM memories WHERE indexed = TRUE")
    }

    pub fn average_score(&self) -> f64 {
        self.connection
            .query_row("SELECT COALESCE(AVG(score), 0.0) FROM memories", [], |row| {
                row.get(0)
            })
            .unwrap_or(0.0)
    }

    pub fn type_distribution(&self) -> Vec<(String, usize)> {
        self.query_distribution(
            "SELECT memory_type, COUNT(*) FROM memories \
             GROUP BY memory_type ORDER BY COUNT(*) DESC",
        )
    }

    pub fn project_distribution(&self) -> Vec<(String, usize)> {
        self.query_distribution(
            "SELECT COALESCE(project, '(none)'), COUNT(*) FROM memories \
             GROUP BY project ORDER BY COUNT(*) DESC",
        )
    }

    pub fn score_distribution(&self) -> Vec<usize> {
        let mut buckets = vec![0_usize; 10];
        let query = "SELECT score FROM memories";
        let Ok(mut statement) = self.connection.prepare(query) else {
            return buckets;
        };
        let Ok(rows) = statement.query_map([], |row| row.get::<_, f64>(0)) else {
            return buckets;
        };
        for score_result in rows.flatten() {
            buckets[bucket_index(score_result)] += 1;
        }
        buckets
    }

    pub fn feedback_stats(&self) -> (usize, usize) {
        let searched = self.query_count("SELECT COUNT(*) FROM feedback_tracking");
        let judged = self.query_count(
            "SELECT COUNT(*) FROM feedback_tracking WHERE judged = TRUE",
        );
        (searched, judged)
    }

    pub fn recent_memories(&self, limit: usize) -> Vec<MemorySummary> {
        let query = "SELECT memory_type, context, COALESCE(action, ''), COALESCE(result, ''), \
                     score, COALESCE(project, ''), created_at \
                     FROM memories ORDER BY created_at DESC LIMIT ?1";
        let Ok(mut statement) = self.connection.prepare(query) else {
            return Vec::new();
        };
        let Ok(rows) = statement.query_map([limit as i64], |row| {
            Ok(MemorySummary {
                memory_type: row.get(0)?,
                context: truncate_context(row.get::<_, String>(1)?),
                action: row.get(2)?,
                result: row.get(3)?,
                score: row.get(4)?,
                project: non_empty_string(row.get::<_, String>(5)?),
                created_at: row.get(6)?,
            })
        }) else {
            return Vec::new();
        };
        rows.flatten().collect()
    }

    pub fn list_memories(&self, limit: usize) -> Vec<MemorySummary> {
        let query = "SELECT memory_type, context, COALESCE(action, ''), COALESCE(result, ''), \
                     score, COALESCE(project, ''), created_at \
                     FROM memories ORDER BY created_at DESC LIMIT ?1";
        let Ok(mut statement) = self.connection.prepare(query) else {
            return Vec::new();
        };
        let Ok(rows) = statement.query_map([limit as i64], |row| {
            Ok(MemorySummary {
                memory_type: row.get(0)?,
                context: row.get(1)?,
                action: row.get(2)?,
                result: row.get(3)?,
                score: row.get(4)?,
                project: non_empty_string(row.get::<_, String>(5)?),
                created_at: row.get(6)?,
            })
        }) else {
            return Vec::new();
        };
        rows.flatten().collect()
    }

    pub fn q_table_entries(&self) -> Vec<QTableEntry> {
        let query = "SELECT router_level, state, action, value, update_count \
                     FROM q_table ORDER BY router_level, state, action";
        let Ok(mut statement) = self.connection.prepare(query) else {
            return Vec::new();
        };
        let Ok(rows) = statement.query_map([], |row| {
            Ok(QTableEntry {
                router_level: row.get(0)?,
                state: row.get(1)?,
                action: row.get(2)?,
                value: row.get(3)?,
                update_count: row.get(4)?,
            })
        }) else {
            return Vec::new();
        };
        rows.flatten().collect()
    }

    pub fn models_info(&self, models_path: &str) -> Vec<ModelInfo> {
        let path = Path::new(models_path);
        let Ok(entries) = std::fs::read_dir(path) else {
            return Vec::new();
        };
        let mut models: Vec<ModelInfo> = entries
            .flatten()
            .filter(|entry| {
                entry
                    .path()
                    .extension()
                    .is_some_and(|ext| ext == "onnx")
            })
            .filter_map(|entry| model_info_from_entry(&entry))
            .collect();
        models.sort_by(|first, second| first.filename.cmp(&second.filename));
        models
    }

    fn query_count(&self, query: &str) -> usize {
        self.connection
            .query_row(query, [], |row| row.get::<_, i64>(0))
            .unwrap_or(0) as usize
    }

    fn query_distribution(&self, query: &str) -> Vec<(String, usize)> {
        let Ok(mut statement) = self.connection.prepare(query) else {
            return Vec::new();
        };
        let Ok(rows) = statement.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)? as usize))
        }) else {
            return Vec::new();
        };
        rows.flatten().collect()
    }
}

fn truncate_context(text: String) -> String {
    if text.len() <= 80 {
        return text;
    }
    let truncated: String = text.chars().take(77).collect();
    format!("{truncated}...")
}

fn non_empty_string(value: String) -> Option<String> {
    if value.is_empty() { None } else { Some(value) }
}

fn bucket_index(score: f64) -> usize {
    let index = (score * 10.0) as usize;
    index.min(9)
}

fn model_info_from_entry(entry: &std::fs::DirEntry) -> Option<ModelInfo> {
    let metadata = entry.metadata().ok()?;
    let modified = metadata
        .modified()
        .ok()
        .and_then(|time| {
            let duration = time.duration_since(std::time::UNIX_EPOCH).ok()?;
            Some(format_unix_timestamp(duration.as_secs()))
        })
        .unwrap_or_default();
    Some(ModelInfo {
        filename: entry.file_name().to_string_lossy().into_owned(),
        size_bytes: metadata.len(),
        modified,
    })
}

fn format_unix_timestamp(seconds: u64) -> String {
    let days = seconds / 86400;
    let years = (days as f64 / 365.25) as u64;
    let year = 1970 + years;
    let remaining_days = days - (years as f64 * 365.25) as u64;
    let month = remaining_days / 30 + 1;
    let day = remaining_days % 30 + 1;
    format!("{year}-{month:02}-{day:02}")
}
