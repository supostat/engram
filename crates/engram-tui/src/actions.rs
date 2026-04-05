use std::fs;
use std::io;
use std::path::PathBuf;

use serde_json::json;

use crate::data::{DatabaseReader, SocketClient};
use crate::overlays::StatusMessage;

pub fn judge_memory(
    socket: &mut Option<SocketClient>,
    memory_id: &str,
) -> StatusMessage {
    let Some(client) = socket.as_mut() else {
        return StatusMessage::error("Server offline".to_string());
    };
    let params = json!({"memory_id": memory_id});
    match client.call("memory_judge", params) {
        Ok(data) => {
            let score = data["score"].as_f64().unwrap_or(0.0);
            StatusMessage::info(format!("Judged: score {score:.2}"))
        }
        Err(error) => StatusMessage::error(format!("Judge failed: {error}")),
    }
}

pub fn delete_memory(
    database: &DatabaseReader,
    database_path: &str,
    memory_id: &str,
) -> StatusMessage {
    match database.delete_memory(database_path, memory_id) {
        Ok(()) => StatusMessage::info(format!("Deleted memory {}", truncate_id(memory_id))),
        Err(error) => StatusMessage::error(format!("Delete failed: {error}")),
    }
}

pub fn export_memories(socket: &mut Option<SocketClient>) -> StatusMessage {
    let Some(client) = socket.as_mut() else {
        return StatusMessage::error("Server offline".to_string());
    };
    match client.call("memory_export", json!({})) {
        Ok(data) => match save_export(data) {
            Ok(path) => StatusMessage::info(format!("Exported to {path}")),
            Err(error) => StatusMessage::error(format!("Save failed: {error}")),
        },
        Err(error) => StatusMessage::error(format!("Export failed: {error}")),
    }
}

pub fn consolidation_preview(socket: &mut Option<SocketClient>) -> Option<String> {
    let client = socket.as_mut()?;
    let params = json!({});
    match client.call("memory_consolidate_preview", params) {
        Ok(data) => {
            let duplicates = data["duplicates"].as_u64().unwrap_or(0);
            let stale = data["stale"].as_u64().unwrap_or(0);
            let garbage = data["garbage"].as_u64().unwrap_or(0);
            let total = data["total_candidates"].as_u64().unwrap_or(0);
            Some(format!(
                "Consolidation Preview\n\n\
                 Duplicates: {duplicates}\n\
                 Stale:      {stale}\n\
                 Garbage:    {garbage}\n\
                 Total:      {total}"
            ))
        }
        Err(error) => Some(format!("Error: {error}")),
    }
}

fn save_export(data: serde_json::Value) -> io::Result<String> {
    let home = dirs::home_dir().ok_or_else(|| io::Error::other("cannot determine home dir"))?;
    let engram_dir = home.join(".engram");
    fs::create_dir_all(&engram_dir)?;
    let timestamp = chrono_like_timestamp();
    let filename = format!("backup_{timestamp}.json");
    let path: PathBuf = engram_dir.join(&filename);
    let content = serde_json::to_string_pretty(&data).map_err(io::Error::other)?;
    fs::write(&path, content)?;
    Ok(path.to_string_lossy().into_owned())
}

fn chrono_like_timestamp() -> String {
    use std::time::SystemTime;
    let seconds = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .expect("system clock before UNIX epoch")
        .as_secs();
    format!("{seconds}")
}

fn truncate_id(id: &str) -> &str {
    if id.len() > 12 {
        &id[..12]
    } else {
        id
    }
}
