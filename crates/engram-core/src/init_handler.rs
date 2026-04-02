use std::fs;
use std::path::Path;

use engram_storage::Database;

use crate::config::{expand_tilde, home_directory};
use crate::error::CoreError;

const ENGRAM_DIRECTORY: &str = ".engram";
const CONFIG_FILENAME: &str = "engram.toml";
const DATABASE_RELATIVE_PATH: &str = ".engram/memories.db";

const DEFAULT_CONFIG_TEMPLATE: &str = r#"[database]
path = "~/.engram/memories.db"

[embedding]
provider = "voyage"
model = "voyage-code-3"
dimension = 1024

[llm]
provider = "openai"
model = "gpt-4o-mini"

[server]
socket_path = "~/.engram/engram.sock"
reindex_interval_secs = 3600

[hnsw]
max_connections = 16
ef_construction = 200
ef_search = 40
dimension = 1024

[consolidation]
stale_days = 90
min_score = 0.3
"#;

pub fn execute() -> Result<(), CoreError> {
    if config_already_exists() {
        println!("engram.toml already exists, skipping initialization.");
        return Ok(());
    }
    let engram_directory = resolve_engram_directory()?;
    create_engram_directory(&engram_directory)?;
    write_default_config(&engram_directory)?;
    initialize_database(&engram_directory)?;
    print_mcp_snippets();
    Ok(())
}

fn config_already_exists() -> bool {
    if Path::new(CONFIG_FILENAME).exists() {
        return true;
    }
    if let Some(home) = home_directory() {
        let home_config = Path::new(&home)
            .join(ENGRAM_DIRECTORY)
            .join(CONFIG_FILENAME);
        return home_config.exists();
    }
    false
}

fn resolve_engram_directory() -> Result<String, CoreError> {
    let home = home_directory()
        .ok_or_else(|| CoreError::InitFailed("HOME environment variable not set".into()))?;
    Ok(format!("{home}/{ENGRAM_DIRECTORY}"))
}

fn create_engram_directory(path: &str) -> Result<(), CoreError> {
    fs::create_dir_all(path)
        .map_err(|error| CoreError::InitFailed(format!("failed to create {path}: {error}")))
}

fn write_default_config(engram_directory: &str) -> Result<(), CoreError> {
    let config_path = Path::new(engram_directory).join(CONFIG_FILENAME);
    fs::write(&config_path, DEFAULT_CONFIG_TEMPLATE)
        .map_err(|error| CoreError::InitFailed(format!("failed to write config: {error}")))
}

fn initialize_database(engram_directory: &str) -> Result<(), CoreError> {
    let database_path = expand_tilde(&format!("~/{DATABASE_RELATIVE_PATH}"));
    let database_directory = Path::new(&database_path)
        .parent()
        .unwrap_or(Path::new(engram_directory));
    fs::create_dir_all(database_directory).map_err(|error| {
        CoreError::InitFailed(format!("failed to create database directory: {error}"))
    })?;
    let _database = Database::open(&database_path)?;
    println!("Database initialized at {database_path}");
    Ok(())
}

fn print_mcp_snippets() {
    println!();
    println!("Claude Desktop — add to claude_desktop_config.json:");
    println!(
        r#"{{
  "mcpServers": {{
    "engram": {{
      "command": "npx",
      "args": ["@engram/mcp-server"]
    }}
  }}
}}"#
    );
    println!();
    println!("Claude Code — add to settings:");
    println!(
        r#"{{
  "mcpServers": {{
    "engram": {{
      "command": "npx",
      "args": ["@engram/mcp-server"]
    }}
  }}
}}"#
    );
    println!();
    println!("Set API keys via environment variables:");
    println!("  export ENGRAM_VOYAGE_API_KEY=your-voyage-key");
    println!("  export ENGRAM_OPENAI_API_KEY=your-openai-key");
}
