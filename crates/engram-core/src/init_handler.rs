use std::fs;
use std::path::{Path, PathBuf};

use engram_storage::Database;

use crate::config::home_directory;
use crate::error::CoreError;

const ENGRAM_DIRECTORY: &str = ".engram";
const CONFIG_FILENAME: &str = "engram.toml";
const AGENT_MD_FILENAME: &str = "AGENT.md";
const DATABASE_FILENAME: &str = "engram.db";
const SOCKET_FILENAME: &str = "engram.sock";
const MCP_JSON_FILENAME: &str = ".mcp.json";
const GITIGNORE_FILENAME: &str = ".gitignore";
const GITIGNORE_MARKER: &str = ".engram/";
const AGENT_MD_CONTENT: &str = include_str!("../AGENT.md");
const UNIX_SOCKET_PATH_MAX_BYTES: usize = 104;
const MCP_NPX_PACKAGE: &str = "@engramm/engram-mcp-server";
const DEFAULT_ENGRAM_BIN_NAME: &str = "engram";

const DEFAULT_CONFIG_TEMPLATE: &str = r#"# Global engram config. Runtime always prefers per-project state under
# <project>/.engram/{engram.db, engram.sock} (discovered by walking up from cwd
# like .git). The database.path and server.socket_path values below are
# fallbacks used only when no .engram/ marker is found and no ENGRAM_DB_PATH /
# ENGRAM_SOCKET_PATH env override is set.

[database]
path = "~/.engram/memories.db"

[embedding]
provider = "voyage"
model = "voyage-code-3"
dimension = 1024
# hyde_threshold: opt-in HyDE. 0 = disabled (default). N>0 = enable HyDE
# when the query has fewer than N words. HyDE adds ~1.5s latency on cache
# miss but improves recall on terse queries; cache is keyed by the
# original query so repeated calls are instant.
hyde_threshold = 0

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

#[derive(Debug)]
pub enum McpCommand {
    Node { script_path: PathBuf },
    Npx,
}

pub fn execute() -> Result<(), CoreError> {
    let cwd = std::env::current_dir()
        .map_err(|error| CoreError::InitFailed(format!("cwd unavailable: {error}")))?;
    let home = home_directory()
        .ok_or_else(|| CoreError::InitFailed("HOME environment variable not set".into()))?;
    try_interactive_wizard();
    execute_with_dirs(&cwd, Path::new(&home))
}

pub fn execute_with_dirs(project_dir: &Path, home_dir: &Path) -> Result<(), CoreError> {
    validate_socket_path(project_dir)?;

    let home_config_path = home_dir.join(ENGRAM_DIRECTORY).join(CONFIG_FILENAME);
    let mcp_command = resolve_mcp_server_command();

    if home_config_path.exists() {
        create_engram_directory(project_dir)?;
        initialize_database(project_dir)?;
        write_agent_instructions(project_dir)?;
        write_mcp_json(project_dir, &mcp_command)?;
        write_gitignore(project_dir)?;
        return Ok(());
    }

    create_engram_directory(project_dir)?;
    write_default_config_to_home(home_dir)?;
    write_mcp_json(project_dir, &mcp_command)?;
    initialize_database(project_dir)?;
    write_agent_instructions(project_dir)?;
    write_gitignore(project_dir)?;
    print_completion_info(project_dir);
    Ok(())
}

fn try_interactive_wizard() {
    if !std::io::IsTerminal::is_terminal(&std::io::stdin()) {
        return;
    }
    let _ = std::process::Command::new("engram-tui")
        .arg("init")
        .status();
}

fn validate_socket_path(project_dir: &Path) -> Result<(), CoreError> {
    let socket_path = project_dir.join(ENGRAM_DIRECTORY).join(SOCKET_FILENAME);
    let length = socket_path.as_os_str().as_encoded_bytes().len();
    if length >= UNIX_SOCKET_PATH_MAX_BYTES {
        return Err(CoreError::InitFailed(format!(
            "project path too deep: socket path '{}' is {length} bytes, max {} (UNIX limit)",
            socket_path.display(),
            UNIX_SOCKET_PATH_MAX_BYTES - 1
        )));
    }
    Ok(())
}

fn create_engram_directory(project_dir: &Path) -> Result<(), CoreError> {
    let target = project_dir.join(ENGRAM_DIRECTORY);
    fs::create_dir_all(&target).map_err(|error| {
        CoreError::InitFailed(format!("failed to create {}: {error}", target.display()))
    })
}

fn write_default_config_to_home(home_dir: &Path) -> Result<(), CoreError> {
    let home_engram = home_dir.join(ENGRAM_DIRECTORY);
    fs::create_dir_all(&home_engram).map_err(|error| {
        CoreError::InitFailed(format!(
            "failed to create {}: {error}",
            home_engram.display()
        ))
    })?;
    let config_path = home_engram.join(CONFIG_FILENAME);
    if config_path.exists() {
        return Ok(());
    }
    fs::write(&config_path, DEFAULT_CONFIG_TEMPLATE)
        .map_err(|error| CoreError::InitFailed(format!("failed to write config: {error}")))
}

fn initialize_database(project_dir: &Path) -> Result<(), CoreError> {
    let engram_dir = project_dir.join(ENGRAM_DIRECTORY);
    fs::create_dir_all(&engram_dir).map_err(|error| {
        CoreError::InitFailed(format!("failed to create database directory: {error}"))
    })?;
    let database_path = engram_dir.join(DATABASE_FILENAME);
    let database_path_str = database_path
        .to_str()
        .ok_or_else(|| CoreError::InitFailed("database path is not valid utf-8".into()))?;
    let _database = Database::open(database_path_str)?;
    println!("Database initialized at {database_path_str}");
    Ok(())
}

fn write_agent_instructions(project_dir: &Path) -> Result<(), CoreError> {
    let agent_md_path = project_dir.join(ENGRAM_DIRECTORY).join(AGENT_MD_FILENAME);
    fs::write(&agent_md_path, AGENT_MD_CONTENT)
        .map_err(|error| CoreError::InitFailed(format!("failed to write AGENT.md: {error}")))?;
    println!("Agent instructions written to {}", agent_md_path.display());
    Ok(())
}

fn resolve_mcp_server_command() -> McpCommand {
    if let Ok(env_path) = std::env::var("ENGRAM_MCP_SERVER_PATH") {
        let path = PathBuf::from(&env_path);
        if path.exists() {
            return McpCommand::Node { script_path: path };
        }
    }
    if let Some(bin_path) = resolve_engram_bin()
        && let Some(bin_dir) = bin_path.parent()
    {
        let candidate = bin_dir
            .join("..")
            .join("mcp-server")
            .join("dist")
            .join("index.js");
        if candidate.exists() {
            if let Ok(canonical) = candidate.canonicalize() {
                return McpCommand::Node {
                    script_path: canonical,
                };
            }
            return McpCommand::Node {
                script_path: candidate,
            };
        }
    }
    eprintln!(
        "warning: could not locate bundled engram mcp-server script; falling back to 'npx {MCP_NPX_PACKAGE}'. \
         Set ENGRAM_MCP_SERVER_PATH to an absolute path of mcp-server/dist/index.js to silence this warning."
    );
    McpCommand::Npx
}

fn resolve_engram_bin() -> Option<PathBuf> {
    if let Ok(value) = std::env::var("ENGRAM_BIN") {
        let candidate = PathBuf::from(value);
        if candidate.exists() {
            return Some(candidate);
        }
    }
    let output = std::process::Command::new("which")
        .arg(DEFAULT_ENGRAM_BIN_NAME)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8(output.stdout).ok()?;
    let trimmed = stdout.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(PathBuf::from(trimmed))
}

fn write_mcp_json(project_dir: &Path, command: &McpCommand) -> Result<(), CoreError> {
    let (command_name, args): (&str, Vec<String>) = match command {
        McpCommand::Node { script_path } => {
            let path_str = script_path
                .to_str()
                .ok_or_else(|| CoreError::InitFailed("mcp script path is not valid utf-8".into()))?
                .to_string();
            ("node", vec![path_str])
        }
        McpCommand::Npx => ("npx", vec![MCP_NPX_PACKAGE.to_string()]),
    };
    let engram_bin = resolve_engram_bin();
    let engram_bin_string = engram_bin.as_ref().and_then(|path| {
        let canonical = path.canonicalize().unwrap_or_else(|_| path.clone());
        canonical.to_str().map(|value| value.to_string())
    });
    let mut server_entry = serde_json::json!({
        "command": command_name,
        "args": args,
    });
    if let Some(bin_string) = engram_bin_string.as_ref() {
        server_entry["env"] = serde_json::json!({ "ENGRAM_BIN": bin_string });
    } else {
        eprintln!(
            "warning: could not resolve engram binary; MCP config has no ENGRAM_BIN env — set it manually before starting Claude Code"
        );
    }
    let document = serde_json::json!({
        "mcpServers": {
            "engram": server_entry,
        }
    });
    let serialized = serde_json::to_string_pretty(&document)
        .map_err(|error| CoreError::InitFailed(format!("failed to serialize mcp json: {error}")))?;
    let mcp_path = project_dir.join(MCP_JSON_FILENAME);
    fs::write(&mcp_path, serialized)
        .map_err(|error| CoreError::InitFailed(format!("failed to write .mcp.json: {error}")))?;
    println!("MCP config written to {}", mcp_path.display());
    Ok(())
}

fn write_gitignore(project_dir: &Path) -> Result<(), CoreError> {
    let gitignore_path = project_dir.join(GITIGNORE_FILENAME);
    let existing = match fs::read_to_string(&gitignore_path) {
        Ok(content) => content,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(error) => {
            return Err(CoreError::InitFailed(format!(
                "failed to read .gitignore: {error}"
            )));
        }
    };
    if gitignore_contains_marker(&existing) {
        return Ok(());
    }
    let mut updated = existing;
    if !updated.is_empty() && !updated.ends_with('\n') {
        updated.push('\n');
    }
    updated.push_str(GITIGNORE_MARKER);
    updated.push('\n');
    fs::write(&gitignore_path, updated)
        .map_err(|error| CoreError::InitFailed(format!("failed to write .gitignore: {error}")))
}

fn gitignore_contains_marker(content: &str) -> bool {
    content.lines().any(|line| {
        line.trim() == GITIGNORE_MARKER.trim_end_matches('/') || line.trim() == GITIGNORE_MARKER
    })
}

fn print_completion_info(project_dir: &Path) {
    println!();
    println!("Engram initialized at {}", project_dir.display());
    println!();
    println!("Set API keys via environment variables:");
    println!("  export ENGRAM_VOYAGE_API_KEY=your-voyage-key");
    println!("  export ENGRAM_OPENAI_API_KEY=your-openai-key");
    println!();
    println!("MCP config written to .mcp.json; Claude Code will pick it up automatically.");
    println!("Add to your project's CLAUDE.md:");
    println!("  Engram memory system: .engram/AGENT.md");
}
