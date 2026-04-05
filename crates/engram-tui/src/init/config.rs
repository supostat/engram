use std::fs;
use std::io;

use super::wizard::{EMBEDDING_OPTIONS, InitWizard, LLM_OPTIONS};

pub const MCP_JSON_SNIPPET: &str =
    r#"{ "mcpServers": { "engram": { "command": "engram-mcp" } } }"#;

impl InitWizard {
    pub(super) fn create_config_files(&self) -> io::Result<()> {
        let home = dirs::home_dir().ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, "HOME directory not found")
        })?;
        let engram_directory = home.join(".engram");
        fs::create_dir_all(&engram_directory)?;

        let toml_content = self.build_toml();
        let config_path = engram_directory.join("engram.toml");
        fs::write(&config_path, toml_content)?;
        Ok(())
    }

    fn build_toml(&self) -> String {
        let embedding_name = EMBEDDING_OPTIONS[self.embedding_provider];
        let llm_name = LLM_OPTIONS[self.llm_provider];

        let embedding_model = if embedding_name == "voyage" {
            "voyage-code-3"
        } else {
            "deterministic"
        };
        let embedding_dimension: u32 = if embedding_name == "voyage" { 1024 } else { 128 };

        let llm_model = match llm_name {
            "openai" => "gpt-4o-mini",
            "local" => "local",
            _ => "none",
        };

        let mut toml = format!(
            r#"[database]
path = "{database_path}"

[embedding]
provider = "{embedding_name}"
model = "{embedding_model}"
dimension = {embedding_dimension}

[llm]
provider = "{llm_name}"
model = "{llm_model}"

[server]
socket_path = "~/.engram/engram.sock"
reindex_interval_secs = 3600

[hnsw]
max_connections = 16
ef_construction = 200
ef_search = 40
dimension = {embedding_dimension}

[consolidation]
stale_days = 90
min_score = 0.3
"#,
            database_path = self.database_path,
        );

        if embedding_name == "voyage" && !self.embedding_api_key.is_empty() {
            toml.push_str(&format!(
                "\n[secrets]\nvoyage_api_key = \"{}\"\n",
                self.embedding_api_key
            ));
        }
        if llm_name == "openai" && !self.llm_api_key.is_empty() {
            if self.embedding_api_key.is_empty() || embedding_name != "voyage" {
                toml.push_str("\n[secrets]\n");
            }
            toml.push_str(&format!(
                "openai_api_key = \"{}\"\n",
                self.llm_api_key
            ));
        }
        toml
    }
}
