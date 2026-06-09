use std::fs;
use std::io;

use super::wizard::{EMBEDDING_OPTIONS, InitWizard, LLM_OPTIONS};

pub const MCP_JSON_SNIPPET: &str = r#"{ "mcpServers": { "engram": { "command": "engram-mcp" } } }"#;

impl InitWizard {
    pub(super) fn create_config_files(&self) -> io::Result<()> {
        let home = dirs::home_dir()
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "HOME directory not found"))?;
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

        let (embedding_model, embedding_dimension): (&str, u32) = match embedding_name {
            "voyage" => ("voyage-4", 1024),
            "ollama" => ("qwen3-embedding:0.6b", 1024),
            _ => ("deterministic", 128),
        };

        let llm_model = match llm_name {
            "openai" => "gpt-4o-mini",
            "ollama" => "qwen3:4b",
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
            toml.push_str(&format!("openai_api_key = \"{}\"\n", self.llm_api_key));
        }
        toml
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const OLLAMA_OPTION_INDEX: usize = 1;

    #[test]
    fn build_toml_emits_ollama_models_for_embedding_and_llm() {
        let mut wizard = InitWizard::new();
        wizard.embedding_provider = OLLAMA_OPTION_INDEX;
        wizard.llm_provider = OLLAMA_OPTION_INDEX;

        let toml = wizard.build_toml();

        assert!(toml.contains("provider = \"ollama\""));
        assert!(toml.contains("model = \"qwen3-embedding:0.6b\""));
        assert!(toml.contains("dimension = 1024"));
        assert!(toml.contains("model = \"qwen3:4b\""));
    }
}
