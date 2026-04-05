use super::wizard::{
    EMBEDDING_OPTIONS, InitWizard, LLM_OPTIONS, STATUS_MENU_LABELS, Step,
};

impl InitWizard {
    pub(super) fn handle_status_menu_key(&mut self, code: ratatui::crossterm::event::KeyCode) {
        use ratatui::crossterm::event::KeyCode;
        match code {
            KeyCode::Up | KeyCode::Char('k') => {
                self.status_menu_selection = self
                    .status_menu_selection
                    .checked_sub(1)
                    .unwrap_or(STATUS_MENU_LABELS.len() - 1);
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.status_menu_selection =
                    (self.status_menu_selection + 1) % STATUS_MENU_LABELS.len();
            }
            KeyCode::Enter => self.execute_status_menu_action(),
            _ => {}
        }
    }

    fn execute_status_menu_action(&mut self) {
        match self.status_menu_selection {
            0 => {
                self.text_input = self.embedding_api_key.clone();
                self.cursor_position = self.text_input.len();
                self.step = Step::EmbeddingProvider;
            }
            1 => self.step = Step::McpSnippets,
            2 => {
                if let Some(ref config) = self.existing_config {
                    self.cached_health = Some(config.run_health_check());
                }
                self.step = Step::HealthCheck;
            }
            _ => self.should_quit = true,
        }
    }

    pub(super) fn advance(&mut self) {
        match self.step {
            Step::StatusMenu | Step::McpSnippets | Step::HealthCheck => {}
            Step::EmbeddingProvider => {
                if EMBEDDING_OPTIONS[self.embedding_provider] == "voyage" {
                    self.text_input = self.embedding_api_key.clone();
                    self.cursor_position = self.text_input.len();
                    self.step = Step::EmbeddingApiKey;
                } else {
                    self.step = Step::LlmProvider;
                }
            }
            Step::EmbeddingApiKey => {
                self.embedding_api_key = self.text_input.clone();
                self.step = Step::LlmProvider;
            }
            Step::LlmProvider => {
                if LLM_OPTIONS[self.llm_provider] == "openai" {
                    self.text_input = self.llm_api_key.clone();
                    self.cursor_position = self.text_input.len();
                    self.step = Step::LlmApiKey;
                } else {
                    self.text_input = self.database_path.clone();
                    self.cursor_position = self.text_input.len();
                    self.step = Step::DatabasePath;
                }
            }
            Step::LlmApiKey => {
                self.llm_api_key = self.text_input.clone();
                self.text_input = self.database_path.clone();
                self.cursor_position = self.text_input.len();
                self.step = Step::DatabasePath;
            }
            Step::DatabasePath => {
                self.database_path = self.text_input.clone();
                self.step = Step::McpClient;
            }
            Step::McpClient => {
                self.step = Step::Summary;
            }
            Step::Summary => {}
        }
    }

    pub(super) fn go_back(&mut self) {
        match self.step {
            Step::StatusMenu => self.should_quit = true,
            Step::McpSnippets | Step::HealthCheck => self.step = Step::StatusMenu,
            Step::EmbeddingProvider => {
                if self.existing_config.is_some() {
                    self.step = Step::StatusMenu;
                } else {
                    self.should_quit = true;
                }
            }
            Step::EmbeddingApiKey => self.step = Step::EmbeddingProvider,
            Step::LlmProvider => {
                if EMBEDDING_OPTIONS[self.embedding_provider] == "voyage" {
                    self.text_input = self.embedding_api_key.clone();
                    self.cursor_position = self.text_input.len();
                    self.step = Step::EmbeddingApiKey;
                } else {
                    self.step = Step::EmbeddingProvider;
                }
            }
            Step::LlmApiKey => self.step = Step::LlmProvider,
            Step::DatabasePath => {
                if LLM_OPTIONS[self.llm_provider] == "openai" {
                    self.text_input = self.llm_api_key.clone();
                    self.cursor_position = self.text_input.len();
                    self.step = Step::LlmApiKey;
                } else {
                    self.step = Step::LlmProvider;
                }
            }
            Step::McpClient => {
                self.text_input = self.database_path.clone();
                self.cursor_position = self.text_input.len();
                self.step = Step::DatabasePath;
            }
            Step::Summary => self.step = Step::McpClient,
        }
    }
}
