use std::io;
use std::time::Duration;

use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind};
use ratatui::DefaultTerminal;

const POLL_TIMEOUT: Duration = Duration::from_millis(100);

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Step {
    EmbeddingProvider,
    EmbeddingApiKey,
    LlmProvider,
    LlmApiKey,
    DatabasePath,
    McpClient,
    Summary,
}

impl Step {
    pub const ALL: [Step; 7] = [
        Step::EmbeddingProvider,
        Step::EmbeddingApiKey,
        Step::LlmProvider,
        Step::LlmApiKey,
        Step::DatabasePath,
        Step::McpClient,
        Step::Summary,
    ];

    pub fn number(self) -> usize {
        Step::ALL.iter().position(|&s| s == self).unwrap_or(0) + 1
    }
}

pub struct InitWizard {
    pub(super) step: Step,
    pub(super) embedding_provider: usize,
    pub(super) embedding_api_key: String,
    pub(super) llm_provider: usize,
    pub(super) llm_api_key: String,
    pub(super) database_path: String,
    pub(super) mcp_client: usize,
    pub(super) text_input: String,
    pub(super) cursor_position: usize,
    pub(super) created: bool,
    pub(super) error_message: Option<String>,
    pub(super) should_quit: bool,
}

pub const EMBEDDING_OPTIONS: [&str; 2] = ["voyage", "deterministic"];
pub const EMBEDDING_LABELS: [&str; 2] = [
    "Voyage AI  (voyage-code-3, recommended)",
    "Deterministic  (no API key, lower quality)",
];

pub const LLM_OPTIONS: [&str; 3] = ["openai", "local", "none"];
pub const LLM_LABELS: [&str; 3] = [
    "OpenAI  (gpt-4o-mini, recommended)",
    "Local  (via engram-llm, no API key)",
    "None  (disable LLM features)",
];

pub const MCP_OPTIONS: [&str; 4] = ["claude-desktop", "claude-code", "cursor", "skip"];
pub const MCP_LABELS: [&str; 4] = ["Claude Desktop", "Claude Code", "Cursor", "Skip"];

impl InitWizard {
    pub fn new() -> Self {
        let default_database_path = dirs::home_dir()
            .map(|home| home.join(".engram/memories.db").to_string_lossy().into_owned())
            .unwrap_or_else(|| "~/.engram/memories.db".into());

        Self {
            step: Step::EmbeddingProvider,
            embedding_provider: 0,
            embedding_api_key: String::new(),
            llm_provider: 0,
            llm_api_key: String::new(),
            database_path: default_database_path.clone(),
            mcp_client: 0,
            text_input: default_database_path,
            cursor_position: 0,
            created: false,
            error_message: None,
            should_quit: false,
        }
    }

    pub fn run(&mut self, terminal: &mut DefaultTerminal) -> io::Result<()> {
        while !self.should_quit {
            terminal.draw(|frame| self.render(frame))?;
            self.handle_events()?;
        }
        Ok(())
    }

    fn handle_events(&mut self) -> io::Result<()> {
        if !event::poll(POLL_TIMEOUT)? {
            return Ok(());
        }
        let Event::Key(key) = event::read()? else {
            return Ok(());
        };
        if key.kind != KeyEventKind::Press {
            return Ok(());
        }
        self.error_message = None;
        match key.code {
            KeyCode::Char('q') => self.should_quit = true,
            KeyCode::Esc => self.go_back(),
            _ => self.handle_step_key(key.code),
        }
        Ok(())
    }

    fn handle_step_key(&mut self, code: KeyCode) {
        match self.step {
            Step::EmbeddingProvider => self.handle_radio_key(code, EMBEDDING_OPTIONS.len()),
            Step::EmbeddingApiKey => self.handle_text_key(code),
            Step::LlmProvider => self.handle_radio_key(code, LLM_OPTIONS.len()),
            Step::LlmApiKey => self.handle_text_key(code),
            Step::DatabasePath => self.handle_text_key(code),
            Step::McpClient => self.handle_radio_key(code, MCP_OPTIONS.len()),
            Step::Summary => self.handle_summary_key(code),
        }
    }

    fn handle_radio_key(&mut self, code: KeyCode, option_count: usize) {
        let selected = self.current_radio_selection();
        match code {
            KeyCode::Up | KeyCode::Char('k') => {
                let new_value = selected.checked_sub(1).unwrap_or(option_count - 1);
                self.set_radio_selection(new_value);
            }
            KeyCode::Down | KeyCode::Char('j') => {
                let new_value = (selected + 1) % option_count;
                self.set_radio_selection(new_value);
            }
            KeyCode::Enter => self.advance(),
            _ => {}
        }
    }

    fn handle_text_key(&mut self, code: KeyCode) {
        match code {
            KeyCode::Enter => self.advance(),
            KeyCode::Backspace => {
                if self.cursor_position > 0 {
                    self.text_input.remove(self.cursor_position - 1);
                    self.cursor_position -= 1;
                }
            }
            KeyCode::Left => {
                self.cursor_position = self.cursor_position.saturating_sub(1);
            }
            KeyCode::Right => {
                if self.cursor_position < self.text_input.len() {
                    self.cursor_position += 1;
                }
            }
            KeyCode::Char(character) => {
                self.text_input.insert(self.cursor_position, character);
                self.cursor_position += 1;
            }
            _ => {}
        }
    }

    fn handle_summary_key(&mut self, code: KeyCode) {
        if code == KeyCode::Enter && !self.created {
            match self.create_config_files() {
                Ok(()) => self.created = true,
                Err(error) => self.error_message = Some(error.to_string()),
            }
        }
    }

    fn current_radio_selection(&self) -> usize {
        match self.step {
            Step::EmbeddingProvider => self.embedding_provider,
            Step::LlmProvider => self.llm_provider,
            Step::McpClient => self.mcp_client,
            _ => 0,
        }
    }

    fn set_radio_selection(&mut self, value: usize) {
        match self.step {
            Step::EmbeddingProvider => self.embedding_provider = value,
            Step::LlmProvider => self.llm_provider = value,
            Step::McpClient => self.mcp_client = value,
            _ => {}
        }
    }

    fn advance(&mut self) {
        match self.step {
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

    fn go_back(&mut self) {
        match self.step {
            Step::EmbeddingProvider => self.should_quit = true,
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
