use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Padding, Paragraph};
use ratatui::Frame;

use crate::theme;

use super::config::MCP_JSON_SNIPPET;
use super::wizard::{
    EMBEDDING_LABELS, EMBEDDING_OPTIONS, InitWizard, LLM_LABELS, LLM_OPTIONS, MCP_LABELS,
    MCP_OPTIONS, Step,
};

impl InitWizard {
    pub(super) fn render(&self, frame: &mut Frame) {
        let area = centered_rect(60, 22, frame.area());
        let title = format!(
            " engram init --- Step {}/{} ",
            self.step.number(),
            Step::ALL.len()
        );
        let block = Block::default()
            .title(title)
            .title_style(Style::default().fg(theme::PURPLE).add_modifier(Modifier::BOLD))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(theme::MUTED))
            .padding(Padding::new(2, 2, 1, 0));

        let inner = block.inner(area);
        frame.render_widget(block, area);

        let [content_area, footer_area] = Layout::vertical([
            Constraint::Fill(1),
            Constraint::Length(1),
        ])
        .areas(inner);

        self.render_step_content(frame, content_area);
        self.render_step_footer(frame, footer_area);
    }

    fn render_step_content(&self, frame: &mut Frame, area: Rect) {
        match self.step {
            Step::EmbeddingProvider => self.render_radio_step(
                frame, area,
                "Embedding Provider",
                "How should engram generate embeddings for semantic search?",
                &EMBEDDING_LABELS,
                self.embedding_provider,
            ),
            Step::EmbeddingApiKey => self.render_text_step(
                frame, area,
                "Voyage API Key",
                "Enter your Voyage AI API key (dashboard.voyageai.com)",
                true,
            ),
            Step::LlmProvider => self.render_radio_step(
                frame, area,
                "LLM Provider",
                "Which LLM should engram use for consolidation and insights?",
                &LLM_LABELS,
                self.llm_provider,
            ),
            Step::LlmApiKey => self.render_text_step(
                frame, area,
                "OpenAI API Key",
                "Enter your OpenAI API key (platform.openai.com)",
                true,
            ),
            Step::DatabasePath => self.render_text_step(
                frame, area,
                "Database Path",
                "Where should engram store memories?",
                false,
            ),
            Step::McpClient => self.render_radio_step(
                frame, area,
                "MCP Client",
                "Which client will connect to engram via MCP?",
                &MCP_LABELS,
                self.mcp_client,
            ),
            Step::Summary => self.render_summary(frame, area),
        }
    }

    fn render_radio_step(
        &self,
        frame: &mut Frame,
        area: Rect,
        title: &str,
        description: &str,
        labels: &[&str],
        selected: usize,
    ) {
        let mut lines = vec![
            Line::from(Span::styled(
                title,
                Style::default().fg(theme::TEXT).add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::styled(description, Style::default().fg(theme::MUTED))),
            Line::default(),
        ];
        for (index, label) in labels.iter().enumerate() {
            let (marker, style) = if index == selected {
                ("  > ", Style::default().fg(theme::PURPLE).add_modifier(Modifier::BOLD))
            } else {
                ("    ", Style::default().fg(theme::TEXT))
            };
            lines.push(Line::from(Span::styled(format!("{marker}{label}"), style)));
        }
        frame.render_widget(Paragraph::new(lines), area);
    }

    fn render_text_step(
        &self,
        frame: &mut Frame,
        area: Rect,
        title: &str,
        description: &str,
        is_secret: bool,
    ) {
        let display_value = if is_secret && !self.text_input.is_empty() {
            let visible_prefix: String = self.text_input.chars().take(3).collect();
            let masked_len = self.text_input.len().saturating_sub(3);
            format!("{}{}", visible_prefix, "*".repeat(masked_len))
        } else {
            self.text_input.clone()
        };

        let cursor_line = format!("  > {}_", display_value);

        let mut lines = vec![
            Line::from(Span::styled(
                title,
                Style::default().fg(theme::TEXT).add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::styled(description, Style::default().fg(theme::MUTED))),
            Line::default(),
            Line::from(Span::styled(cursor_line, Style::default().fg(theme::PURPLE))),
        ];

        if let Some(ref error) = self.error_message {
            lines.push(Line::default());
            lines.push(Line::from(Span::styled(
                format!("  Error: {error}"),
                Style::default().fg(theme::RED),
            )));
        }

        frame.render_widget(Paragraph::new(lines), area);
    }

    fn render_summary(&self, frame: &mut Frame, area: Rect) {
        let embedding_name = EMBEDDING_OPTIONS[self.embedding_provider];
        let llm_name = LLM_OPTIONS[self.llm_provider];
        let mcp_name = MCP_OPTIONS[self.mcp_client];

        let embedding_model = if embedding_name == "voyage" {
            "voyage-code-3"
        } else {
            "deterministic"
        };
        let llm_model = match llm_name {
            "openai" => "gpt-4o-mini",
            "local" => "local",
            _ => "disabled",
        };

        let embedding_display = format!("{embedding_name} ({embedding_model})");
        let llm_display = format!("{llm_name} ({llm_model})");

        let mut lines = vec![
            Line::from(Span::styled(
                "Configuration",
                Style::default().fg(theme::TEXT).add_modifier(Modifier::BOLD),
            )),
            Line::default(),
            config_line("  Embedding", &embedding_display),
            config_line("  LLM", &llm_display),
            config_line("  Database", &self.database_path),
            config_line("  MCP client", mcp_name),
        ];

        if self.created {
            lines.push(Line::default());
            lines.push(Line::from(Span::styled(
                "  [ok] Created ~/.engram/engram.toml",
                Style::default().fg(theme::GREEN),
            )));
            self.append_env_hints(&mut lines, embedding_name, llm_name);
            self.append_mcp_snippet(&mut lines, mcp_name);
        } else if let Some(ref error) = self.error_message {
            lines.push(Line::default());
            lines.push(Line::from(Span::styled(
                format!("  Error: {error}"),
                Style::default().fg(theme::RED),
            )));
        }

        frame.render_widget(Paragraph::new(lines), area);
    }

    fn append_env_hints(&self, lines: &mut Vec<Line<'_>>, embedding: &str, llm: &str) {
        let needs_voyage = embedding == "voyage" && self.embedding_api_key.is_empty();
        let needs_openai = llm == "openai" && self.llm_api_key.is_empty();
        if !needs_voyage && !needs_openai {
            return;
        }
        lines.push(Line::default());
        lines.push(colored_line("  Add to your shell profile:", theme::MUTED));
        if needs_voyage {
            lines.push(colored_line("    export ENGRAM_VOYAGE_API_KEY=...", theme::BLUE));
        }
        if needs_openai {
            lines.push(colored_line("    export ENGRAM_OPENAI_API_KEY=...", theme::BLUE));
        }
    }

    fn append_mcp_snippet(&self, lines: &mut Vec<Line<'_>>, mcp: &str) {
        let config_path = match mcp {
            "claude-desktop" => "~/.config/claude/claude_desktop_config.json",
            "claude-code" => ".mcp.json (project root)",
            "cursor" => "~/.cursor/mcp.json",
            _ => return,
        };
        lines.push(Line::default());
        lines.push(colored_line(&format!("  MCP config  ({config_path}):"), theme::MUTED));
        for snippet_line in MCP_JSON_SNIPPET.lines() {
            lines.push(colored_line(&format!("    {snippet_line}"), theme::BLUE));
        }
    }

    pub(super) fn render_step_footer(&self, frame: &mut Frame, area: Rect) {
        let hints = match self.step {
            Step::Summary if self.created => vec![("q", "quit")],
            Step::Summary => vec![("Enter", "create"), ("Esc", "back"), ("q", "quit")],
            Step::EmbeddingApiKey | Step::LlmApiKey | Step::DatabasePath => {
                vec![("Enter", "next"), ("Esc", "back"), ("q", "quit")]
            }
            _ => vec![
                ("j/k", "select"),
                ("Enter", "next"),
                ("Esc", "back"),
                ("q", "quit"),
            ],
        };
        let spans = footer_spans(&hints);
        frame.render_widget(
            Paragraph::new(Line::from(spans)).alignment(Alignment::Center),
            area,
        );
    }
}

fn colored_line(text: &str, color: ratatui::style::Color) -> Line<'static> {
    Line::from(Span::styled(text.to_string(), Style::default().fg(color)))
}

fn config_line(label: &str, value: &str) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("{label}: "), Style::default().fg(theme::MUTED)),
        Span::styled(value.to_string(), Style::default().fg(theme::TEXT)),
    ])
}

fn footer_spans(pairs: &[(&str, &str)]) -> Vec<Span<'static>> {
    let mut spans = Vec::with_capacity(pairs.len() * 3);
    for (index, (key, desc)) in pairs.iter().enumerate() {
        if index > 0 {
            spans.push(Span::raw("  "));
        }
        spans.push(Span::styled(key.to_string(), Style::default().fg(theme::PURPLE)));
        spans.push(Span::styled(format!(": {desc}"), Style::default().fg(theme::MUTED)));
    }
    spans
}

fn centered_rect(percent_width: u16, height: u16, area: Rect) -> Rect {
    let row = Layout::vertical([
        Constraint::Fill(1), Constraint::Length(height), Constraint::Fill(1),
    ]).split(area);
    let margin = (100 - percent_width) / 2;
    Layout::horizontal([
        Constraint::Percentage(margin), Constraint::Percentage(percent_width), Constraint::Percentage(margin),
    ]).split(row[1])[1]
}
