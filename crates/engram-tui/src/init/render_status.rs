use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Padding, Paragraph};
use ratatui::Frame;

use crate::theme;

use super::existing_config::format_size;
use super::wizard::{InitWizard, Step, STATUS_MENU_LABELS};

impl InitWizard {
    pub(super) fn render_status_screen(&self, frame: &mut Frame) {
        match self.step {
            Step::StatusMenu => self.render_status_menu(frame),
            Step::McpSnippets => self.render_mcp_snippets(frame),
            Step::HealthCheck => self.render_health_check(frame),
            _ => {}
        }
    }

    fn render_status_menu(&self, frame: &mut Frame) {
        let area = centered_rect(60, 28, frame.area());
        let block = Block::default()
            .title(" engram ")
            .title_style(Style::default().fg(theme::PURPLE).add_modifier(Modifier::BOLD))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(theme::MUTED))
            .padding(Padding::new(2, 2, 1, 0));

        let inner = block.inner(area);
        frame.render_widget(block, area);

        let [content_area, footer_area] =
            Layout::vertical([Constraint::Fill(1), Constraint::Length(1)]).areas(inner);

        let mut lines = Vec::new();

        lines.push(styled_line(
            "  ✓ engram configured",
            theme::GREEN,
            Modifier::BOLD,
        ));
        lines.push(Line::default());

        if let Some(ref config) = self.existing_config {
            self.append_config_section(&mut lines, config);
            lines.push(Line::default());
            self.append_stats_section(&mut lines);
            lines.push(Line::default());
        }

        lines.push(Line::from(Span::styled(
            "  Actions:",
            Style::default().fg(theme::TEXT).add_modifier(Modifier::BOLD),
        )));

        for (index, label) in STATUS_MENU_LABELS.iter().enumerate() {
            let (marker, style) = if index == self.status_menu_selection {
                (
                    "    ● ",
                    Style::default()
                        .fg(theme::PURPLE)
                        .add_modifier(Modifier::BOLD),
                )
            } else {
                ("    ○ ", Style::default().fg(theme::TEXT))
            };
            lines.push(Line::from(Span::styled(
                format!("{marker}{label}"),
                style,
            )));
        }

        frame.render_widget(Paragraph::new(lines), content_area);

        let footer = footer_spans(&[("↑↓", "select"), ("Enter", "confirm")]);
        frame.render_widget(
            Paragraph::new(Line::from(footer)).alignment(Alignment::Center),
            footer_area,
        );
    }

    fn append_config_section(
        &self,
        lines: &mut Vec<Line<'_>>,
        config: &super::existing_config::ExistingConfig,
    ) {
        lines.push(Line::from(Span::styled(
            "  Configuration:",
            Style::default().fg(theme::TEXT).add_modifier(Modifier::BOLD),
        )));

        let embedding_key_status =
            if config.embedding_api_key.as_ref().is_some_and(|k| !k.is_empty()) {
                ("✓ API key", theme::GREEN)
            } else if config.embedding_provider == "deterministic" {
                ("not required", theme::MUTED)
            } else {
                ("✗ API key", theme::RED)
            };
        lines.push(config_line_with_status(
            "    Embedding",
            &format!("{} ({})", config.embedding_provider, config.embedding_model),
            embedding_key_status.0,
            embedding_key_status.1,
        ));

        let llm_key_status = if config.llm_api_key.as_ref().is_some_and(|k| !k.is_empty()) {
            ("✓ API key", theme::GREEN)
        } else if config.llm_provider == "openai" {
            ("✗ API key", theme::RED)
        } else {
            ("not required", theme::MUTED)
        };
        lines.push(config_line_with_status(
            "    LLM",
            &format!("{} ({})", config.llm_provider, config.llm_model),
            llm_key_status.0,
            llm_key_status.1,
        ));

        lines.push(simple_config_line("    Database", &config.database_path));
        lines.push(simple_config_line("    Socket", &config.socket_path));
    }

    fn append_stats_section(&self, lines: &mut Vec<Line<'_>>) {
        let Some(ref stats) = self.cached_stats else {
            return;
        };
        lines.push(Line::from(Span::styled(
            "  Statistics:",
            Style::default().fg(theme::TEXT).add_modifier(Modifier::BOLD),
        )));

        let memories_line = format!(
            "    Memories: {:<12}  Models: {} ONNX",
            stats.memory_count, stats.model_count
        );
        lines.push(Line::from(Span::styled(
            memories_line,
            Style::default().fg(theme::TEXT),
        )));

        let indexed_line = format!(
            "    Indexed:  {:<12}  Size:   {}",
            stats.indexed_count,
            format_size(stats.models_size_bytes)
        );
        lines.push(Line::from(Span::styled(
            indexed_line,
            Style::default().fg(theme::TEXT),
        )));

        lines.push(Line::from(Span::styled(
            format!("    Avg Score: {:.2}", stats.average_score),
            Style::default().fg(theme::TEXT),
        )));
    }
}

pub fn centered_rect(percent_width: u16, height: u16, area: Rect) -> Rect {
    let row = Layout::vertical([
        Constraint::Fill(1),
        Constraint::Length(height),
        Constraint::Fill(1),
    ])
    .split(area);
    let margin = (100 - percent_width) / 2;
    Layout::horizontal([
        Constraint::Percentage(margin),
        Constraint::Percentage(percent_width),
        Constraint::Percentage(margin),
    ])
    .split(row[1])[1]
}

pub fn footer_spans(pairs: &[(&str, &str)]) -> Vec<Span<'static>> {
    let mut spans = Vec::with_capacity(pairs.len() * 3);
    for (index, (key, desc)) in pairs.iter().enumerate() {
        if index > 0 {
            spans.push(Span::raw("  "));
        }
        spans.push(Span::styled(
            key.to_string(),
            Style::default().fg(theme::PURPLE),
        ));
        spans.push(Span::styled(
            format!(": {desc}"),
            Style::default().fg(theme::MUTED),
        ));
    }
    spans
}

fn styled_line(text: &str, color: ratatui::style::Color, modifier: Modifier) -> Line<'static> {
    Line::from(Span::styled(
        text.to_string(),
        Style::default().fg(color).add_modifier(modifier),
    ))
}

fn simple_config_line(label: &str, value: &str) -> Line<'static> {
    Line::from(vec![
        Span::styled(
            format!("{label}:   "),
            Style::default().fg(theme::MUTED),
        ),
        Span::styled(value.to_string(), Style::default().fg(theme::TEXT)),
    ])
}

fn config_line_with_status(
    label: &str,
    value: &str,
    status: &str,
    status_color: ratatui::style::Color,
) -> Line<'static> {
    Line::from(vec![
        Span::styled(
            format!("{label}:  "),
            Style::default().fg(theme::MUTED),
        ),
        Span::styled(
            format!("{value}  "),
            Style::default().fg(theme::TEXT),
        ),
        Span::styled(status.to_string(), Style::default().fg(status_color)),
    ])
}
