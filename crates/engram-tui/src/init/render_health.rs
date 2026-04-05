use ratatui::layout::{Alignment, Constraint, Layout};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Padding, Paragraph};
use ratatui::Frame;

use crate::theme;

use super::existing_config::{format_size, ExistingConfig, HealthStatus};
use super::render_status::{centered_rect, footer_spans};
use super::wizard::InitWizard;

impl InitWizard {
    pub(super) fn render_mcp_snippets(&self, frame: &mut Frame) {
        let area = centered_rect(65, 28, frame.area());
        let block = Block::default()
            .title(" MCP Configuration ")
            .title_style(Style::default().fg(theme::PURPLE).add_modifier(Modifier::BOLD))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(theme::MUTED))
            .padding(Padding::new(2, 2, 1, 0));

        let inner = block.inner(area);
        frame.render_widget(block, area);

        let [content_area, footer_area] =
            Layout::vertical([Constraint::Fill(1), Constraint::Length(1)]).areas(inner);

        let snippet = r#"{ "mcpServers": { "engram": { "command": "engram-mcp" } } }"#;

        let mut lines = Vec::new();
        append_snippet_block(
            &mut lines,
            "Claude Desktop (~/.config/claude/claude_desktop_config.json):",
            snippet,
        );
        lines.push(Line::default());
        append_snippet_block(&mut lines, "Claude Code (.mcp.json):", snippet);
        lines.push(Line::default());
        append_snippet_block(&mut lines, "Cursor (~/.cursor/mcp.json):", snippet);

        frame.render_widget(Paragraph::new(lines), content_area);

        let footer = footer_spans(&[("Esc", "back")]);
        frame.render_widget(
            Paragraph::new(Line::from(footer)).alignment(Alignment::Center),
            footer_area,
        );
    }

    pub(super) fn render_health_check(&self, frame: &mut Frame) {
        let area = centered_rect(60, 22, frame.area());
        let block = Block::default()
            .title(" Connection Check ")
            .title_style(Style::default().fg(theme::PURPLE).add_modifier(Modifier::BOLD))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(theme::MUTED))
            .padding(Padding::new(2, 2, 1, 0));

        let inner = block.inner(area);
        frame.render_widget(block, area);

        let [content_area, footer_area] =
            Layout::vertical([Constraint::Fill(1), Constraint::Length(1)]).areas(inner);

        let lines = match self.cached_health {
            Some(ref health) => build_health_lines(health, &self.existing_config),
            None => vec![Line::from(Span::styled(
                "  No configuration data",
                Style::default().fg(theme::RED),
            ))],
        };

        frame.render_widget(Paragraph::new(lines), content_area);

        let footer = footer_spans(&[("Esc", "back")]);
        frame.render_widget(
            Paragraph::new(Line::from(footer)).alignment(Alignment::Center),
            footer_area,
        );
    }
}

fn build_health_lines(
    health: &HealthStatus,
    existing_config: &Option<ExistingConfig>,
) -> Vec<Line<'static>> {
    let mut lines = vec![
        Line::from(Span::styled(
            "  Checking connections...",
            Style::default()
                .fg(theme::TEXT)
                .add_modifier(Modifier::BOLD),
        )),
        Line::default(),
    ];

    let embedding_provider = existing_config
        .as_ref()
        .map(|c| c.embedding_provider.as_str())
        .unwrap_or("unknown");

    if embedding_provider == "voyage" {
        let (icon, text, color) = if health.embedding_key_present {
            ("✓", "configured", theme::GREEN)
        } else {
            ("✗", "not configured", theme::RED)
        };
        lines.push(health_line("Voyage API", icon, text, color));
    }

    let llm_provider = existing_config
        .as_ref()
        .map(|c| c.llm_provider.as_str())
        .unwrap_or("unknown");

    if llm_provider == "openai" {
        let (icon, text, color) = if health.llm_key_present {
            ("✓", "configured", theme::GREEN)
        } else {
            ("✗", "not configured", theme::RED)
        };
        lines.push(health_line("OpenAI API", icon, text, color));
    }

    if health.database_found {
        let text = format!(
            "{} memories ({})",
            health.database_memory_count,
            format_size(health.database_size_bytes)
        );
        lines.push(health_line("Database", "✓", &text, theme::GREEN));
    } else {
        lines.push(health_line("Database", "✗", "not found", theme::RED));
    }

    if health.hnsw_found {
        lines.push(health_line(
            "HNSW Index",
            "✓",
            &format_size(health.hnsw_size_bytes),
            theme::GREEN,
        ));
    } else {
        lines.push(health_line("HNSW Index", "✗", "not found", theme::RED));
    }

    if health.socket_exists {
        lines.push(health_line("Socket", "✓", "running", theme::GREEN));
    } else {
        lines.push(health_line(
            "Socket",
            "✗",
            "server not running",
            theme::RED,
        ));
    }

    if health.model_count > 0 {
        let text = format!(
            "{} files ({})",
            health.model_count,
            format_size(health.models_size_bytes)
        );
        lines.push(health_line("Models", "✓", &text, theme::GREEN));
    } else {
        lines.push(health_line("Models", "✗", "no models", theme::RED));
    }

    lines
}

fn health_line(
    label: &str,
    icon: &str,
    text: &str,
    color: ratatui::style::Color,
) -> Line<'static> {
    Line::from(vec![
        Span::styled(
            format!("  {label:<14}"),
            Style::default().fg(theme::MUTED),
        ),
        Span::styled(format!("{icon} {text}"), Style::default().fg(color)),
    ])
}

fn append_snippet_block(lines: &mut Vec<Line<'_>>, title: &str, snippet: &str) {
    lines.push(Line::from(Span::styled(
        format!("  {title}"),
        Style::default().fg(theme::TEXT),
    )));

    let border_width = snippet.len() + 4;
    let top_border = format!("  ┌{}┐", "─".repeat(border_width));
    let bottom_border = format!("  └{}┘", "─".repeat(border_width));

    lines.push(Line::from(Span::styled(
        top_border,
        Style::default().fg(theme::MUTED),
    )));
    lines.push(Line::from(vec![
        Span::styled("  │ ", Style::default().fg(theme::MUTED)),
        Span::styled(snippet.to_string(), Style::default().fg(theme::BLUE)),
        Span::styled("   │", Style::default().fg(theme::MUTED)),
    ]));
    lines.push(Line::from(Span::styled(
        bottom_border,
        Style::default().fg(theme::MUTED),
    )));
}
