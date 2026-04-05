mod models_state;

pub use models_state::{ModelsKeyAction, ModelsPopup, ModelsTabState};

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Row, Table, TableState, Wrap};
use ratatui::Frame;

use crate::theme;

pub fn render_models_tab(frame: &mut Frame, area: Rect, state: &mut ModelsTabState) {
    if state.expired_status_message() {
        state.status_message = None;
    }

    let [table_area, summary_area, footer_area] = Layout::vertical([
        Constraint::Fill(1),
        Constraint::Length(3),
        Constraint::Length(1),
    ])
    .areas(area);

    render_models_table(frame, table_area, state);
    render_summary(frame, summary_area, state);
    render_models_footer(frame, footer_area, state);

    if let Some(popup) = &state.popup {
        render_popup(frame, frame.area(), popup);
    }
}

fn render_models_table(frame: &mut Frame, area: Rect, state: &mut ModelsTabState) {
    let header = Row::new(vec!["Status", "Filename", "Size", "Modified"])
        .style(
            Style::default()
                .fg(theme::BLUE)
                .add_modifier(Modifier::BOLD),
        )
        .bottom_margin(0);

    let present: Vec<&str> = state.models.iter().map(|m| m.filename.as_str()).collect();

    let mut rows: Vec<Row> = state
        .models
        .iter()
        .map(|model| {
            Row::new(vec![
                Span::styled(" \u{25CF} ", Style::default().fg(theme::GREEN)),
                Span::styled(model.filename.clone(), Style::default().fg(theme::TEXT)),
                Span::styled(format_size(model.size_bytes), Style::default().fg(theme::MUTED)),
                Span::styled(model.modified.clone(), Style::default().fg(theme::MUTED)),
            ])
        })
        .collect();

    for expected in models_state::expected_files() {
        if !present.contains(expected) {
            rows.push(Row::new(vec![
                Span::styled(" \u{25CB} ", Style::default().fg(theme::MUTED)),
                Span::styled((*expected).to_string(), Style::default().fg(theme::MUTED)),
                Span::styled("\u{2014}".to_string(), Style::default().fg(theme::MUTED)),
                Span::styled("missing".to_string(), Style::default().fg(theme::RED)),
            ]));
        }
    }

    let widths = [
        Constraint::Length(4),
        Constraint::Fill(1),
        Constraint::Length(12),
        Constraint::Length(12),
    ];

    let table = Table::new(rows, widths)
        .header(header)
        .row_highlight_style(
            Style::default()
                .fg(theme::PURPLE)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("> ")
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(theme::MUTED))
                .title(Span::styled(
                    format!("Models ({})", state.models_path),
                    Style::default().fg(theme::BLUE),
                )),
        )
        .column_spacing(1);

    let mut table_state = TableState::default().with_selected(Some(state.selected));
    frame.render_stateful_widget(table, area, &mut table_state);
}

fn render_summary(frame: &mut Frame, area: Rect, state: &ModelsTabState) {
    let total_bytes: u64 = state.models.iter().map(|m| m.size_bytes).sum();
    let text = Line::from(vec![
        Span::styled("  Files: ", Style::default().fg(theme::BLUE)),
        Span::styled(
            state.models.len().to_string(),
            Style::default().fg(theme::PURPLE).add_modifier(Modifier::BOLD),
        ),
        Span::styled("  Total size: ", Style::default().fg(theme::BLUE)),
        Span::styled(
            format_size(total_bytes),
            Style::default().fg(theme::PURPLE).add_modifier(Modifier::BOLD),
        ),
    ]);
    let paragraph = Paragraph::new(text).block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(theme::MUTED)),
    );
    frame.render_widget(paragraph, area);
}

fn render_models_footer(frame: &mut Frame, area: Rect, state: &ModelsTabState) {
    if let Some((message, _)) = &state.status_message {
        let line = Line::from(vec![
            Span::styled(" ", Style::default()),
            Span::styled(message.clone(), Style::default().fg(theme::GREEN)),
        ]);
        frame.render_widget(Paragraph::new(line), area);
        return;
    }
    let hints = footer_hints(&[
        ("j/k", "scroll"),
        ("d", "delete"),
        ("t", "train"),
        ("T", "deep train"),
    ]);
    frame.render_widget(Paragraph::new(Line::from(hints)), area);
}

fn render_popup(frame: &mut Frame, area: Rect, popup: &ModelsPopup) {
    let popup_area = centered_rect(60, 20, area);
    frame.render_widget(Clear, popup_area);

    match popup {
        ModelsPopup::TrainCommand { command } => {
            let text = vec![
                Line::from(""),
                Line::from(Span::styled(
                    " Run in another terminal:",
                    Style::default().fg(theme::TEXT),
                )),
                Line::from(""),
                Line::from(Span::styled(
                    format!("  $ {command}"),
                    Style::default().fg(theme::GREEN).add_modifier(Modifier::BOLD),
                )),
                Line::from(""),
                Line::from(Span::styled(
                    " Press Esc to close",
                    Style::default().fg(theme::MUTED),
                )),
            ];
            let paragraph = Paragraph::new(text)
                .wrap(Wrap { trim: false })
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .border_style(Style::default().fg(theme::PURPLE))
                        .title(Span::styled(
                            " Train ",
                            Style::default().fg(theme::PURPLE).add_modifier(Modifier::BOLD),
                        )),
                );
            frame.render_widget(paragraph, popup_area);
        }
        ModelsPopup::DeleteConfirm { filename } => {
            let text = vec![
                Line::from(""),
                Line::from(Span::styled(
                    format!(" Delete {filename}?"),
                    Style::default().fg(theme::RED),
                )),
                Line::from(""),
                Line::from(vec![
                    Span::styled(" y", Style::default().fg(theme::PURPLE)),
                    Span::styled(": confirm  ", Style::default().fg(theme::MUTED)),
                    Span::styled("any", Style::default().fg(theme::PURPLE)),
                    Span::styled(": cancel", Style::default().fg(theme::MUTED)),
                ]),
            ];
            let paragraph = Paragraph::new(text).block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(theme::RED))
                    .title(Span::styled(
                        " Confirm Delete ",
                        Style::default().fg(theme::RED).add_modifier(Modifier::BOLD),
                    )),
            );
            frame.render_widget(paragraph, popup_area);
        }
    }
}

fn footer_hints(pairs: &[(&str, &str)]) -> Vec<Span<'static>> {
    let mut spans = Vec::with_capacity(pairs.len() * 2);
    for (key, description) in pairs {
        let prefix = if spans.is_empty() { " " } else { "  " };
        spans.push(Span::styled(format!("{prefix}{key}"), Style::default().fg(theme::PURPLE)));
        spans.push(Span::styled(format!(": {description}"), Style::default().fg(theme::MUTED)));
    }
    spans
}

fn centered_rect(percent_x: u16, percent_y: u16, area: Rect) -> Rect {
    let [_, vertical_center, _] = Layout::vertical([
        Constraint::Percentage((100 - percent_y) / 2),
        Constraint::Percentage(percent_y),
        Constraint::Percentage((100 - percent_y) / 2),
    ])
    .areas(area);
    let [_, horizontal_center, _] = Layout::horizontal([
        Constraint::Percentage((100 - percent_x) / 2),
        Constraint::Percentage(percent_x),
        Constraint::Percentage((100 - percent_x) / 2),
    ])
    .areas(vertical_center);
    horizontal_center
}

fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * KB;
    const GB: u64 = 1024 * MB;

    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}
