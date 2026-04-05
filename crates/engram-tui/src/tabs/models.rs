use ratatui::layout::Constraint;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Row, Table};
use ratatui::Frame;
use ratatui::layout::Rect;

use crate::data::ModelInfo;
use crate::theme;

const EXPECTED_FILES: &[&str] = &[
    "mode_classifier.onnx",
    "ranking_model.onnx",
    "text_generator.onnx",
    "tokenizer.json",
];

pub struct ModelsTabState {
    pub models: Vec<ModelInfo>,
    pub models_path: String,
}

impl ModelsTabState {
    pub fn new(models_path: String) -> Self {
        Self {
            models: Vec::new(),
            models_path,
        }
    }
}

pub fn render_models_tab(frame: &mut Frame, area: Rect, state: &ModelsTabState) {
    let [table_area, summary_area] = ratatui::layout::Layout::vertical([
        Constraint::Fill(1),
        Constraint::Length(3),
    ])
    .areas(area);

    render_models_table(frame, table_area, state);
    render_summary(frame, summary_area, state);
}

fn render_models_table(frame: &mut Frame, area: Rect, state: &ModelsTabState) {
    let header = Row::new(vec!["Status", "Filename", "Size", "Modified"])
        .style(
            Style::default()
                .fg(theme::BLUE)
                .add_modifier(Modifier::BOLD),
        )
        .bottom_margin(0);

    let present_filenames: Vec<&str> = state.models.iter().map(|m| m.filename.as_str()).collect();

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

    for expected in EXPECTED_FILES {
        if !present_filenames.contains(expected) {
            rows.push(Row::new(vec![
                Span::styled(" \u{25CB} ", Style::default().fg(theme::MUTED)),
                Span::styled((*expected).to_string(), Style::default().fg(theme::MUTED)),
                Span::styled("—".to_string(), Style::default().fg(theme::MUTED)),
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

    frame.render_widget(table, area);
}

fn render_summary(frame: &mut Frame, area: Rect, state: &ModelsTabState) {
    let total_bytes: u64 = state.models.iter().map(|m| m.size_bytes).sum();
    let file_count = state.models.len();

    let text = Line::from(vec![
        Span::styled("  Files: ", Style::default().fg(theme::BLUE)),
        Span::styled(
            file_count.to_string(),
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
