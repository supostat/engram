use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, Borders, Clear, Paragraph, Row, Scrollbar, ScrollbarOrientation, ScrollbarState, Table,
    TableState, Wrap,
};
use ratatui::Frame;

use crate::data::MemorySummary;
use crate::theme;

pub struct MemoriesTabState {
    pub memories: Vec<MemorySummary>,
    pub selected: usize,
    pub detail_open: bool,
}

impl MemoriesTabState {
    pub fn new() -> Self {
        Self {
            memories: Vec::new(),
            selected: 0,
            detail_open: false,
        }
    }

    pub fn move_up(&mut self) {
        self.selected = self.selected.saturating_sub(1);
    }

    pub fn move_down(&mut self) {
        if !self.memories.is_empty() {
            self.selected = (self.selected + 1).min(self.memories.len() - 1);
        }
    }

    pub fn toggle_detail(&mut self) {
        if !self.memories.is_empty() {
            self.detail_open = !self.detail_open;
        }
    }

    pub fn close_detail(&mut self) {
        self.detail_open = false;
    }

    pub fn clamp_selection(&mut self) {
        if self.memories.is_empty() {
            self.selected = 0;
        } else {
            self.selected = self.selected.min(self.memories.len() - 1);
        }
    }
}

pub fn render_memories_tab(frame: &mut Frame, area: Rect, state: &mut MemoriesTabState) {
    if state.memories.is_empty() {
        render_empty(frame, area);
        return;
    }

    render_table(frame, area, state);

    if state.detail_open {
        render_detail_popup(frame, frame.area(), state);
    }
}

fn render_empty(frame: &mut Frame, area: Rect) {
    let paragraph = Paragraph::new("No memories found")
        .style(Style::default().fg(theme::MUTED))
        .alignment(ratatui::layout::Alignment::Center)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(theme::MUTED))
                .title(Span::styled("Memories", Style::default().fg(theme::BLUE))),
        );
    frame.render_widget(paragraph, area);
}

fn type_color(memory_type: &str) -> ratatui::style::Color {
    match memory_type {
        "decision" => theme::PURPLE,
        "pattern" => theme::BLUE,
        "bugfix" => theme::RED,
        "antipattern" => theme::RED,
        "context" => theme::MUTED,
        "insight" => theme::GREEN,
        _ => theme::TEXT,
    }
}

fn render_table(frame: &mut Frame, area: Rect, state: &mut MemoriesTabState) {
    let header = Row::new(vec!["Type", "Project", "Context", "Score", "Created"])
        .style(
            Style::default()
                .fg(theme::BLUE)
                .add_modifier(Modifier::BOLD),
        )
        .bottom_margin(0);

    let rows: Vec<Row> = state
        .memories
        .iter()
        .map(|memory| {
            let score_color = if memory.score >= 0.5 {
                theme::GREEN
            } else {
                theme::MUTED
            };
            let truncated_context = truncate_to_width(&memory.context, 60);
            let created_short = truncate_to_width(&memory.created_at, 10);
            Row::new(vec![
                Span::styled(
                    memory.memory_type.clone(),
                    Style::default().fg(type_color(&memory.memory_type)),
                ),
                Span::styled(memory.project_display(), Style::default().fg(theme::MUTED)),
                Span::styled(truncated_context, Style::default().fg(theme::TEXT)),
                Span::styled(format!("{:.2}", memory.score), Style::default().fg(score_color)),
                Span::styled(created_short, Style::default().fg(theme::MUTED)),
            ])
        })
        .collect();

    let widths = [
        Constraint::Length(14),
        Constraint::Length(14),
        Constraint::Fill(1),
        Constraint::Length(6),
        Constraint::Length(10),
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
                    format!("Memories ({})", state.memories.len()),
                    Style::default().fg(theme::BLUE),
                )),
        )
        .column_spacing(1);

    let mut table_state = TableState::default().with_selected(Some(state.selected));
    frame.render_stateful_widget(table, area, &mut table_state);

    let scrollbar = Scrollbar::new(ScrollbarOrientation::VerticalRight);
    let mut scrollbar_state =
        ScrollbarState::new(state.memories.len()).position(state.selected);
    frame.render_stateful_widget(scrollbar, area, &mut scrollbar_state);
}

fn render_detail_popup(frame: &mut Frame, area: Rect, state: &MemoriesTabState) {
    let memory = &state.memories[state.selected];

    let popup_area = centered_rect(80, 60, area);
    frame.render_widget(Clear, popup_area);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::PURPLE))
        .title(Span::styled(
            format!(" {} ", memory.memory_type),
            Style::default()
                .fg(type_color(&memory.memory_type))
                .add_modifier(Modifier::BOLD),
        ));
    let inner = block.inner(popup_area);
    frame.render_widget(block, popup_area);

    let [meta_area, context_area, action_area, result_area] = Layout::vertical([
        Constraint::Length(3),
        Constraint::Fill(1),
        Constraint::Fill(1),
        Constraint::Fill(1),
    ])
    .areas(inner);

    let meta_text = Line::from(vec![
        Span::styled("Project: ", Style::default().fg(theme::BLUE)),
        Span::styled(memory.project_display(), Style::default().fg(theme::TEXT)),
        Span::styled("  Score: ", Style::default().fg(theme::BLUE)),
        Span::styled(format!("{:.2}", memory.score), Style::default().fg(theme::GREEN)),
        Span::styled("  Created: ", Style::default().fg(theme::BLUE)),
        Span::styled(&memory.created_at, Style::default().fg(theme::MUTED)),
    ]);
    frame.render_widget(
        Paragraph::new(meta_text).wrap(Wrap { trim: false }),
        meta_area,
    );

    render_section(frame, context_area, "Context", &memory.context);
    render_section(frame, action_area, "Action", &memory.action);
    render_section(frame, result_area, "Result", &memory.result);
}

fn render_section(frame: &mut Frame, area: Rect, title: &str, content: &str) {
    let display = if content.is_empty() { "(empty)" } else { content };
    let paragraph = Paragraph::new(display.to_string())
        .style(Style::default().fg(theme::TEXT))
        .wrap(Wrap { trim: false })
        .block(
            Block::default()
                .borders(Borders::TOP)
                .border_style(Style::default().fg(theme::MUTED))
                .title(Span::styled(title, Style::default().fg(theme::BLUE))),
        );
    frame.render_widget(paragraph, area);
}

fn truncate_to_width(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let truncated: String = text.chars().take(max_chars.saturating_sub(3)).collect();
    format!("{truncated}...")
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
