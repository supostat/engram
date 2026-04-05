use std::time::Instant;

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Wrap};
use ratatui::Frame;

use crate::theme;

const STATUS_DISPLAY_DURATION_SECS: u64 = 3;

pub struct StatusMessage {
    pub text: String,
    pub is_error: bool,
    pub created_at: Instant,
}

impl StatusMessage {
    pub fn info(text: String) -> Self {
        Self {
            text,
            is_error: false,
            created_at: Instant::now(),
        }
    }

    pub fn error(text: String) -> Self {
        Self {
            text,
            is_error: true,
            created_at: Instant::now(),
        }
    }

    pub fn is_expired(&self) -> bool {
        self.created_at.elapsed().as_secs() >= STATUS_DISPLAY_DURATION_SECS
    }
}

pub enum ConfirmAction {
    DeleteMemory(String),
}

pub struct ConfirmDialog {
    pub message: String,
    pub action: ConfirmAction,
}

pub struct FilterState {
    pub options: Vec<String>,
    pub selected: usize,
    pub current_filter: Option<String>,
}

impl FilterState {
    pub fn new(options: Vec<String>) -> Self {
        Self {
            options,
            selected: 0,
            current_filter: None,
        }
    }

    pub fn move_up(&mut self) {
        self.selected = self.selected.saturating_sub(1);
    }

    pub fn move_down(&mut self) {
        if !self.options.is_empty() {
            self.selected = (self.selected + 1).min(self.options.len() - 1);
        }
    }

    pub fn selected_type(&self) -> Option<&str> {
        self.options.get(self.selected).map(|s| s.as_str())
    }
}

pub fn render_status_message(frame: &mut Frame, area: Rect, message: &StatusMessage) {
    let color = if message.is_error {
        theme::RED
    } else {
        theme::GREEN
    };
    let line = Line::from(Span::styled(&message.text, Style::default().fg(color)));
    frame.render_widget(Paragraph::new(line), area);
}

pub fn render_confirm_dialog(frame: &mut Frame, area: Rect, dialog: &ConfirmDialog) {
    let popup_area = centered_rect(50, 20, area);
    frame.render_widget(Clear, popup_area);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::RED))
        .title(Span::styled(
            " Confirm ",
            Style::default()
                .fg(theme::RED)
                .add_modifier(Modifier::BOLD),
        ));
    let inner = block.inner(popup_area);
    frame.render_widget(block, popup_area);

    let text = format!("{}\n\n  y: confirm  n/Esc: cancel", dialog.message);
    let paragraph = Paragraph::new(text)
        .style(Style::default().fg(theme::TEXT))
        .wrap(Wrap { trim: false });
    frame.render_widget(paragraph, inner);
}

pub fn render_filter_popup(frame: &mut Frame, area: Rect, filter: &FilterState) {
    let popup_height = (filter.options.len() as u16 + 4).min(20);
    let popup_area = centered_rect_fixed(30, popup_height, area);
    frame.render_widget(Clear, popup_area);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::PURPLE))
        .title(Span::styled(
            " Filter by Type ",
            Style::default()
                .fg(theme::PURPLE)
                .add_modifier(Modifier::BOLD),
        ));
    let inner = block.inner(popup_area);
    frame.render_widget(block, popup_area);

    let lines: Vec<Line> = filter
        .options
        .iter()
        .enumerate()
        .map(|(index, option)| {
            let marker = if index == filter.selected { "> " } else { "  " };
            let is_active = filter
                .current_filter
                .as_ref()
                .is_some_and(|current| current == option);
            let color = if index == filter.selected {
                theme::PURPLE
            } else if is_active {
                theme::GREEN
            } else {
                theme::TEXT
            };
            Line::from(Span::styled(
                format!("{marker}{option}"),
                Style::default().fg(color),
            ))
        })
        .collect();

    let hint = Line::from(vec![
        Span::styled("  Enter", Style::default().fg(theme::PURPLE)),
        Span::styled(": select  ", Style::default().fg(theme::MUTED)),
        Span::styled("Esc", Style::default().fg(theme::PURPLE)),
        Span::styled(": cancel", Style::default().fg(theme::MUTED)),
    ]);

    let mut all_lines = lines;
    all_lines.push(Line::from(""));
    all_lines.push(hint);

    let paragraph = Paragraph::new(all_lines);
    frame.render_widget(paragraph, inner);
}

pub fn render_consolidation_preview(frame: &mut Frame, area: Rect, text: &str) {
    let popup_area = centered_rect(50, 40, area);
    frame.render_widget(Clear, popup_area);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::BLUE))
        .title(Span::styled(
            " Consolidation ",
            Style::default()
                .fg(theme::BLUE)
                .add_modifier(Modifier::BOLD),
        ));
    let inner = block.inner(popup_area);
    frame.render_widget(block, popup_area);

    let mut lines: Vec<Line> = text
        .lines()
        .map(|line| Line::from(Span::styled(line.to_string(), Style::default().fg(theme::TEXT))))
        .collect();
    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::styled("  Esc", Style::default().fg(theme::PURPLE)),
        Span::styled(": close", Style::default().fg(theme::MUTED)),
    ]));

    let paragraph = Paragraph::new(lines).wrap(Wrap { trim: false });
    frame.render_widget(paragraph, inner);
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

fn centered_rect_fixed(width: u16, height: u16, area: Rect) -> Rect {
    let top = area.y + area.height.saturating_sub(height) / 2;
    let left = area.x + area.width.saturating_sub(width) / 2;
    Rect::new(left, top, width.min(area.width), height.min(area.height))
}
