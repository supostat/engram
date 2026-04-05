mod search_state;

pub use search_state::{SearchKeyAction, SearchStatus, SearchTabState};

use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, Borders, Clear, Paragraph, Row, Scrollbar, ScrollbarOrientation, ScrollbarState, Table,
    TableState, Wrap,
};
use ratatui::Frame;

use crate::theme;

pub fn render_search_tab(frame: &mut Frame, area: Rect, state: &mut SearchTabState) {
    if matches!(state.status, SearchStatus::Offline) {
        render_offline(frame, area);
        return;
    }

    let [input_area, results_area, footer_area] = Layout::vertical([
        Constraint::Length(3),
        Constraint::Fill(1),
        Constraint::Length(1),
    ])
    .areas(area);

    render_input(frame, input_area, state);
    render_results(frame, results_area, state);
    render_search_footer(frame, footer_area, state);

    if state.detail_open && !state.results.is_empty() {
        render_detail_popup(frame, frame.area(), state);
    }
}

fn render_offline(frame: &mut Frame, area: Rect) {
    let text = "\n\
                \x20 Сервер не запущен\n\
                \x20 Запустите: engram server\n\n\
                \x20 Search требует подключения к engram-core через\n\
                \x20 unix socket (~/.engram/engram.sock)";
    let paragraph = Paragraph::new(text)
        .style(Style::default().fg(theme::MUTED))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(theme::MUTED))
                .title(Span::styled("Search", Style::default().fg(theme::BLUE))),
        );
    frame.render_widget(paragraph, area);
}

fn render_input(frame: &mut Frame, area: Rect, state: &SearchTabState) {
    let display_query = if state.input_active {
        let (before, after) = state.query.split_at(state.cursor_position);
        format!("{before}\u{2588}{after}")
    } else {
        state.query.clone()
    };

    let input_line = Line::from(vec![
        Span::styled("  Query: ", Style::default().fg(theme::BLUE)),
        Span::styled(display_query, Style::default().fg(theme::TEXT)),
    ]);

    let border_color = if state.input_active {
        theme::PURPLE
    } else {
        theme::MUTED
    };

    let paragraph = Paragraph::new(input_line).block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(border_color))
            .title(Span::styled("Search", Style::default().fg(theme::BLUE))),
    );
    frame.render_widget(paragraph, area);
}

fn render_results(frame: &mut Frame, area: Rect, state: &mut SearchTabState) {
    match &state.status {
        SearchStatus::Idle => {
            render_centered_message(frame, area, "Введите запрос и нажмите Enter");
        }
        SearchStatus::NoResults => {
            render_centered_message(frame, area, "Ничего не найдено");
        }
        SearchStatus::Error(message) => {
            let display = format!("Ошибка: {message}");
            render_centered_message(frame, area, &display);
        }
        SearchStatus::Offline => {}
        SearchStatus::HasResults => {
            render_results_table(frame, area, state);
        }
    }
}

fn render_centered_message(frame: &mut Frame, area: Rect, message: &str) {
    let paragraph = Paragraph::new(message)
        .style(Style::default().fg(theme::MUTED))
        .alignment(Alignment::Center)
        .block(Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(theme::MUTED))
            .title(Span::styled("Results", Style::default().fg(theme::BLUE))));
    frame.render_widget(paragraph, area);
}

fn render_results_table(frame: &mut Frame, area: Rect, state: &mut SearchTabState) {
    let header = Row::new(vec!["Score", "Type", "Context"])
        .style(
            Style::default()
                .fg(theme::BLUE)
                .add_modifier(Modifier::BOLD),
        )
        .bottom_margin(0);

    let rows: Vec<Row> = state
        .results
        .iter()
        .map(|result| {
            let score_color = if result.score >= 0.4 {
                theme::GREEN
            } else {
                theme::MUTED
            };
            let truncated_context = truncate_to_width(&result.context, 80);
            Row::new(vec![
                Span::styled(
                    format!("{:.2}", result.score),
                    Style::default().fg(score_color),
                ),
                Span::styled(
                    result.memory_type.clone(),
                    Style::default().fg(type_color(&result.memory_type)),
                ),
                Span::styled(truncated_context, Style::default().fg(theme::TEXT)),
            ])
        })
        .collect();

    let widths = [
        Constraint::Length(6),
        Constraint::Length(14),
        Constraint::Fill(1),
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
                    format!("Results ({})", state.results.len()),
                    Style::default().fg(theme::BLUE),
                )),
        )
        .column_spacing(1);

    let mut table_state = TableState::default().with_selected(Some(state.selected));
    frame.render_stateful_widget(table, area, &mut table_state);

    let scrollbar = Scrollbar::new(ScrollbarOrientation::VerticalRight);
    let mut scrollbar_state =
        ScrollbarState::new(state.results.len()).position(state.selected);
    frame.render_stateful_widget(scrollbar, area, &mut scrollbar_state);
}

fn render_search_footer(frame: &mut Frame, area: Rect, state: &mut SearchTabState) {
    if state.expired_status_message() {
        state.status_message = None;
    }
    if let Some((message, _)) = &state.status_message {
        let line = Line::from(vec![
            Span::styled(" ", Style::default()),
            Span::styled(message.clone(), Style::default().fg(theme::GREEN)),
        ]);
        frame.render_widget(Paragraph::new(line), area);
        return;
    }
    let spans: Vec<Span> = if state.input_active {
        footer_hints(&[("Enter", "search"), ("Esc", "clear/exit")])
    } else {
        footer_hints(&[
            ("j/k", "scroll"),
            ("Enter", "details"),
            ("J", "judge"),
            ("s", "save"),
            ("/", "edit query"),
            ("Esc", "back"),
        ])
    };
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
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

fn render_detail_popup(frame: &mut Frame, area: Rect, state: &SearchTabState) {
    let result = &state.results[state.selected];
    let popup_area = centered_rect(80, 60, area);
    frame.render_widget(Clear, popup_area);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::PURPLE))
        .title(Span::styled(
            format!(" {} ", result.memory_type),
            Style::default()
                .fg(type_color(&result.memory_type))
                .add_modifier(Modifier::BOLD),
        ));
    let inner = block.inner(popup_area);
    frame.render_widget(block, popup_area);

    let [meta_area, context_area, action_area, result_area] = Layout::vertical([
        Constraint::Length(2),
        Constraint::Fill(1),
        Constraint::Fill(1),
        Constraint::Fill(1),
    ])
    .areas(inner);

    let meta_line = Line::from(vec![
        Span::styled("Score: ", Style::default().fg(theme::BLUE)),
        Span::styled(format!("{:.4}", result.score), Style::default().fg(theme::GREEN)),
    ]);
    frame.render_widget(Paragraph::new(meta_line), meta_area);

    render_section(frame, context_area, "Context", &result.context);
    render_section(frame, action_area, "Action", &result.action);
    render_section(frame, result_area, "Result", &result.result);
}

fn render_section(frame: &mut Frame, area: Rect, title: &str, content: &str) {
    let display = if content.is_empty() { "(empty)" } else { content };
    let paragraph = Paragraph::new(display.to_string())
        .style(Style::default().fg(theme::TEXT))
        .wrap(Wrap { trim: false })
        .block(Block::default()
            .borders(Borders::TOP)
            .border_style(Style::default().fg(theme::MUTED))
            .title(Span::styled(title, Style::default().fg(theme::BLUE))));
    frame.render_widget(paragraph, area);
}

fn type_color(memory_type: &str) -> ratatui::style::Color {
    match memory_type {
        "decision" => theme::PURPLE,
        "pattern" => theme::BLUE,
        "bugfix" | "antipattern" => theme::RED,
        "context" => theme::MUTED,
        "insight" => theme::GREEN,
        _ => theme::TEXT,
    }
}

fn truncate_to_width(text: &str, max_chars: usize) -> String {
    let single_line: String = text.chars().take_while(|c| *c != '\n').collect();
    if single_line.chars().count() <= max_chars { return single_line; }
    let truncated: String = single_line.chars().take(max_chars.saturating_sub(3)).collect();
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
