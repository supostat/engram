use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Row, Table};
use ratatui::Frame;

use crate::data::QTableEntry;
use crate::theme;

const STATES: &[&str] = &["query", "research", "brainstorm", "debugging"];

struct RouterLevel {
    title: &'static str,
    actions: &'static [&'static str],
}

const ROUTER_LEVELS: &[RouterLevel] = &[
    RouterLevel {
        title: "Search Strategy",
        actions: &["vector_only", "sparse_only", "hybrid", "hyde"],
    },
    RouterLevel {
        title: "LLM Selection",
        actions: &["default", "fast", "quality"],
    },
    RouterLevel {
        title: "Contextualization",
        actions: &["none", "light", "full"],
    },
    RouterLevel {
        title: "Proactivity",
        actions: &["passive", "nudge", "proactive"],
    },
];

pub fn render_qlearning_tab(frame: &mut Frame, area: Rect, entries: &[QTableEntry]) {
    if entries.is_empty() {
        render_empty_state(frame, area);
        return;
    }

    let [top_row, bottom_row] = Layout::vertical([
        Constraint::Percentage(50),
        Constraint::Percentage(50),
    ])
    .areas(area);

    let [top_left, top_right] =
        Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
            .areas(top_row);

    let [bottom_left, bottom_right] =
        Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
            .areas(bottom_row);

    let panels = [top_left, top_right, bottom_left, bottom_right];
    for (level_index, panel_area) in panels.iter().enumerate() {
        render_level_panel(frame, *panel_area, level_index as i32, entries);
    }
}

fn render_empty_state(frame: &mut Frame, area: Rect) {
    let message = Paragraph::new("Q-таблица пуста. Используйте memory_judge для обучения роутера.")
        .style(Style::default().fg(theme::MUTED))
        .alignment(Alignment::Center)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(theme::MUTED))
                .title(Span::styled(
                    "Q-Learning",
                    Style::default().fg(theme::BLUE),
                )),
        );
    frame.render_widget(message, area);
}

fn render_level_panel(frame: &mut Frame, area: Rect, level: i32, entries: &[QTableEntry]) {
    let router = &ROUTER_LEVELS[level as usize];
    let level_entries: Vec<&QTableEntry> = entries.iter().filter(|e| e.router_level == level).collect();
    let total_updates: i64 = level_entries.iter().map(|e| e.update_count).sum();

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::MUTED))
        .title(Span::styled(
            router.title,
            Style::default().fg(theme::BLUE),
        ));

    let [table_area, footer_area] = Layout::vertical([
        Constraint::Fill(1),
        Constraint::Length(1),
    ])
    .areas(block.inner(area));

    frame.render_widget(block, area);

    let header_cells: Vec<Span> = std::iter::once(Span::styled(
        "",
        Style::default().fg(theme::MUTED),
    ))
    .chain(router.actions.iter().map(|action| {
        Span::styled(
            format!("{action:>10}"),
            Style::default()
                .fg(theme::BLUE)
                .add_modifier(Modifier::BOLD),
        )
    }))
    .collect();

    let header = Row::new(header_cells).bottom_margin(0);

    let rows: Vec<Row> = STATES
        .iter()
        .map(|state| {
            let cells: Vec<Span> = std::iter::once(Span::styled(
                format!("{state:<12}"),
                Style::default().fg(theme::TEXT),
            ))
            .chain(router.actions.iter().map(|action| {
                let value = find_value(&level_entries, state, action);
                Span::styled(format!("{value:>10.2}"), Style::default().fg(value_color(value)))
            }))
            .collect();
            Row::new(cells)
        })
        .collect();

    let mut widths: Vec<Constraint> = vec![Constraint::Length(12)];
    widths.extend(router.actions.iter().map(|_| Constraint::Length(10)));

    let table = Table::new(rows, &widths).header(header).column_spacing(1);
    frame.render_widget(table, table_area);

    let footer = Paragraph::new(Line::from(vec![
        Span::styled(" Updates: ", Style::default().fg(theme::MUTED)),
        Span::styled(
            total_updates.to_string(),
            Style::default().fg(theme::PURPLE),
        ),
    ]));
    frame.render_widget(footer, footer_area);
}

fn find_value(entries: &[&QTableEntry], state: &str, action: &str) -> f64 {
    entries
        .iter()
        .find(|e| e.state == state && e.action == action)
        .map(|e| e.value)
        .unwrap_or(0.0)
}

fn value_color(value: f64) -> ratatui::style::Color {
    if value > 0.5 {
        theme::GREEN
    } else if value > 0.2 {
        theme::PURPLE
    } else if value > 0.0 {
        theme::MUTED
    } else {
        theme::SURFACE
    }
}
