use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Bar, BarChart, BarGroup, Block, Borders, Paragraph, Row, Table};

use crate::data::DashboardStats;
use crate::theme;

pub fn render_status_tab(frame: &mut Frame, area: Rect, stats: &DashboardStats) {
    let hints_height = if stats.hints.is_empty() {
        0
    } else {
        stats.hints.len() as u16 + 2
    };

    let [cards_area, distributions_area, histogram_area, hints_area, table_area] =
        Layout::vertical([
            Constraint::Length(5),
            Constraint::Length(10),
            Constraint::Length(6),
            Constraint::Length(hints_height),
            Constraint::Fill(1),
        ])
        .areas(area);

    render_stat_cards(frame, cards_area, stats);
    render_distributions(frame, distributions_area, stats);
    render_score_histogram(frame, histogram_area, stats);
    if !stats.hints.is_empty() {
        render_hints(frame, hints_area, &stats.hints);
    }
    render_recent_memories(frame, table_area, stats);
}

fn render_stat_cards(frame: &mut Frame, area: Rect, stats: &DashboardStats) {
    let [mem_area, idx_area, judged_area, score_area] = Layout::horizontal([
        Constraint::Percentage(25),
        Constraint::Percentage(25),
        Constraint::Percentage(25),
        Constraint::Percentage(25),
    ])
    .areas(area);

    render_single_card(frame, mem_area, "Memories", &stats.memory_count.to_string());
    render_single_card(frame, idx_area, "Indexed", &stats.indexed_count.to_string());
    render_single_card(frame, judged_area, "Judged", &stats.feedback_judged.to_string());
    render_single_card(frame, score_area, "Avg Score", &format!("{:.2}", stats.average_score));
}

fn render_single_card(frame: &mut Frame, area: Rect, title: &str, value: &str) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::MUTED))
        .title(Span::styled(title, Style::default().fg(theme::BLUE)));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let text = Paragraph::new(Line::from(Span::styled(
        value,
        Style::default()
            .fg(theme::PURPLE)
            .add_modifier(Modifier::BOLD),
    )))
    .alignment(ratatui::layout::Alignment::Center);
    let centered = center_vertically(inner);
    frame.render_widget(text, centered);
}

fn center_vertically(area: Rect) -> Rect {
    if area.height < 1 {
        return area;
    }
    let offset = area.height / 2;
    Rect::new(area.x, area.y + offset, area.width, 1)
}

fn render_distributions(frame: &mut Frame, area: Rect, stats: &DashboardStats) {
    let [types_area, projects_area] =
        Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)]).areas(area);

    render_distribution_bars(frame, types_area, "Types", &stats.type_distribution);
    render_distribution_bars(frame, projects_area, "Projects", &stats.project_distribution);
}

fn render_distribution_bars(
    frame: &mut Frame,
    area: Rect,
    title: &str,
    distribution: &[(String, usize)],
) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::MUTED))
        .title(Span::styled(title, Style::default().fg(theme::BLUE)));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if distribution.is_empty() {
        let empty = Paragraph::new("No data")
            .style(Style::default().fg(theme::MUTED))
            .alignment(ratatui::layout::Alignment::Center);
        frame.render_widget(empty, inner);
        return;
    }
    let max_count = distribution.iter().map(|(_, count)| *count).max().unwrap_or(1);
    let lines: Vec<Line> = distribution
        .iter()
        .take(inner.height as usize)
        .map(|(label, count)| format_bar_line(label, *count, max_count, inner.width))
        .collect();

    let paragraph = Paragraph::new(lines);
    frame.render_widget(paragraph, inner);
}

fn format_bar_line(label: &str, count: usize, max_count: usize, total_width: u16) -> Line<'static> {
    let label_width = 14;
    let count_str = count.to_string();
    let bar_max_width = total_width.saturating_sub(label_width + count_str.len() as u16 + 2);
    let bar_length = if max_count > 0 {
        (count as u16 * bar_max_width) / max_count as u16
    } else {
        0
    }
    .max(1);

    let bar: String = "█".repeat(bar_length as usize);
    let padded_label = format!("{label:>12} ", label = &label[..label.len().min(12)]);

    Line::from(vec![
        Span::styled(padded_label, Style::default().fg(theme::MUTED)),
        Span::styled(bar, Style::default().fg(theme::PURPLE)),
        Span::styled(format!(" {count_str}"), Style::default().fg(theme::TEXT)),
    ])
}

fn render_score_histogram(frame: &mut Frame, area: Rect, stats: &DashboardStats) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::MUTED))
        .title(Span::styled(
            "Score Distribution",
            Style::default().fg(theme::BLUE),
        ));

    let labels = ["0", ".1", ".2", ".3", ".4", ".5", ".6", ".7", ".8", ".9"];
    let bars: Vec<Bar> = stats
        .score_distribution
        .iter()
        .enumerate()
        .map(|(index, &count)| {
            Bar::default()
                .value(count as u64)
                .label(Line::from(labels[index]))
                .style(Style::default().fg(theme::DARK_PURPLE))
        })
        .collect();

    let chart = BarChart::default()
        .block(block)
        .data(BarGroup::default().bars(&bars))
        .bar_width(3)
        .bar_gap(1)
        .bar_style(Style::default().fg(theme::PURPLE))
        .value_style(Style::default().fg(theme::TEXT));

    frame.render_widget(chart, area);
}

fn render_hints(frame: &mut Frame, area: Rect, hints: &[String]) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::MUTED))
        .title(Span::styled("Hints", Style::default().fg(theme::AMBER)));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let lines: Vec<Line> = hints
        .iter()
        .map(|hint| {
            Line::from(Span::styled(
                format!("💡 {hint}"),
                Style::default().fg(theme::AMBER),
            ))
        })
        .collect();

    frame.render_widget(Paragraph::new(lines), inner);
}

fn render_recent_memories(frame: &mut Frame, area: Rect, stats: &DashboardStats) {
    let header = Row::new(vec!["Type", "Project", "Context", "Score"])
        .style(
            Style::default()
                .fg(theme::BLUE)
                .add_modifier(Modifier::BOLD),
        )
        .bottom_margin(0);

    let rows: Vec<Row> = stats
        .recent_memories
        .iter()
        .map(|memory| {
            let score_color = if memory.score >= 0.5 {
                theme::GREEN
            } else {
                theme::MUTED
            };
            Row::new(vec![
                Span::styled(memory.memory_type.clone(), Style::default().fg(theme::PURPLE)),
                Span::styled(memory.project_display(), Style::default().fg(theme::MUTED)),
                Span::styled(memory.context.clone(), Style::default().fg(theme::TEXT)),
                Span::styled(format!("{:.2}", memory.score), Style::default().fg(score_color)),
            ])
        })
        .collect();

    let widths = [
        Constraint::Length(14),
        Constraint::Length(14),
        Constraint::Fill(1),
        Constraint::Length(6),
    ];

    let table = Table::new(rows, widths)
        .header(header)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(theme::MUTED))
                .title(Span::styled(
                    "Recent Memories",
                    Style::default().fg(theme::BLUE),
                )),
        )
        .column_spacing(1);

    frame.render_widget(table, area);
}
