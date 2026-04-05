use std::io;
use std::time::{Duration, Instant};

use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Tabs};
use ratatui::DefaultTerminal;
use ratatui::Frame;

use crate::data::{DatabaseReader, MemorySummary, QTableEntry};
use crate::tabs::{
    render_memories_tab, render_models_tab, render_qlearning_tab, render_status_tab,
    MemoriesTabState, ModelsTabState,
};
use crate::theme;

const POLL_TIMEOUT: Duration = Duration::from_millis(250);
const REFRESH_INTERVAL: Duration = Duration::from_secs(2);

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Status,
    Memories,
    Search,
    QLearning,
    Models,
}

impl Tab {
    const ALL: [Tab; 5] = [
        Tab::Status,
        Tab::Memories,
        Tab::Search,
        Tab::QLearning,
        Tab::Models,
    ];

    fn title(self) -> &'static str {
        match self {
            Tab::Status => "Status",
            Tab::Memories => "Memories",
            Tab::Search => "Search",
            Tab::QLearning => "Q-Learning",
            Tab::Models => "Models",
        }
    }

    fn index(self) -> usize {
        Tab::ALL.iter().position(|&tab| tab == self).unwrap_or(0)
    }
}

pub struct DashboardStats {
    pub memory_count: usize,
    pub indexed_count: usize,
    pub average_score: f64,
    pub type_distribution: Vec<(String, usize)>,
    pub project_distribution: Vec<(String, usize)>,
    pub score_distribution: Vec<usize>,
    pub feedback_judged: usize,
    pub recent_memories: Vec<MemorySummary>,
}

pub struct App {
    tab: Tab,
    database: DatabaseReader,
    database_path: String,
    stats: DashboardStats,
    q_table_entries: Vec<QTableEntry>,
    memories_state: MemoriesTabState,
    models_state: ModelsTabState,
    should_quit: bool,
    last_refresh: Instant,
}

impl App {
    pub fn new(database_path: &str, models_path: &str) -> io::Result<Self> {
        let database = DatabaseReader::new(database_path)?;
        let stats = load_stats(&database);
        let q_table_entries = database.q_table_entries();
        let memories = database.list_memories(500);
        let models = database.models_info(models_path);
        let mut memories_state = MemoriesTabState::new();
        memories_state.memories = memories;
        let mut models_state = ModelsTabState::new(models_path.to_string());
        models_state.models = models;
        Ok(Self {
            tab: Tab::Status,
            database,
            database_path: database_path.to_string(),
            stats,
            q_table_entries,
            memories_state,
            models_state,
            should_quit: false,
            last_refresh: Instant::now(),
        })
    }

    pub fn run(&mut self, terminal: &mut DefaultTerminal) -> io::Result<()> {
        while !self.should_quit {
            terminal.draw(|frame| self.render(frame))?;
            self.handle_events()?;
            self.refresh_if_stale();
        }
        Ok(())
    }

    fn handle_events(&mut self) -> io::Result<()> {
        if !event::poll(POLL_TIMEOUT)? {
            return Ok(());
        }
        let Event::Key(key) = event::read()? else {
            return Ok(());
        };
        if key.kind != KeyEventKind::Press {
            return Ok(());
        }
        self.handle_key(key.code);
        Ok(())
    }

    fn handle_key(&mut self, code: KeyCode) {
        if self.tab == Tab::Memories && self.memories_state.detail_open {
            match code {
                KeyCode::Esc | KeyCode::Enter => self.memories_state.close_detail(),
                _ => {}
            }
            return;
        }

        if self.tab == Tab::Memories {
            match code {
                KeyCode::Char('j') | KeyCode::Down => {
                    self.memories_state.move_down();
                    return;
                }
                KeyCode::Char('k') | KeyCode::Up => {
                    self.memories_state.move_up();
                    return;
                }
                KeyCode::Enter => {
                    self.memories_state.toggle_detail();
                    return;
                }
                _ => {}
            }
        }

        match code {
            KeyCode::Char('q') | KeyCode::Esc => self.should_quit = true,
            KeyCode::Tab => self.next_tab(),
            KeyCode::BackTab => self.previous_tab(),
            KeyCode::Char('1') => self.tab = Tab::Status,
            KeyCode::Char('2') => self.tab = Tab::Memories,
            KeyCode::Char('3') => self.tab = Tab::Search,
            KeyCode::Char('4') => self.tab = Tab::QLearning,
            KeyCode::Char('5') => self.tab = Tab::Models,
            KeyCode::Char('r') => self.force_refresh(),
            _ => {}
        }
    }

    fn next_tab(&mut self) {
        let current = self.tab.index();
        let next = (current + 1) % Tab::ALL.len();
        self.tab = Tab::ALL[next];
    }

    fn previous_tab(&mut self) {
        let current = self.tab.index();
        let previous = current.checked_sub(1).unwrap_or(Tab::ALL.len() - 1);
        self.tab = Tab::ALL[previous];
    }

    fn refresh_if_stale(&mut self) {
        if self.last_refresh.elapsed() >= REFRESH_INTERVAL {
            self.force_refresh();
        }
    }

    fn force_refresh(&mut self) {
        self.stats = load_stats(&self.database);
        self.q_table_entries = self.database.q_table_entries();
        self.memories_state.memories = self.database.list_memories(500);
        self.memories_state.clamp_selection();
        self.models_state.models = self.database.models_info(&self.models_state.models_path);
        self.last_refresh = Instant::now();
    }

    fn render(&mut self, frame: &mut Frame) {
        let [header_area, content_area, footer_area] = Layout::vertical([
            Constraint::Length(3),
            Constraint::Fill(1),
            Constraint::Length(1),
        ])
        .areas(frame.area());

        let current_tab = self.tab;
        self.render_header(frame, header_area);
        self.render_content(frame, content_area);
        render_footer(frame, footer_area, current_tab);
    }

    fn render_header(&self, frame: &mut Frame, area: Rect) {
        let path_width = self.database_path.len() as u16 + 2;
        let [logo_area, tabs_area, path_area] = Layout::horizontal([
            Constraint::Length(10),
            Constraint::Fill(1),
            Constraint::Length(path_width),
        ])
        .areas(area);

        render_logo(frame, logo_area);
        self.render_tabs(frame, tabs_area);
        render_database_path(frame, path_area, &self.database_path);
    }

    fn render_tabs(&self, frame: &mut Frame, area: Rect) {
        let titles: Vec<Line> = Tab::ALL.iter().map(|tab| Line::from(tab.title())).collect();
        let tabs_widget = Tabs::new(titles)
            .select(self.tab.index())
            .style(Style::default().fg(theme::MUTED))
            .highlight_style(Style::default().fg(theme::PURPLE).add_modifier(Modifier::BOLD))
            .block(header_block());
        frame.render_widget(tabs_widget, area);
    }

    fn render_content(&mut self, frame: &mut Frame, area: Rect) {
        match self.tab {
            Tab::Status => render_status_tab(frame, area, &self.stats),
            Tab::Memories => render_memories_tab(frame, area, &mut self.memories_state),
            Tab::QLearning => render_qlearning_tab(frame, area, &self.q_table_entries),
            Tab::Models => render_models_tab(frame, area, &self.models_state),
            _ => render_placeholder(frame, area, self.tab.title()),
        }
    }
}

fn render_logo(frame: &mut Frame, area: Rect) {
    let logo = Paragraph::new(Line::from(Span::styled(
        " engram",
        Style::default().fg(theme::PURPLE).add_modifier(Modifier::BOLD),
    )))
    .block(header_block());
    frame.render_widget(logo, area);
}

fn render_database_path(frame: &mut Frame, area: Rect, path: &str) {
    let display = Paragraph::new(Line::from(Span::styled(
        path,
        Style::default().fg(theme::MUTED),
    )))
    .alignment(ratatui::layout::Alignment::Right)
    .block(header_block());
    frame.render_widget(display, area);
}

fn render_footer(frame: &mut Frame, area: Rect, tab: Tab) {
    let mut spans = vec![
        Span::styled(" q", Style::default().fg(theme::PURPLE)),
        Span::styled(": quit", Style::default().fg(theme::MUTED)),
        Span::styled("  Tab", Style::default().fg(theme::PURPLE)),
        Span::styled(": switch", Style::default().fg(theme::MUTED)),
        Span::styled("  1-5", Style::default().fg(theme::PURPLE)),
        Span::styled(": jump", Style::default().fg(theme::MUTED)),
        Span::styled("  r", Style::default().fg(theme::PURPLE)),
        Span::styled(": refresh", Style::default().fg(theme::MUTED)),
    ];
    if tab == Tab::Memories {
        spans.extend([
            Span::styled("  j/k", Style::default().fg(theme::PURPLE)),
            Span::styled(": scroll", Style::default().fg(theme::MUTED)),
            Span::styled("  Enter", Style::default().fg(theme::PURPLE)),
            Span::styled(": detail", Style::default().fg(theme::MUTED)),
        ]);
    }
    let footer = Paragraph::new(Line::from(spans));
    frame.render_widget(footer, area);
}

fn render_placeholder(frame: &mut Frame, area: Rect, tab_name: &str) {
    let paragraph = Paragraph::new(format!("{tab_name} — coming soon"))
        .style(Style::default().fg(theme::MUTED))
        .alignment(ratatui::layout::Alignment::Center)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(theme::MUTED)),
        );
    frame.render_widget(paragraph, area);
}

fn header_block() -> Block<'static> {
    Block::default()
        .borders(Borders::BOTTOM)
        .border_style(Style::default().fg(theme::MUTED))
}

fn load_stats(database: &DatabaseReader) -> DashboardStats {
    let (_feedback_searched, feedback_judged) = database.feedback_stats();
    DashboardStats {
        memory_count: database.memory_count(),
        indexed_count: database.indexed_count(),
        average_score: database.average_score(),
        type_distribution: database.type_distribution(),
        project_distribution: database.project_distribution(),
        score_distribution: database.score_distribution(),
        feedback_judged,
        recent_memories: database.recent_memories(20),
    }
}
