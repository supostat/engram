mod keys;

use std::io;
use std::time::{Duration, Instant};

use ratatui::crossterm::event::{self, Event, KeyEventKind};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Tabs};
use ratatui::DefaultTerminal;
use ratatui::Frame;

use crate::data::{DashboardStats, DatabaseReader, QTableEntry, SocketClient, load_stats};
use crate::overlays::{
    self, ConfirmDialog, FilterState, StatusMessage,
    render_confirm_dialog, render_filter_popup, render_status_message,
};
use crate::tabs::{
    render_memories_tab, render_models_tab, render_qlearning_tab, render_search_tab,
    render_status_tab, MemoriesTabState, ModelsTabState, SearchStatus, SearchTabState,
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

pub struct App {
    tab: Tab,
    database: DatabaseReader,
    database_path: String,
    stats: DashboardStats,
    q_table_entries: Vec<QTableEntry>,
    memories_state: MemoriesTabState,
    search_state: SearchTabState,
    socket: Option<SocketClient>,
    models_state: ModelsTabState,
    should_quit: bool,
    last_refresh: Instant,
    status_message: Option<StatusMessage>,
    confirm_dialog: Option<ConfirmDialog>,
    filter_state: Option<FilterState>,
    consolidation_preview: Option<String>,
}

impl App {
    pub fn new(database_path: &str, models_path: &str, socket_path: &str) -> io::Result<Self> {
        let database = DatabaseReader::new(database_path)?;
        let stats = load_stats(&database, models_path);
        let q_table_entries = database.q_table_entries();
        let memories = database.list_memories(500);
        let models = database.models_info(models_path);
        let mut memories_state = MemoriesTabState::new();
        memories_state.memories = memories;
        let mut models_state = ModelsTabState::new(models_path.to_string());
        models_state.models = models;
        let socket = SocketClient::connect(socket_path).ok();
        let mut search_state = SearchTabState::new();
        if socket.is_none() {
            search_state.status = SearchStatus::Offline;
        }
        Ok(Self {
            tab: Tab::Status,
            database,
            database_path: database_path.to_string(),
            stats,
            q_table_entries,
            memories_state,
            search_state,
            socket,
            models_state,
            should_quit: false,
            last_refresh: Instant::now(),
            status_message: None,
            confirm_dialog: None,
            filter_state: None,
            consolidation_preview: None,
        })
    }

    pub fn run(&mut self, terminal: &mut DefaultTerminal) -> io::Result<()> {
        while !self.should_quit {
            terminal.draw(|frame| self.render(frame))?;
            self.handle_events()?;
            self.expire_status_message();
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
        self.stats = load_stats(&self.database, &self.models_state.models_path);
        self.q_table_entries = self.database.q_table_entries();
        self.memories_state.memories = self.database.list_memories(500);
        self.memories_state.clamp_selection();
        self.models_state.models = self.database.models_info(&self.models_state.models_path);
        self.models_state.clamp_selection();
        self.last_refresh = Instant::now();
    }

    fn expire_status_message(&mut self) {
        if let Some(ref message) = self.status_message
            && message.is_expired()
        {
            self.status_message = None;
        }
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
        self.render_footer(frame, footer_area, current_tab);

        if let Some(ref dialog) = self.confirm_dialog {
            render_confirm_dialog(frame, frame.area(), dialog);
        }
        if let Some(ref filter) = self.filter_state {
            render_filter_popup(frame, frame.area(), filter);
        }
        if let Some(ref preview) = self.consolidation_preview {
            overlays::render_consolidation_preview(frame, frame.area(), preview);
        }
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
            Tab::Search => render_search_tab(frame, area, &mut self.search_state),
            Tab::QLearning => render_qlearning_tab(frame, area, &self.q_table_entries),
            Tab::Models => render_models_tab(frame, area, &mut self.models_state),
        }
    }

    fn render_footer(&self, frame: &mut Frame, area: Rect, tab: Tab) {
        if let Some(ref message) = self.status_message {
            render_status_message(frame, area, message);
            return;
        }
        render_footer(frame, area, tab);
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
    if tab == Tab::Search {
        return;
    }
    let mut hints: Vec<(&str, &str)> =
        vec![("q", "quit"), ("Tab", "switch"), ("1-5", "jump"), ("r", "refresh")];
    if tab == Tab::Memories {
        hints.extend([("j", "judge"), ("d", "delete"), ("f", "filter"), ("/", "search")]);
    }
    if tab == Tab::Status {
        hints.extend([("e", "export"), ("c", "consolidate")]);
    }
    let spans = footer_hint_spans(&hints);
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

fn footer_hint_spans(pairs: &[(&str, &str)]) -> Vec<Span<'static>> {
    let mut spans = Vec::with_capacity(pairs.len() * 2);
    for (key, description) in pairs {
        let prefix = if spans.is_empty() { " " } else { "  " };
        spans.push(Span::styled(format!("{prefix}{key}"), Style::default().fg(theme::PURPLE)));
        spans.push(Span::styled(format!(": {description}"), Style::default().fg(theme::MUTED)));
    }
    spans
}

fn header_block() -> Block<'static> {
    Block::default()
        .borders(Borders::BOTTOM)
        .border_style(Style::default().fg(theme::MUTED))
}

