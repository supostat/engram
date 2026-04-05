use std::time::Instant;

use ratatui::crossterm::event::KeyCode;

use crate::data::SocketClient;

pub enum SearchKeyAction {
    Handled,
    Quit,
    NextTab,
    PreviousTab,
}

pub struct SearchResult {
    pub id: String,
    pub memory_type: String,
    pub context: String,
    pub action: String,
    pub result: String,
    pub score: f64,
}

pub enum SearchStatus {
    Idle,
    HasResults,
    NoResults,
    Error(String),
    Offline,
}

pub struct SearchTabState {
    pub query: String,
    pub cursor_position: usize,
    pub results: Vec<SearchResult>,
    pub selected: usize,
    pub status: SearchStatus,
    pub detail_open: bool,
    pub input_active: bool,
    pub status_message: Option<(String, Instant)>,
}

impl SearchTabState {
    pub fn new() -> Self {
        Self {
            query: String::new(),
            cursor_position: 0,
            results: Vec::new(),
            selected: 0,
            status: SearchStatus::Idle,
            detail_open: false,
            input_active: true,
            status_message: None,
        }
    }

    pub fn set_status_message(&mut self, message: String) {
        self.status_message = Some((message, Instant::now()));
    }

    pub fn expired_status_message(&self) -> bool {
        self.status_message
            .as_ref()
            .is_some_and(|(_, ts)| ts.elapsed().as_secs() >= 3)
    }

    pub fn insert_char(&mut self, character: char) {
        self.query.insert(self.cursor_position, character);
        self.cursor_position += character.len_utf8();
    }

    pub fn delete_char_before_cursor(&mut self) {
        if self.cursor_position == 0 {
            return;
        }
        let before_cursor = &self.query[..self.cursor_position];
        let removed = before_cursor
            .chars()
            .next_back()
            .expect("cursor_position > 0 guarantees at least one char");
        self.cursor_position -= removed.len_utf8();
        self.query.remove(self.cursor_position);
    }

    pub fn move_cursor_left(&mut self) {
        if self.cursor_position == 0 {
            return;
        }
        let before = &self.query[..self.cursor_position];
        let previous_char = before
            .chars()
            .next_back()
            .expect("cursor_position > 0 guarantees at least one char");
        self.cursor_position -= previous_char.len_utf8();
    }

    pub fn move_cursor_right(&mut self) {
        if self.cursor_position >= self.query.len() {
            return;
        }
        let after = &self.query[self.cursor_position..];
        let next_char = after
            .chars()
            .next()
            .expect("cursor_position < len guarantees at least one char");
        self.cursor_position += next_char.len_utf8();
    }

    pub fn move_result_up(&mut self) {
        self.selected = self.selected.saturating_sub(1);
    }

    pub fn move_result_down(&mut self) {
        if !self.results.is_empty() {
            self.selected = (self.selected + 1).min(self.results.len() - 1);
        }
    }

    pub fn toggle_detail(&mut self) {
        if !self.results.is_empty() {
            self.detail_open = !self.detail_open;
        }
    }

    pub fn close_detail(&mut self) {
        self.detail_open = false;
    }

    pub fn clear(&mut self) {
        self.query.clear();
        self.cursor_position = 0;
        self.results.clear();
        self.selected = 0;
        self.status = SearchStatus::Idle;
        self.detail_open = false;
        self.input_active = true;
    }

    pub fn handle_key(
        &mut self,
        code: KeyCode,
        socket: &mut Option<SocketClient>,
    ) -> SearchKeyAction {
        if self.detail_open {
            match code {
                KeyCode::Esc | KeyCode::Enter => self.close_detail(),
                _ => {}
            }
            return SearchKeyAction::Handled;
        }
        if self.input_active {
            self.handle_input_key(code, socket)
        } else {
            self.handle_results_key(code, socket)
        }
    }

    fn handle_input_key(
        &mut self,
        code: KeyCode,
        socket: &mut Option<SocketClient>,
    ) -> SearchKeyAction {
        match code {
            KeyCode::Enter => self.execute_search(socket),
            KeyCode::Backspace => self.delete_char_before_cursor(),
            KeyCode::Left => self.move_cursor_left(),
            KeyCode::Right => self.move_cursor_right(),
            KeyCode::Esc if self.query.is_empty() && self.results.is_empty() => {
                return SearchKeyAction::Quit;
            }
            KeyCode::Esc => self.clear(),
            KeyCode::Down if !self.results.is_empty() => self.input_active = false,
            KeyCode::Tab => return SearchKeyAction::NextTab,
            KeyCode::BackTab => return SearchKeyAction::PreviousTab,
            KeyCode::Char(character) => self.insert_char(character),
            _ => {}
        }
        SearchKeyAction::Handled
    }

    fn handle_results_key(
        &mut self,
        code: KeyCode,
        socket: &mut Option<SocketClient>,
    ) -> SearchKeyAction {
        match code {
            KeyCode::Char('j') | KeyCode::Down => self.move_result_down(),
            KeyCode::Char('k') | KeyCode::Up => self.move_result_up(),
            KeyCode::Enter => self.toggle_detail(),
            KeyCode::Char('/') | KeyCode::Esc => self.input_active = true,
            KeyCode::Tab => return SearchKeyAction::NextTab,
            KeyCode::BackTab => return SearchKeyAction::PreviousTab,
            KeyCode::Char('q') => return SearchKeyAction::Quit,
            KeyCode::Char('J') => self.judge_selected(socket),
            KeyCode::Char('s') => self.store_search_as_memory(socket),
            _ => {}
        }
        SearchKeyAction::Handled
    }

    pub fn execute_search(&mut self, socket: &mut Option<SocketClient>) {
        let Some(client) = socket.as_mut() else {
            self.status = SearchStatus::Offline;
            return;
        };
        if self.query.trim().is_empty() {
            return;
        }

        let params = serde_json::json!({
            "query": self.query,
            "limit": 20,
        });

        match client.call("memory_search", params) {
            Ok(data) => self.parse_results(data),
            Err(error) => {
                self.results.clear();
                self.status = SearchStatus::Error(error.to_string());
            }
        }
    }

    fn parse_results(&mut self, data: serde_json::Value) {
        self.results.clear();
        self.selected = 0;
        self.detail_open = false;

        let entries = match data.as_array() {
            Some(array) => array,
            None => {
                self.status = SearchStatus::NoResults;
                return;
            }
        };

        for entry in entries {
            self.results.push(SearchResult {
                id: json_string(&entry["id"]),
                memory_type: json_string(&entry["memory_type"]),
                context: json_string(&entry["context"]),
                action: json_string(&entry["action"]),
                result: json_string(&entry["result"]),
                score: entry["score"].as_f64().unwrap_or(0.0),
            });
        }

        self.status = if self.results.is_empty() {
            SearchStatus::NoResults
        } else {
            SearchStatus::HasResults
        };
    }

    fn judge_selected(&mut self, socket: &mut Option<SocketClient>) {
        if self.results.is_empty() {
            return;
        }
        let Some(client) = socket.as_mut() else {
            self.set_status_message("Socket offline".to_string());
            return;
        };
        let memory_id = self.results[self.selected].id.clone();
        let query = self.query.clone();
        let params = serde_json::json!({
            "memory_id": memory_id,
            "query": query,
        });
        match client.call("memory_judge", params) {
            Ok(data) => {
                let score = data["score"].as_f64().unwrap_or(0.0);
                self.results[self.selected].score = score;
                self.set_status_message(format!("Judged: score {score:.2}"));
            }
            Err(error) => self.set_status_message(format!("Judge error: {error}")),
        }
    }

    fn store_search_as_memory(&mut self, socket: &mut Option<SocketClient>) {
        if self.query.trim().is_empty() {
            return;
        }
        let Some(client) = socket.as_mut() else {
            self.set_status_message("Socket offline".to_string());
            return;
        };
        let top_context = self
            .results
            .first()
            .map_or("no results".to_string(), |r| r.context.clone());
        let params = serde_json::json!({
            "memory_type": "context",
            "context": format!("Searched for: {}", self.query),
            "action": format!("Found {} results", self.results.len()),
            "result": format!("Top result: {top_context}"),
        });
        match client.call("memory_store", params) {
            Ok(_) => self.set_status_message("Search saved as memory".to_string()),
            Err(error) => self.set_status_message(format!("Store error: {error}")),
        }
    }
}

fn json_string(value: &serde_json::Value) -> String {
    value.as_str().unwrap_or("").to_string()
}
