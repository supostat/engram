use ratatui::crossterm::event::KeyCode;

use crate::actions;
use crate::overlays::{ConfirmAction, ConfirmDialog, FilterState, StatusMessage};
use crate::tabs::{ModelsKeyAction, SearchKeyAction, SearchStatus};

use super::{App, Tab};

impl App {
    pub(super) fn handle_key(&mut self, code: KeyCode) {
        if self.confirm_dialog.is_some() {
            self.handle_confirm_key(code);
            return;
        }
        if self.filter_state.is_some() {
            self.handle_filter_key(code);
            return;
        }
        if self.consolidation_preview.is_some() {
            if matches!(code, KeyCode::Esc | KeyCode::Enter | KeyCode::Char('q')) {
                self.consolidation_preview = None;
            }
            return;
        }
        if self.tab == Tab::Memories && self.memories_state.search_active {
            self.handle_memories_search_key(code);
            return;
        }
        if self.tab == Tab::Memories && self.memories_state.detail_open {
            match code {
                KeyCode::Esc | KeyCode::Enter => self.memories_state.close_detail(),
                _ => {}
            }
            return;
        }
        if self.tab == Tab::Search && !matches!(self.search_state.status, SearchStatus::Offline) {
            match self.search_state.handle_key(code, &mut self.socket) {
                SearchKeyAction::Quit => self.should_quit = true,
                SearchKeyAction::NextTab => self.next_tab(),
                SearchKeyAction::PreviousTab => self.previous_tab(),
                SearchKeyAction::Handled => {}
            }
            return;
        }
        if self.tab == Tab::Memories && self.handle_memories_key(code) {
            return;
        }
        if self.tab == Tab::Status && self.handle_status_key(code) {
            return;
        }
        if self.tab == Tab::Models {
            match self.models_state.handle_key(code) {
                ModelsKeyAction::Handled => return,
                ModelsKeyAction::Fallthrough(fallthrough_code) => {
                    self.handle_global_key(fallthrough_code);
                    return;
                }
            }
        }
        self.handle_global_key(code);
    }

    fn handle_memories_key(&mut self, code: KeyCode) -> bool {
        match code {
            KeyCode::Down => {
                self.memories_state.move_down();
                true
            }
            KeyCode::Up => {
                self.memories_state.move_up();
                true
            }
            KeyCode::Enter => {
                self.memories_state.toggle_detail();
                true
            }
            KeyCode::Char('j') => {
                self.action_judge_memory();
                true
            }
            KeyCode::Char('k') => {
                self.memories_state.move_up();
                true
            }
            KeyCode::Char('d') => {
                self.action_confirm_delete();
                true
            }
            KeyCode::Char('f') => {
                self.action_open_filter();
                true
            }
            KeyCode::Char('/') => {
                self.memories_state.activate_search();
                true
            }
            _ => false,
        }
    }

    fn handle_status_key(&mut self, code: KeyCode) -> bool {
        match code {
            KeyCode::Char('e') => {
                self.action_export();
                true
            }
            KeyCode::Char('c') => {
                self.action_consolidation_preview();
                true
            }
            _ => false,
        }
    }

    fn handle_global_key(&mut self, code: KeyCode) {
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

    fn handle_confirm_key(&mut self, code: KeyCode) {
        match code {
            KeyCode::Char('y') => {
                if let Some(dialog) = self.confirm_dialog.take() {
                    self.execute_confirm(dialog.action);
                }
            }
            KeyCode::Char('n') | KeyCode::Esc => {
                self.confirm_dialog = None;
            }
            _ => {}
        }
    }

    fn handle_filter_key(&mut self, code: KeyCode) {
        match code {
            KeyCode::Up | KeyCode::Char('k') => {
                if let Some(ref mut filter) = self.filter_state {
                    filter.move_up();
                }
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if let Some(ref mut filter) = self.filter_state {
                    filter.move_down();
                }
            }
            KeyCode::Enter => {
                if let Some(filter) = self.filter_state.take()
                    && let Some(selected_type) = filter.selected_type()
                {
                    self.memories_state.apply_filter(selected_type.to_string());
                }
            }
            KeyCode::Esc => {
                self.filter_state = None;
            }
            _ => {}
        }
    }

    fn handle_memories_search_key(&mut self, code: KeyCode) {
        match code {
            KeyCode::Esc => self.memories_state.deactivate_search(),
            KeyCode::Backspace => self.memories_state.search_delete_char(),
            KeyCode::Char(character) => self.memories_state.search_insert_char(character),
            _ => {}
        }
    }

    fn execute_confirm(&mut self, action: ConfirmAction) {
        match action {
            ConfirmAction::DeleteMemory(memory_id) => {
                let message =
                    actions::delete_memory(&self.database, &self.database_path, &memory_id);
                self.status_message = Some(message);
                self.force_refresh();
            }
        }
    }

    fn action_judge_memory(&mut self) {
        let memory_id = match self.memories_state.selected_memory_id() {
            Some(id) => id.to_string(),
            None => return,
        };
        let message = actions::judge_memory(&mut self.socket, &memory_id);
        self.status_message = Some(message);
        self.force_refresh();
    }

    fn action_confirm_delete(&mut self) {
        let memory_id = match self.memories_state.selected_memory_id() {
            Some(id) => id.to_string(),
            None => return,
        };
        let short_id = if memory_id.len() > 12 {
            &memory_id[..12]
        } else {
            &memory_id
        };
        self.confirm_dialog = Some(ConfirmDialog {
            message: format!("Delete memory {short_id}...?"),
            action: ConfirmAction::DeleteMemory(memory_id),
        });
    }

    fn action_open_filter(&mut self) {
        if self.filter_state.is_some() {
            self.filter_state = None;
            self.memories_state.clear_filter();
            return;
        }
        let types = self.database.memory_types();
        if types.is_empty() {
            return;
        }
        let mut filter = FilterState::new(types);
        filter.current_filter = self.memories_state.type_filter.clone();
        self.filter_state = Some(filter);
    }

    fn action_export(&mut self) {
        let message = actions::export_memories(&mut self.socket);
        self.status_message = Some(message);
    }

    fn action_consolidation_preview(&mut self) {
        match actions::consolidation_preview(&mut self.socket) {
            Some(text) => self.consolidation_preview = Some(text),
            None => {
                self.status_message = Some(StatusMessage::error("Server offline".to_string()));
            }
        }
    }
}
