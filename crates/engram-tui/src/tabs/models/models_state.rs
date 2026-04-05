use std::path::PathBuf;
use std::time::Instant;

use ratatui::crossterm::event::KeyCode;

use crate::data::ModelInfo;

const EXPECTED_FILES: &[&str] = &[
    "mode_classifier.onnx",
    "ranking_model.onnx",
    "text_generator.onnx",
    "tokenizer.json",
];

pub enum ModelsKeyAction {
    Handled,
    Fallthrough(KeyCode),
}

pub enum ModelsPopup {
    TrainCommand { command: String },
    DeleteConfirm { filename: String },
}

pub struct ModelsTabState {
    pub models: Vec<ModelInfo>,
    pub models_path: String,
    pub selected: usize,
    pub status_message: Option<(String, Instant)>,
    pub popup: Option<ModelsPopup>,
}

impl ModelsTabState {
    pub fn new(models_path: String) -> Self {
        Self {
            models: Vec::new(),
            models_path,
            selected: 0,
            status_message: None,
            popup: None,
        }
    }

    pub fn total_row_count(&self) -> usize {
        let present: Vec<&str> = self.models.iter().map(|m| m.filename.as_str()).collect();
        let missing_count = EXPECTED_FILES
            .iter()
            .filter(|f| !present.contains(&**f))
            .count();
        self.models.len() + missing_count
    }

    pub fn clamp_selection(&mut self) {
        let total = self.total_row_count();
        if total == 0 {
            self.selected = 0;
        } else if self.selected >= total {
            self.selected = total - 1;
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

    pub fn handle_key(&mut self, code: KeyCode) -> ModelsKeyAction {
        if let Some(popup) = &self.popup {
            match popup {
                ModelsPopup::TrainCommand { .. } => {
                    if matches!(code, KeyCode::Esc | KeyCode::Enter | KeyCode::Char('q')) {
                        self.popup = None;
                    }
                    return ModelsKeyAction::Handled;
                }
                ModelsPopup::DeleteConfirm { .. } => {
                    if matches!(code, KeyCode::Char('y')) {
                        self.confirm_delete();
                    } else {
                        self.popup = None;
                    }
                    return ModelsKeyAction::Handled;
                }
            }
        }

        match code {
            KeyCode::Char('j') | KeyCode::Down => {
                let total = self.total_row_count();
                if total > 0 {
                    self.selected = (self.selected + 1).min(total - 1);
                }
            }
            KeyCode::Char('k') | KeyCode::Up => {
                self.selected = self.selected.saturating_sub(1);
            }
            KeyCode::Char('t') => {
                self.popup = Some(ModelsPopup::TrainCommand {
                    command: "engram train".to_string(),
                });
            }
            KeyCode::Char('T') => {
                self.popup = Some(ModelsPopup::TrainCommand {
                    command: "engram train --deep".to_string(),
                });
            }
            KeyCode::Char('d') => {
                self.request_delete();
            }
            _ => return ModelsKeyAction::Fallthrough(code),
        }
        ModelsKeyAction::Handled
    }

    fn request_delete(&mut self) {
        if self.selected >= self.models.len() {
            self.set_status_message("Cannot delete missing model".to_string());
            return;
        }
        let filename = self.models[self.selected].filename.clone();
        self.popup = Some(ModelsPopup::DeleteConfirm { filename });
    }

    fn confirm_delete(&mut self) {
        let filename = match &self.popup {
            Some(ModelsPopup::DeleteConfirm { filename }) => filename.clone(),
            _ => return,
        };
        self.popup = None;
        let path = PathBuf::from(&self.models_path).join(&filename);
        match std::fs::remove_file(&path) {
            Ok(()) => {
                self.models.retain(|m| m.filename != filename);
                self.clamp_selection();
                self.set_status_message(format!("Deleted: {filename}"));
            }
            Err(error) => {
                self.set_status_message(format!("Delete failed: {error}"));
            }
        }
    }
}

pub fn expected_files() -> &'static [&'static str] {
    EXPECTED_FILES
}
