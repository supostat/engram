mod render;

pub use render::render_memories_tab;

use crate::data::MemorySummary;

pub struct MemoriesTabState {
    pub memories: Vec<MemorySummary>,
    pub selected: usize,
    pub detail_open: bool,
    pub search_active: bool,
    pub search_query: String,
    pub type_filter: Option<String>,
}

impl MemoriesTabState {
    pub fn new() -> Self {
        Self {
            memories: Vec::new(),
            selected: 0,
            detail_open: false,
            search_active: false,
            search_query: String::new(),
            type_filter: None,
        }
    }

    pub fn move_up(&mut self) {
        self.selected = self.selected.saturating_sub(1);
    }

    pub fn move_down(&mut self) {
        let visible_count = self.visible_memories().count();
        if visible_count > 0 {
            self.selected = (self.selected + 1).min(visible_count - 1);
        }
    }

    pub fn toggle_detail(&mut self) {
        if self.visible_memories().count() > 0 {
            self.detail_open = !self.detail_open;
        }
    }

    pub fn close_detail(&mut self) {
        self.detail_open = false;
    }

    pub fn clamp_selection(&mut self) {
        let visible_count = self.visible_memories().count();
        if visible_count == 0 {
            self.selected = 0;
        } else {
            self.selected = self.selected.min(visible_count - 1);
        }
    }

    pub fn selected_memory_id(&self) -> Option<&str> {
        self.visible_memories()
            .nth(self.selected)
            .map(|memory| memory.id.as_str())
    }

    pub fn visible_memories(&self) -> impl Iterator<Item = &MemorySummary> {
        let type_filter = self.type_filter.clone();
        let search_query = self.search_query.to_lowercase();
        self.memories.iter().filter(move |memory| {
            if let Some(ref filter_type) = type_filter
                && memory.memory_type != *filter_type
            {
                return false;
            }
            if !search_query.is_empty() && !memory.context.to_lowercase().contains(&search_query) {
                return false;
            }
            true
        })
    }

    pub fn activate_search(&mut self) {
        self.search_active = true;
        self.search_query.clear();
    }

    pub fn deactivate_search(&mut self) {
        self.search_active = false;
        self.search_query.clear();
        self.clamp_selection();
    }

    pub fn search_insert_char(&mut self, character: char) {
        self.search_query.push(character);
        self.selected = 0;
    }

    pub fn search_delete_char(&mut self) {
        self.search_query.pop();
        self.clamp_selection();
    }

    pub fn apply_filter(&mut self, memory_type: String) {
        if self.type_filter.as_ref() == Some(&memory_type) {
            self.type_filter = None;
        } else {
            self.type_filter = Some(memory_type);
        }
        self.selected = 0;
    }

    pub fn clear_filter(&mut self) {
        self.type_filter = None;
        self.clamp_selection();
    }
}
