mod config;
mod existing_config;
mod navigation;
mod render;
mod render_health;
mod render_status;
mod wizard;

use std::io;

use ratatui::DefaultTerminal;

use wizard::InitWizard;

pub fn run_init_wizard(terminal: &mut DefaultTerminal) -> io::Result<()> {
    let mut wizard = InitWizard::new();
    wizard.run(terminal)
}
