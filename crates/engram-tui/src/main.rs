mod app;
mod data;
mod tabs;
#[allow(dead_code)]
mod theme;

use clap::Parser;

#[derive(Parser)]
#[command(name = "engram-tui", about = "Terminal dashboard for engram")]
struct Args {
    /// Path to engram SQLite database
    #[arg(long, short)]
    database: Option<String>,

    /// Path to ONNX models directory
    #[arg(long, short)]
    models_path: Option<String>,

    /// Path to engram-core unix socket
    #[arg(long, short)]
    socket: Option<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let database_path = resolve_database_path(args.database);
    let models_path = resolve_models_path(args.models_path);
    let socket_path = resolve_socket_path(args.socket);
    let mut application = app::App::new(&database_path, &models_path, &socket_path)?;
    ratatui::run(|terminal| application.run(terminal))?;
    Ok(())
}

fn resolve_socket_path(explicit_path: Option<String>) -> String {
    if let Some(path) = explicit_path {
        return expand_tilde(&path);
    }
    let home = dirs::home_dir().expect("cannot determine home directory");
    home.join(".engram")
        .join("engram.sock")
        .to_string_lossy()
        .into_owned()
}

fn resolve_models_path(explicit_path: Option<String>) -> String {
    if let Some(path) = explicit_path {
        return expand_tilde(&path);
    }
    let home = dirs::home_dir().expect("cannot determine home directory");
    home.join(".engram")
        .join("models")
        .to_string_lossy()
        .into_owned()
}

fn resolve_database_path(explicit_path: Option<String>) -> String {
    if let Some(path) = explicit_path {
        return expand_tilde(&path);
    }
    let home = dirs::home_dir().expect("cannot determine home directory");
    home.join(".engram")
        .join("memories.db")
        .to_string_lossy()
        .into_owned()
}

fn expand_tilde(path: &str) -> String {
    if !path.starts_with('~') {
        return path.to_string();
    }
    let home = dirs::home_dir().expect("cannot determine home directory");
    home.join(&path[2..]).to_string_lossy().into_owned()
}
