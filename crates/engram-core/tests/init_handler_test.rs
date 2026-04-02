use std::sync::Mutex;

use tempfile::tempdir;

use engram_core::init_handler;

static SERIAL_TEST: Mutex<()> = Mutex::new(());

fn with_temp_home(test_body: impl FnOnce(&str)) {
    let _lock = SERIAL_TEST
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let temp = tempdir().expect("failed to create temp dir");
    let original_home = std::env::var("HOME").ok();
    let temp_path = temp.path().to_str().expect("temp path must be valid utf-8");
    unsafe { std::env::set_var("HOME", temp_path) };
    test_body(temp_path);
    match original_home {
        Some(value) => unsafe { std::env::set_var("HOME", value) },
        None => unsafe { std::env::remove_var("HOME") },
    }
}

#[test]
fn init_creates_config_file() {
    with_temp_home(|home| {
        let result = init_handler::execute();
        assert!(result.is_ok(), "execute should succeed: {result:?}");
        let config_path = format!("{home}/.engram/engram.toml");
        assert!(
            std::path::Path::new(&config_path).exists(),
            "config file should exist at {config_path}"
        );
        let content = std::fs::read_to_string(&config_path).expect("should read config file");
        assert!(content.contains("[database]"));
        assert!(content.contains("[embedding]"));
        assert!(content.contains("[llm]"));
        assert!(content.contains("[server]"));
        assert!(content.contains("[hnsw]"));
        assert!(content.contains("[consolidation]"));
    });
}

#[test]
fn init_creates_database() {
    with_temp_home(|home| {
        let result = init_handler::execute();
        assert!(result.is_ok(), "execute should succeed: {result:?}");
        let database_path = format!("{home}/.engram/memories.db");
        assert!(
            std::path::Path::new(&database_path).exists(),
            "database file should exist at {database_path}"
        );
    });
}

#[test]
fn init_skips_if_config_exists() {
    with_temp_home(|home| {
        let engram_directory = format!("{home}/.engram");
        std::fs::create_dir_all(&engram_directory).expect("create dir");
        let config_path = format!("{engram_directory}/engram.toml");
        let original_content = "# existing config\n";
        std::fs::write(&config_path, original_content).expect("write config");

        let result = init_handler::execute();
        assert!(result.is_ok(), "execute should succeed: {result:?}");

        let content = std::fs::read_to_string(&config_path).expect("should read config");
        assert_eq!(
            content, original_content,
            "config should not be overwritten"
        );
    });
}

#[test]
fn init_creates_engram_directory() {
    with_temp_home(|home| {
        let engram_directory = format!("{home}/.engram");
        assert!(
            !std::path::Path::new(&engram_directory).exists(),
            "directory should not exist before init"
        );

        let result = init_handler::execute();
        assert!(result.is_ok(), "execute should succeed: {result:?}");

        assert!(
            std::path::Path::new(&engram_directory).is_dir(),
            "~/.engram/ should be created"
        );
    });
}
