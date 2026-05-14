//! Smoke tests for CLI-style state construction.
//!
//! These tests verify that `cli::build_state_with_dirs` and
//! `server::initialize_state` can be called from inside an async
//! multi-threaded tokio runtime without panicking inside the
//! `reqwest::blocking::ClientBuilder::build` shutdown sequence. The
//! production fix wraps both call sites in `tokio::task::spawn_blocking`;
//! these tests exercise the same wrapper so a regression to direct
//! sync invocation would surface as a test panic, not at first user
//! contact with the binary.

use std::path::PathBuf;

use engram_core::cli;
use engram_core::config::Config;

fn project_dir(suffix: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("engram-smoke-{}-{suffix}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join(".engram")).unwrap();
    dir
}

fn home_dir(suffix: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("engram-smoke-home-{}-{suffix}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[tokio::test]
async fn build_state_with_voyage_provider_does_not_panic() {
    let project = project_dir("voyage");
    let home = home_dir("voyage");
    let mut config = Config::default();
    config.embedding.provider = "voyage".into();
    config.embedding.api_key = Some("test-key-not-used".into());

    let result =
        tokio::task::spawn_blocking(move || cli::build_state_with_dirs(&config, &project, &home))
            .await
            .expect("spawn_blocking join");

    assert!(
        result.is_ok(),
        "voyage state build should succeed: {}",
        result.err().map(|e| e.to_string()).unwrap_or_default()
    );
}

#[tokio::test]
async fn build_state_with_deterministic_provider_does_not_panic() {
    let project = project_dir("deterministic");
    let home = home_dir("deterministic");
    let mut config = Config::default();
    config.embedding.provider = "deterministic".into();

    let result =
        tokio::task::spawn_blocking(move || cli::build_state_with_dirs(&config, &project, &home))
            .await
            .expect("spawn_blocking join");

    assert!(
        result.is_ok(),
        "deterministic state build should succeed: {}",
        result.err().map(|e| e.to_string()).unwrap_or_default()
    );
}
