use std::path::PathBuf;
use std::sync::Mutex;

use tempfile::tempdir;

use engram_core::init_handler;

// Serializes tests that mutate process-global env vars (ENGRAM_MCP_SERVER_PATH,
// ENGRAM_BIN, HOME). Documented as acceptable minor risk in plan.
static ENV_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn init_creates_config_file() {
    let project = tempdir().expect("project");
    let home = tempdir().expect("home");
    let result = init_handler::execute_with_dirs(project.path(), home.path());
    assert!(result.is_ok(), "execute should succeed: {result:?}");

    let config_path = home.path().join(".engram").join("engram.toml");
    assert!(
        config_path.exists(),
        "home config file should exist at {config_path:?}"
    );
    let content = std::fs::read_to_string(&config_path).expect("should read config file");
    // [database] block intentionally absent — database.path is resolved from
    // per-project layout and the field is optional; see ADR
    // 2026-05-13-config-optional-database-and-socket-fallbacks.
    assert!(content.contains("[embedding]"));
    assert!(content.contains("[llm]"));
    assert!(content.contains("[server]"));
    assert!(content.contains("[hnsw]"));
    assert!(content.contains("[consolidation]"));
}

#[test]
fn init_creates_database() {
    let project = tempdir().expect("project");
    let home = tempdir().expect("home");
    let result = init_handler::execute_with_dirs(project.path(), home.path());
    assert!(result.is_ok(), "execute should succeed: {result:?}");
    let database_path = project.path().join(".engram").join("engram.db");
    assert!(
        database_path.exists(),
        "project database should exist at {database_path:?}"
    );
}

#[test]
fn init_socket_path_too_long() {
    // Path length validation is a pure string check; physical directories need not exist.
    let home = tempdir().expect("home");
    let root = tempdir().expect("project root");
    let mut deep: PathBuf = root.path().to_path_buf();
    // 30 segments of 10 chars + separators guarantees the socket path exceeds 104 bytes.
    let segment = "abcdefghij";
    for _ in 0..30 {
        deep.push(segment);
    }

    let result = init_handler::execute_with_dirs(&deep, home.path());
    let error = result.expect_err("should fail due to socket path length");
    let message = error.to_string();
    assert!(
        message.contains("[6012]") && message.to_lowercase().contains("socket"),
        "expected socket length error, got: {message}"
    );
}

#[test]
fn gitignore_idempotent_empty() {
    let project = tempdir().expect("project");
    let home = tempdir().expect("home");
    init_handler::execute_with_dirs(project.path(), home.path()).expect("init");
    let gitignore_path = project.path().join(".gitignore");
    let content = std::fs::read_to_string(&gitignore_path).expect("gitignore exists");
    assert!(content.contains(".engram/"), "marker missing: {content:?}");
}

#[test]
fn gitignore_append_when_missing_marker() {
    let project = tempdir().expect("project");
    let home = tempdir().expect("home");
    let gitignore_path = project.path().join(".gitignore");
    std::fs::write(&gitignore_path, "node_modules\n").expect("seed gitignore");

    init_handler::execute_with_dirs(project.path(), home.path()).expect("init");

    let content = std::fs::read_to_string(&gitignore_path).expect("gitignore exists");
    assert!(
        content.contains("node_modules"),
        "original content retained"
    );
    assert!(content.contains(".engram/"), "marker appended");
}

#[test]
fn gitignore_noop_when_marker_present() {
    let project = tempdir().expect("project");
    let home = tempdir().expect("home");
    let gitignore_path = project.path().join(".gitignore");
    let original = "node_modules\n.engram/\nother\n";
    std::fs::write(&gitignore_path, original).expect("seed gitignore");

    init_handler::execute_with_dirs(project.path(), home.path()).expect("init");

    let content = std::fs::read_to_string(&gitignore_path).expect("gitignore exists");
    assert_eq!(content, original, "gitignore must not be modified");
}

#[test]
fn mcp_json_with_node_path() {
    let _lock = ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let project = tempdir().expect("project");
    let home = tempdir().expect("home");
    let script = tempdir().expect("script tempdir");
    let script_path = script.path().join("index.js");
    std::fs::write(&script_path, b"// stub").expect("write stub script");
    let bin_tempdir = tempdir().expect("bin tempdir");
    let bin_path = bin_tempdir.path().join("engram");
    std::fs::write(&bin_path, b"#!/bin/sh\n").expect("stub bin");

    let original_bin = std::env::var("ENGRAM_BIN").ok();
    // SAFETY: serialized via ENV_LOCK; restored before returning.
    unsafe {
        std::env::set_var("ENGRAM_BIN", &bin_path);
        std::env::set_var("ENGRAM_MCP_SERVER_PATH", &script_path);
    }
    let result = init_handler::execute_with_dirs(project.path(), home.path());
    unsafe {
        std::env::remove_var("ENGRAM_MCP_SERVER_PATH");
        match original_bin {
            Some(value) => std::env::set_var("ENGRAM_BIN", value),
            None => std::env::remove_var("ENGRAM_BIN"),
        }
    }
    assert!(result.is_ok(), "init should succeed: {result:?}");

    let mcp_path = project.path().join(".mcp.json");
    let content = std::fs::read_to_string(&mcp_path).expect("mcp json");
    let parsed: serde_json::Value = serde_json::from_str(&content).expect("valid json");
    let command = parsed["mcpServers"]["engram"]["command"]
        .as_str()
        .expect("command is string");
    assert_eq!(command, "node");
    let args = parsed["mcpServers"]["engram"]["args"]
        .as_array()
        .expect("args array");
    assert_eq!(args.len(), 1);
    let canonical = script_path
        .canonicalize()
        .map(|path| path.to_string_lossy().to_string())
        .unwrap_or_else(|_| script_path.to_string_lossy().to_string());
    let arg_value = args[0].as_str().expect("arg string").to_string();
    assert!(
        arg_value == canonical || arg_value == script_path.to_string_lossy(),
        "arg={arg_value}, canonical={canonical}"
    );
    let env_block = parsed["mcpServers"]["engram"]["env"]
        .as_object()
        .expect("env block must be present when ENGRAM_BIN is resolved");
    let engram_bin_value = env_block
        .get("ENGRAM_BIN")
        .and_then(|value| value.as_str())
        .expect("ENGRAM_BIN must be a string");
    let canonical_bin = bin_path
        .canonicalize()
        .map(|path| path.to_string_lossy().to_string())
        .unwrap_or_else(|_| bin_path.to_string_lossy().to_string());
    assert_eq!(
        engram_bin_value, canonical_bin,
        "ENGRAM_BIN env must match the resolved absolute binary path"
    );
}

#[test]
fn mcp_json_fallback_to_npx() {
    let _lock = ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let project = tempdir().expect("project");
    let home = tempdir().expect("home");
    let isolated = tempdir().expect("isolated bin");
    let bin_path = isolated.path().join("engram");
    std::fs::write(&bin_path, b"#!/bin/sh\n").expect("stub bin");

    let original_bin = std::env::var("ENGRAM_BIN").ok();
    // SAFETY: serialized via ENV_LOCK; restored before returning.
    unsafe {
        std::env::remove_var("ENGRAM_MCP_SERVER_PATH");
        std::env::set_var("ENGRAM_BIN", &bin_path);
    }
    let result = init_handler::execute_with_dirs(project.path(), home.path());
    unsafe {
        match original_bin {
            Some(value) => std::env::set_var("ENGRAM_BIN", value),
            None => std::env::remove_var("ENGRAM_BIN"),
        }
    }
    assert!(result.is_ok(), "init should succeed: {result:?}");

    let mcp_path = project.path().join(".mcp.json");
    let content = std::fs::read_to_string(&mcp_path).expect("mcp json");
    let parsed: serde_json::Value = serde_json::from_str(&content).expect("valid json");
    let command = parsed["mcpServers"]["engram"]["command"]
        .as_str()
        .expect("command is string");
    assert_eq!(command, "npx");
    let args = parsed["mcpServers"]["engram"]["args"]
        .as_array()
        .expect("args array");
    assert_eq!(args.len(), 1);
    assert_eq!(args[0].as_str().unwrap(), "@engramm/engram-mcp-server");
    let env_block = parsed["mcpServers"]["engram"]["env"]
        .as_object()
        .expect("env block must be present when ENGRAM_BIN is set");
    let engram_bin_value = env_block
        .get("ENGRAM_BIN")
        .and_then(|value| value.as_str())
        .expect("ENGRAM_BIN must be a string");
    let canonical_bin = bin_path
        .canonicalize()
        .map(|path| path.to_string_lossy().to_string())
        .unwrap_or_else(|_| bin_path.to_string_lossy().to_string());
    assert_eq!(
        engram_bin_value, canonical_bin,
        "ENGRAM_BIN env must match the resolved absolute binary path"
    );
}

#[test]
fn init_execute_fails_when_home_not_set() {
    let _lock = ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    // The public `execute()` wrapper must surface a clear error when HOME is missing,
    // so the underlying invariant (resolve cwd + HOME, then delegate) is verified.
    let original_home = std::env::var("HOME").ok();
    // SAFETY: serialized via ENV_LOCK; restored before returning.
    unsafe {
        std::env::remove_var("HOME");
    }
    let result = init_handler::execute();
    unsafe {
        if let Some(value) = original_home {
            std::env::set_var("HOME", value);
        }
    }

    let error = result.expect_err("execute() must fail without HOME");
    let message = error.to_string();
    assert!(
        message.contains("[6012]") && message.contains("HOME"),
        "expected HOME-related init failure, got: {message}"
    );
}

#[test]
fn gitignore_appends_after_missing_trailing_newline() {
    let project = tempdir().expect("project");
    let home = tempdir().expect("home");
    let gitignore_path = project.path().join(".gitignore");
    std::fs::write(&gitignore_path, "*.log").expect("seed gitignore without trailing newline");

    init_handler::execute_with_dirs(project.path(), home.path()).expect("init");

    let content = std::fs::read_to_string(&gitignore_path).expect("gitignore exists");
    assert_eq!(
        content, "*.log\n.engram/\n",
        "marker should follow an inserted newline, existing content preserved"
    );
}

#[test]
fn gitignore_empty_file() {
    let project = tempdir().expect("project");
    let home = tempdir().expect("home");
    let gitignore_path = project.path().join(".gitignore");
    std::fs::write(&gitignore_path, "").expect("seed empty gitignore");

    init_handler::execute_with_dirs(project.path(), home.path()).expect("init");

    let content = std::fs::read_to_string(&gitignore_path).expect("gitignore exists");
    assert!(
        content.contains(".engram/"),
        "marker must be present in seeded empty file: {content:?}"
    );
}

// Socket path layout: <project_dir>/.engram/engram.sock — exactly 20 bytes
// for the relative tail (including the leading separator). The UNIX limit
// is 104 bytes inclusive; `validate_socket_path` rejects at length >= 104.
const SOCKET_RELATIVE_TAIL_BYTES: usize = "/.engram/engram.sock".len();

fn build_project_dir_of_length(target_len: usize) -> (tempfile::TempDir, PathBuf) {
    let root = tempdir().expect("project root tempdir");
    let root_len = root.path().as_os_str().len();
    assert!(
        target_len > root_len,
        "tempdir path ({root_len} bytes) already longer than target {target_len}"
    );
    // A single separator already sits between root and the padding segment.
    let remaining = target_len - root_len - 1;
    let padding: String = "a".repeat(remaining);
    let deep = root.path().join(&padding);
    std::fs::create_dir_all(&deep).expect("create padded project dir");
    assert_eq!(
        deep.as_os_str().len(),
        target_len,
        "project dir length must equal target"
    );
    (root, deep)
}

#[test]
fn init_socket_path_boundary_103_bytes() {
    // project_dir length such that socket path = 103 bytes (allowed).
    let target = 103 - SOCKET_RELATIVE_TAIL_BYTES;
    let (_root, project_dir) = build_project_dir_of_length(target);
    let home = tempdir().expect("home");

    let result = init_handler::execute_with_dirs(&project_dir, home.path());
    assert!(
        result.is_ok(),
        "socket path of 103 bytes must succeed: {result:?}"
    );
}

#[test]
fn init_socket_path_boundary_104_bytes() {
    // project_dir length such that socket path = 104 bytes (rejected).
    let target = 104 - SOCKET_RELATIVE_TAIL_BYTES;
    let (_root, project_dir) = build_project_dir_of_length(target);
    let home = tempdir().expect("home");

    let result = init_handler::execute_with_dirs(&project_dir, home.path());
    let error = result.expect_err("socket path of 104 bytes must fail");
    let message = error.to_string();
    assert!(
        message.contains("104"),
        "error message must cite the 104-byte UNIX limit, got: {message}"
    );
    assert!(
        message.contains("[6012]") && message.to_lowercase().contains("socket"),
        "expected socket length error, got: {message}"
    );
}
