use clap::{Parser, Subcommand};
use serde_json::json;

use engram_core::cli;
use engram_core::config::Config;
use engram_core::output::OutputFormat;
use engram_core::server;

#[derive(Parser)]
#[command(name = "engram", about = "Memory system for AI agents")]
struct EngramCli {
    #[command(subcommand)]
    command: Command,

    #[arg(long, global = true)]
    config: Option<String>,

    #[arg(long, global = true, default_value = "json")]
    format: OutputFormat,
}

#[derive(Subcommand)]
enum Command {
    /// Start the Unix socket server
    Server,

    /// Store a new memory
    Store {
        #[arg(long)]
        context: String,
        #[arg(long)]
        action: String,
        #[arg(long)]
        result: String,
        #[arg(long)]
        memory_type: Option<String>,
        #[arg(long)]
        tags: Option<String>,
        #[arg(long)]
        project: Option<String>,
    },

    /// Search memories
    Search {
        #[arg(long)]
        query: String,
        #[arg(long, default_value = "10")]
        limit: usize,
        #[arg(long)]
        project: Option<String>,
    },

    /// Judge a memory's relevance
    Judge {
        #[arg(long)]
        memory_id: String,
        #[arg(long)]
        query: Option<String>,
        #[arg(long)]
        score: Option<f64>,
    },

    /// Delete a memory by ID
    Delete {
        #[arg(long)]
        id: String,
    },

    /// Show system status
    Status,

    /// Consolidation operations
    Consolidate {
        #[command(subcommand)]
        action: ConsolidateAction,
    },

    /// Initialize Engram: create config, database, print MCP setup
    Init,

    /// Migrate memories from legacy ~/.engram/engram.db into the current project database.
    /// Default filter: cwd basename exact match against memories.project; NULL project rows
    /// are skipped unless --all is passed.
    Migrate {
        /// Include memories whose project field is NULL or differs from the cwd basename.
        #[arg(long)]
        all: bool,
        /// Report what would be migrated without writing to the destination database.
        #[arg(long)]
        dry_run: bool,
    },

    /// Training operations
    Train {
        #[command(subcommand)]
        action: TrainAction,
    },

    /// Re-embed all memories with the currently configured provider.
    /// Use after switching `embedding.model` in engram.toml. See ADR
    /// 2026-05-14-voyage-4-migration-via-reembed-cli for the migration flow.
    Reembed {
        /// Reserved for future safety thresholds (e.g., refuse if memory
        /// count exceeds a configured limit). Currently a no-op placeholder.
        #[arg(long)]
        force: bool,
    },

    /// Show version
    Version,
}

#[derive(Subcommand)]
enum TrainAction {
    /// Generate insights from memory patterns
    Generate,
    /// List active insights
    List,
    /// Delete an insight by ID
    Delete {
        #[arg(long)]
        id: String,
    },
}

#[derive(Subcommand)]
enum ConsolidateAction {
    /// Preview consolidation candidates
    Preview {
        #[arg(long)]
        stale_days: Option<u32>,
        #[arg(long, value_parser = parse_finite_f64)]
        min_score: Option<f64>,
    },
    /// Analyze and generate recommendations
    Analyze {
        #[arg(long)]
        stale_days: Option<u32>,
        #[arg(long, value_parser = parse_finite_f64)]
        min_score: Option<f64>,
    },
    /// Apply consolidation recommendations
    Apply {
        #[arg(long)]
        stale_days: Option<u32>,
        #[arg(long, value_parser = parse_finite_f64)]
        min_score: Option<f64>,
        /// Skip recommendations below this confidence (0.0-1.0, server default 0.0)
        #[arg(long, value_parser = parse_finite_f32)]
        min_confidence: Option<f32>,
    },
    /// List consolidation history (merge/delete/archive audit trail), newest first
    Log {
        #[arg(long)]
        limit: Option<usize>,
    },
}

#[tokio::main]
async fn main() {
    let parsed = EngramCli::parse();
    if let Err(error) = run(parsed).await {
        eprintln!("engram: {error}");
        std::process::exit(1);
    }
}

async fn run(parsed: EngramCli) -> Result<(), engram_core::CoreError> {
    match parsed.command {
        Command::Init => engram_core::init_handler::execute(),
        Command::Migrate { all, dry_run } => {
            engram_core::migrate_handler::execute(all, dry_run, &parsed.format)
        }
        command => run_with_config(parsed.config, command, parsed.format).await,
    }
}

async fn run_with_config(
    config_path: Option<String>,
    command: Command,
    format: OutputFormat,
) -> Result<(), engram_core::CoreError> {
    let config = load_config(&config_path)?;
    match command {
        Command::Server => server::run(config).await,
        Command::Version => {
            print_version(&format);
            Ok(())
        }
        command => execute_command(config, command, &format).await,
    }
}

fn load_config(path: &Option<String>) -> Result<Config, engram_core::CoreError> {
    match path {
        Some(explicit_path) => Config::load_from_path(explicit_path),
        None => Config::load(),
    }
}

fn print_version(format: &OutputFormat) {
    let version = env!("CARGO_PKG_VERSION");
    let output = engram_core::output::format_output(&json!({ "version": version }), format);
    println!("{output}");
}

async fn execute_command(
    config: Config,
    command: Command,
    format: &OutputFormat,
) -> Result<(), engram_core::CoreError> {
    // `cli::build_state` ends up calling `reqwest::blocking::ClientBuilder::build`,
    // which spins up and immediately drops an internal current-thread tokio
    // runtime. Running that drop on a worker of the outer multi-threaded runtime
    // panics (`Cannot drop a runtime in a context where blocking is not allowed`).
    // `spawn_blocking` moves the construction to the blocking pool where the
    // inner runtime can shut down cleanly.
    let state = {
        let config = config.clone();
        tokio::task::spawn_blocking(move || cli::build_state(&config))
            .await
            .map_err(|error| engram_core::CoreError::SocketError(error.to_string()))??
    };
    let (method, params) = build_dispatch_args(command);
    cli::execute(state, &method, params, format).await
}

fn build_dispatch_args(command: Command) -> (String, serde_json::Value) {
    match command {
        Command::Store {
            context,
            action,
            result,
            memory_type,
            tags,
            project,
        } => build_store_args(context, action, result, memory_type, tags, project),
        Command::Search {
            query,
            limit,
            project,
        } => build_search_args(query, limit, project),
        Command::Judge {
            memory_id,
            query,
            score,
        } => build_judge_args(memory_id, query, score),
        Command::Delete { id } => ("memory_delete".into(), json!({ "id": id })),
        Command::Status => ("memory_status".into(), json!({})),
        Command::Consolidate { action } => build_consolidate_args(action),
        Command::Train { action } => build_train_args(action),
        Command::Reembed { force } => ("memory_reembed".into(), json!({ "force": force })),
        Command::Server | Command::Version | Command::Init | Command::Migrate { .. } => {
            unreachable!()
        }
    }
}

fn build_store_args(
    context: String,
    action: String,
    result: String,
    memory_type: Option<String>,
    tags: Option<String>,
    project: Option<String>,
) -> (String, serde_json::Value) {
    let mut params = json!({
        "context": context,
        "action": action,
        "result": result,
        "memory_type": memory_type.unwrap_or_else(|| "decision".into()),
    });
    if let Some(tags_value) = tags {
        let tag_array: Vec<String> = tags_value
            .split(',')
            .map(|tag| tag.trim().to_string())
            .filter(|tag| !tag.is_empty())
            .collect();
        if !tag_array.is_empty() {
            params["tags"] = serde_json::json!(tag_array);
        }
    }
    if let Some(project_value) = project {
        params["project"] = serde_json::Value::String(project_value);
    }
    ("memory_store".into(), params)
}

fn build_search_args(
    query: String,
    limit: usize,
    project: Option<String>,
) -> (String, serde_json::Value) {
    let mut params = json!({
        "query": query,
        "limit": limit,
    });
    if let Some(project_value) = project {
        params["project"] = serde_json::Value::String(project_value);
    }
    ("memory_search".into(), params)
}

fn build_judge_args(
    memory_id: String,
    query: Option<String>,
    score: Option<f64>,
) -> (String, serde_json::Value) {
    let mut params = json!({ "memory_id": memory_id });
    if let Some(query_value) = query {
        params["query"] = serde_json::Value::String(query_value);
    }
    if let Some(score_value) = score {
        params["score"] = serde_json::json!(score_value);
    }
    ("memory_judge".into(), params)
}

fn build_consolidate_args(action: ConsolidateAction) -> (String, serde_json::Value) {
    match action {
        ConsolidateAction::Preview {
            stale_days,
            min_score,
        } => (
            "memory_consolidate_preview".into(),
            consolidation_params(stale_days, min_score),
        ),
        ConsolidateAction::Analyze {
            stale_days,
            min_score,
        } => (
            "memory_consolidate".into(),
            consolidation_params(stale_days, min_score),
        ),
        ConsolidateAction::Apply {
            stale_days,
            min_score,
            min_confidence,
        } => {
            let mut params = consolidation_params(stale_days, min_score);
            if let Some(value) = min_confidence {
                params["min_confidence"] = json!(value);
            }
            ("memory_consolidate_apply".into(), params)
        }
        ConsolidateAction::Log { limit } => {
            let mut params = json!({});
            if let Some(value) = limit {
                params["limit"] = json!(value);
            }
            ("memory_consolidate_log".into(), params)
        }
    }
}

fn build_train_args(action: TrainAction) -> (String, serde_json::Value) {
    match action {
        TrainAction::Generate => ("memory_train_generate".into(), json!({})),
        TrainAction::List => ("memory_train_list".into(), json!({})),
        TrainAction::Delete { id } => ("memory_train_delete".into(), json!({"id": id})),
    }
}

fn consolidation_params(stale_days: Option<u32>, min_score: Option<f64>) -> serde_json::Value {
    let mut params = json!({});
    if let Some(days) = stale_days {
        params["stale_days"] = json!(days);
    }
    if let Some(score) = min_score {
        params["min_score"] = json!(score);
    }
    params
}

// Non-finite floats serialize to JSON null, which the server reads as an absent
// parameter and replaces with its default — silently disabling the threshold.
// Reject them at the CLI boundary; range validation stays server-side.
fn parse_finite_f32(text: &str) -> Result<f32, String> {
    let value = text.parse::<f32>().map_err(|error| error.to_string())?;
    if !value.is_finite() {
        return Err("must be a finite number".into());
    }
    Ok(value)
}

fn parse_finite_f64(text: &str) -> Result<f64, String> {
    let value = text.parse::<f64>().map_err(|error| error.to_string())?;
    if !value.is_finite() {
        return Err("must be a finite number".into());
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn consolidate_apply_forwards_min_confidence() {
        let (method, params) = build_consolidate_args(ConsolidateAction::Apply {
            stale_days: None,
            min_score: None,
            min_confidence: Some(0.7),
        });
        assert_eq!(method, "memory_consolidate_apply");
        let forwarded = params["min_confidence"]
            .as_f64()
            .expect("min_confidence must be forwarded as a number");
        // f32 -> JSON widens through f64, so 0.7f32 is not bit-equal to 0.7f64.
        assert!((forwarded - 0.7).abs() < 1e-6);
    }

    #[test]
    fn consolidate_apply_omits_absent_min_confidence() {
        let (method, params) = build_consolidate_args(ConsolidateAction::Apply {
            stale_days: None,
            min_score: None,
            min_confidence: None,
        });
        assert_eq!(method, "memory_consolidate_apply");
        assert!(
            params.get("min_confidence").is_none(),
            "absent flag must leave the key out so the server default applies"
        );
    }

    #[test]
    fn finite_threshold_parsers_accept_decimal_input() {
        assert_eq!(parse_finite_f32("0.7"), Ok(0.7_f32));
        assert_eq!(parse_finite_f64("0.7"), Ok(0.7_f64));
    }

    #[test]
    fn finite_threshold_parsers_reject_non_finite_input() {
        for literal in ["nan", "inf", "-inf"] {
            assert!(
                parse_finite_f32(literal).is_err(),
                "f32 parser must reject {literal}"
            );
            assert!(
                parse_finite_f64(literal).is_err(),
                "f64 parser must reject {literal}"
            );
        }
    }

    #[test]
    fn cli_rejects_non_finite_consolidation_thresholds() {
        for arguments in [
            ["engram", "consolidate", "preview", "--min-score", "nan"],
            ["engram", "consolidate", "analyze", "--min-score", "inf"],
            ["engram", "consolidate", "apply", "--min-score", "-inf"],
            ["engram", "consolidate", "apply", "--min-confidence", "nan"],
        ] {
            assert!(
                EngramCli::try_parse_from(arguments).is_err(),
                "non-finite threshold must be rejected: {arguments:?}"
            );
        }
    }
}
