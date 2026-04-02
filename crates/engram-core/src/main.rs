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

    /// Show system status
    Status,

    /// Consolidation operations
    Consolidate {
        #[command(subcommand)]
        action: ConsolidateAction,
    },

    /// Initialize Engram: create config, database, print MCP setup
    Init,

    /// Training operations
    Train {
        #[command(subcommand)]
        action: TrainAction,
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
        #[arg(long)]
        min_score: Option<f64>,
    },
    /// Analyze and generate recommendations
    Analyze {
        #[arg(long)]
        stale_days: Option<u32>,
        #[arg(long)]
        min_score: Option<f64>,
    },
    /// Apply consolidation recommendations
    Apply {
        #[arg(long)]
        stale_days: Option<u32>,
        #[arg(long)]
        min_score: Option<f64>,
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
    let state = cli::build_state(&config)?;
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
        Command::Status => ("memory_status".into(), json!({})),
        Command::Consolidate { action } => build_consolidate_args(action),
        Command::Train { action } => build_train_args(action),
        Command::Server | Command::Version | Command::Init => unreachable!(),
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
        params["tags"] = serde_json::Value::String(tags_value);
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
        } => (
            "memory_consolidate_apply".into(),
            consolidation_params(stale_days, min_score),
        ),
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
