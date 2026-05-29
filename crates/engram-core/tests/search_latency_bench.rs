//! Latency harness: measures memory_search p99 under reindex contention.
//!
//! Opt-in, `#[ignore]`d. Run manually:
//!   cargo test --release -p engram-core -- --ignored --nocapture search_latency
//!
//! Two scenarios:
//!   - `search_latency_idle`   — baseline without reindex activity.
//!   - `search_latency_during_reindex` — background driver forces reindex.
//!
//! Prints to stdout:
//!   search_latency_idle: p50=X.XXms p95=X.XXms p99=X.XXms p99.9=X.XXms max=X.XXms samples=N
//!
//! Expected seeding time ~15s (5000 memories at 1024-dim embeddings).
//! Expected total runtime per test: seeding + warmup + BENCH_DURATION_SECS + teardown approx 60s.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use hdrhistogram::Histogram;
use serde_json::json;

use engram_core::config::Config;
use engram_core::dispatch;
use engram_core::indexes::IndexSet;
use engram_core::persistence::{deterministic_rng, hash_string_to_u64};
use engram_core::server::{ServerState, reindex_unindexed_memories};
use engram_embeddings::{Embedder, ThreeFieldEmbedding};
use engram_llm_client::{EmbeddingProvider, TextGenerator};
use engram_router::Router;
use engram_storage::{Database, Memory};

// -------------------- tunable constants --------------------
// Increasing M_CONCURRENT_CLIENTS above 8 requires bumping worker_threads on both
// #[tokio::test] attributes to stay >= M_CONCURRENT_CLIENTS + 2 (reindex + headroom).
const N_MEMORIES_SEEDED: usize = 5000;
const N_MEMORIES_UNINDEXED_PER_TICK: usize = 100;
const M_CONCURRENT_CLIENTS: usize = 8;
const REINDEX_INTERVAL_MS: u64 = 500;
const BENCH_DURATION_SECS: u64 = 30;
// Warm up ALL N_MEMORIES_SEEDED unique query strings to fully prime the embedding
// cache before measurement (ADR 2026-04-24 intent: "all memory_search go through
// cache-hit"). Quick because deterministic provider is hash-based (no sleep).
const WARMUP_QUERIES: usize = N_MEMORIES_SEEDED;
const TOP_K: usize = 10;
const HISTOGRAM_SIGFIG: u8 = 3;
const MAX_TRACKABLE_LATENCY_US: u64 = 60_000_000; // 60s safety ceiling
const TIMESTAMP: &str = "2026-04-24T00:00:00Z";

// -------------------- fixture --------------------
fn build_bench_state() -> Arc<ServerState> {
    let database = Database::in_memory().expect("in-memory database");
    let mut config = Config::default();
    config.embedding.provider = "deterministic".into();
    let indexes = IndexSet::new(|| config.build_hnsw_params()).expect("index set");
    let embedder = Embedder::new(0);
    let router = Router::new(0.1, 0.15);
    let embedding_provider: Arc<dyn EmbeddingProvider + Send + Sync> = Arc::from(
        config
            .build_embedding_provider()
            .expect("embedding provider"),
    );
    let text_generator: Option<Arc<dyn TextGenerator + Send + Sync>> =
        config.build_text_generator().ok().map(Arc::from);
    Arc::new(ServerState {
        database: Mutex::new(database),
        indexes: RwLock::new(indexes),
        embedder,
        router: Mutex::new(router),
        config,
        database_path: String::new(),
        embedding_provider,
        text_generator,
    })
}

fn f32_vec_to_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn build_memory(index: usize, indexed: bool) -> (String, Memory, ThreeFieldEmbedding) {
    let id = format!("seed-memory-{index}");
    let embedding = ThreeFieldEmbedding {
        context: vec![0.1; 1024],
        action: vec![0.1; 1024],
        result: vec![0.1; 1024],
    };
    let memory = Memory {
        id: id.clone(),
        memory_type: "decision".into(),
        context: format!("seed context {index}"),
        action: format!("seed action {index}"),
        result: format!("seed result {index}"),
        score: 0.0,
        embedding_context: Some(f32_vec_to_bytes(&embedding.context)),
        embedding_action: Some(f32_vec_to_bytes(&embedding.action)),
        embedding_result: Some(f32_vec_to_bytes(&embedding.result)),
        indexed,
        tags: None,
        project: None,
        parent_id: None,
        source_ids: None,
        insight_type: None,
        created_at: TIMESTAMP.into(),
        updated_at: TIMESTAMP.into(),
        used_count: 0,
        last_used_at: None,
        superseded_by: None,
    };
    (id, memory, embedding)
}

// Seeds N memories into both Database (indexed=true) and IndexSet.
fn seed_memories(state: &Arc<ServerState>, n: usize) {
    // DB inserts (under a single lock to speed up).
    {
        let database = state.database.lock().unwrap();
        for i in 0..n {
            let (_, memory, _) = build_memory(i, true);
            database.insert_memory(&memory).expect("seed db insert");
        }
    }
    // HNSW inserts (under a single write-lock).
    {
        let mut indexes = state.indexes.write().unwrap();
        for i in 0..n {
            let (id, _, embedding) = build_memory(i, true);
            let hashed = hash_string_to_u64(&id);
            let rng = deterministic_rng(hashed);
            indexes
                .insert_atomic(hashed, &id, &embedding, rng)
                .expect("seed hnsw insert");
        }
    }
}

// Mark the last N rows as indexed=0 (for the reindex driver).
fn mark_last_rows_unindexed(state: &Arc<ServerState>, n: usize) {
    let database = state.database.lock().unwrap();
    for i in 0..n {
        let id = format!("seed-memory-{}", N_MEMORIES_SEEDED - 1 - i);
        let _ = database.set_memory_indexed(&id, false);
    }
}

// -------------------- harness --------------------
static QUERY_COUNTER: AtomicUsize = AtomicUsize::new(0);

fn next_query_string() -> String {
    let n = QUERY_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("seed context {}", n % N_MEMORIES_SEEDED)
}

async fn client_loop(
    state: Arc<ServerState>,
    stop: Arc<AtomicBool>,
    hist: Arc<Mutex<Histogram<u64>>>,
) {
    while !stop.load(Ordering::Relaxed) {
        let params = json!({ "query": next_query_string(), "limit": TOP_K });
        let start = Instant::now();
        let _ = dispatch::route("memory_search", &state, params).await;
        let latency_us = start.elapsed().as_micros() as u64;
        let _ = hist
            .lock()
            .unwrap()
            .record(latency_us.min(MAX_TRACKABLE_LATENCY_US));
    }
}

async fn reindex_driver(state: Arc<ServerState>, stop: Arc<AtomicBool>) {
    let interval = Duration::from_millis(REINDEX_INTERVAL_MS);
    while !stop.load(Ordering::Relaxed) {
        tokio::time::sleep(interval).await;
        if stop.load(Ordering::Relaxed) {
            break;
        }
        mark_last_rows_unindexed(&state, N_MEMORIES_UNINDEXED_PER_TICK);
        let state_clone = Arc::clone(&state);
        let _ = tokio::task::spawn_blocking(move || {
            reindex_unindexed_memories(&state_clone);
        })
        .await;
    }
}

async fn warm_up(state: &Arc<ServerState>) {
    for i in 0..WARMUP_QUERIES {
        let params = json!({ "query": format!("seed context {i}"), "limit": TOP_K });
        let _ = dispatch::route("memory_search", state, params).await;
    }
}

fn percentiles_summary(hist: &Histogram<u64>) -> String {
    let ms = |us: u64| us as f64 / 1000.0;
    format!(
        "p50={:.2}ms p95={:.2}ms p99={:.2}ms p99.9={:.2}ms max={:.2}ms samples={}",
        ms(hist.value_at_quantile(0.50)),
        ms(hist.value_at_quantile(0.95)),
        ms(hist.value_at_quantile(0.99)),
        ms(hist.value_at_quantile(0.999)),
        ms(hist.max()),
        hist.len(),
    )
}

async fn run_scenario(label: &str, with_reindex: bool) {
    let state = build_bench_state();
    seed_memories(&state, N_MEMORIES_SEEDED);
    warm_up(&state).await;

    let hist = Arc::new(Mutex::new(
        Histogram::<u64>::new_with_bounds(1, MAX_TRACKABLE_LATENCY_US, HISTOGRAM_SIGFIG)
            .expect("histogram"),
    ));
    let stop = Arc::new(AtomicBool::new(false));

    let mut handles = Vec::with_capacity(M_CONCURRENT_CLIENTS);
    for _ in 0..M_CONCURRENT_CLIENTS {
        handles.push(tokio::spawn(client_loop(
            Arc::clone(&state),
            Arc::clone(&stop),
            Arc::clone(&hist),
        )));
    }
    let reindex_handle = if with_reindex {
        Some(tokio::spawn(reindex_driver(
            Arc::clone(&state),
            Arc::clone(&stop),
        )))
    } else {
        None
    };

    tokio::time::sleep(Duration::from_secs(BENCH_DURATION_SECS)).await;
    stop.store(true, Ordering::Relaxed);

    for handle in handles {
        let _ = handle.await;
    }
    if let Some(h) = reindex_handle {
        let _ = h.await;
    }

    let final_hist = hist.lock().unwrap();
    println!("{label}: {}", percentiles_summary(&final_hist));
}

// -------------------- tests --------------------
// worker_threads = 10 = M_CONCURRENT_CLIENTS(8) + 2 (driver + headroom).
// If M_CONCURRENT_CLIENTS is raised, bump worker_threads accordingly.

#[ignore]
#[tokio::test(flavor = "multi_thread", worker_threads = 10)]
async fn search_latency_idle() {
    run_scenario("search_latency_idle", false).await;
}

#[ignore]
#[tokio::test(flavor = "multi_thread", worker_threads = 10)]
async fn search_latency_during_reindex() {
    run_scenario("search_latency_during_reindex", true).await;
}
