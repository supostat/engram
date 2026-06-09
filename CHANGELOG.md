# Changelog

## 0.5.0 (2026-06-09)

### Breaking changes
- None.

### New features
- **Local Ollama embedding provider.** Set `[embedding] provider = "ollama"`
  (e.g. `model = "qwen3-embedding:0.6b"` at 1024-dim) to produce real dense
  vectors against a local [Ollama](https://ollama.com) daemon with no API key.
  Switching an existing database is a one-time `engram reembed`; the `[6020]`
  guard enforces that stored vectors match the active provider.
- **Local Ollama text generator.** Set `[llm] provider = "ollama"` to run the
  LLM judge, HyDE, and consolidation on a local chat model (e.g. `qwen3:4b`)
  with no API key. An unreachable daemon falls back to the heuristic judge.
- **Ollama endpoint configuration** — new `[embedding].host` / `[llm].host`
  config fields and the `ENGRAM_OLLAMA_HOST` environment override select the
  Ollama host (default `http://localhost:11434`).
- **Ollama in the `engram init` TUI wizard** — selectable as both an embedding
  and an LLM provider, with no API-key prompt.

### Improvements
- **Search degrades to FTS5 when embeddings are unavailable** instead of
  failing the request. `memory_search` returns a `{ results, degraded }`
  envelope on both the healthy and degraded paths; `degraded: true` signals
  the vector branch was skipped (e.g. the embedding provider is unreachable).
- **Localhost-tuned retry backoff** for the Ollama HTTP clients — short, few
  retries — instead of reusing the cloud-latency budget. Provider
  documentation is pinned against the bundled config template and README by a
  docs-invariant test.

### Fixes
- None.

## 0.4.0 (2026-06-01)

### Breaking changes
- None.

### New features
- **Write-time deduplication.** Storing a memory whose context, action *and*
  result are each at least `dedup_threshold` (default `0.95` cosine) similar to
  an existing memory now folds into that memory — its `used_count` and
  `last_used_at` are bumped — instead of creating a duplicate row. The
  `memory_store` response gains additive `deduplicated` and `merged_into`
  fields. Configurable via `[deduplication] dedup_threshold`.
- **Configurable hybrid search** via a new `[search]` config block: `rrf_k`
  (default `60`), `vector_weight` (default `0.7`), and `sparse_weight` (default
  `0.3`).

### Improvements
- **Hybrid search now fuses results with Reciprocal Rank Fusion** —
  `1/(k + rank)`, rank-based and scale-free — instead of the previous max+sum
  of incomparably-scaled vector and sparse scores.
- **Diversity-aware HNSW neighbor selection** (Malkov-Yashunin heuristic, now
  the default; the naive top-M selector is retained and selectable). The new
  graph structure applies to existing data after `engram reindex`.
- **HNSW multi-graph insert is now all-or-nothing** — a failure rolls back the
  graphs already inserted for that memory. u64-hash collisions now fail loudly
  with `[6021]` instead of silently dropping a record. Lock access recovers
  from a poisoned lock instead of cascading a panic.
- **Router and chain terminology corrected** — adaptive contextual bandit (not
  "Q-Learning"), temporal and co-occurrence chains (not "causal").

### Fixes
- The `indexed` flag is now written `false` until the HNSW insert is confirmed,
  so a memory whose indexing fails is recovered by background reindex instead
  of being marked indexed.
- **Configuration is validated at startup.** An out-of-range `dedup_threshold`
  and invalid `[search]` config (`rrf_k = 0`, non-finite weights, or all-zero
  weights) now fail fast with `[6022] ConfigValidation`.

## 0.3.0 (2026-05-14)

### Breaking changes
- **Embedding default switched to `voyage-4`** (ADR 2026-05-14). Existing
  databases populated with `voyage-code-3` vectors now refuse to boot with
  `[6020] EmbeddingModelMismatch` until embeddings are recomputed. Run
  `engram reembed` once after upgrading, then restart the daemon.
  Dimension stays at `1024` — `voyage-4` API default matches, so the
  on-disk HNSW geometry is unchanged. The voyage-4 family (large/regular/
  lite/nano) shares one embedding space, so subsequent switches within
  the family will not require another reembed.
- **`EmbeddingProvider::embed` trait signature** gained an `input_type:
  Option<&str>` parameter. Voyage providers now send `"document"` on the
  store path and `"query"` on the search path; other implementors
  (`DeterministicEmbeddingProvider`, custom mocks) ignore the value but
  must update their method signature. Pre-1.0 deliberate break — the
  trait is internal-only and has one real implementor.

### New features
- **`engram reembed [--force]` CLI** recomputes embeddings for every
  stored memory with the currently configured provider. Replaces vectors
  in HNSW (delete + insert per memory), writes fresh BLOBs to SQLite with
  `indexed=true`, and records the active model in `schema_meta` only when
  every memory succeeded. Per-memory provider/HNSW failure flips
  `indexed=0` so background reindex retries; database failure propagates.
- **Server startup guard** against silent model drift. `server::run`
  reads `schema_meta.embedding_model` after `initialize_state` and
  refuses to start with `[6020]` if it differs from the configured
  `embedding.model`. Bootstrap on an empty marker writes the configured
  model; if the database already contains memories at bootstrap time
  (legacy upgrade path), a stderr warning recommends running `engram
  reembed` first.
- **`embedding.output_dimension` config knob** for Voyage-4 Matryoshka
  truncation (256/512/1024/2048). Omit to let the API use its default
  (1024). Must match `[hnsw].dimension` when set.
- **`EmbeddingCache` keyed by `(text, input_type)`** instead of text
  alone, so a memory stored as a document and a query of the same text
  no longer collide on a single cache slot.

### Migration

Upgrading from 0.2.x with existing memories:

```bash
engram server     # fails with [6020] EmbeddingModelMismatch — expected
engram reembed    # recomputes all embeddings with voyage-4
engram server     # now starts cleanly
```

The reembed run takes roughly 50ms × 3 fields × N memories under
Voyage API latency; budget a few minutes for a thousand records.

### Notes
- Error codes 6001-6019 were already in use; this release adds **6020
  EmbeddingModelMismatch**.

## 0.2.0 (2026-05-10)

### Breaking changes
- **Per-project state layout** (ADR 2026-04-22). Database and unix socket moved
  from `~/.engram/{engram.db, engram.sock}` to `<project>/.engram/{engram.db,
  engram.sock}`. Discovery walks up from cwd looking for `.engram/`, like
  `.git`. `~/.engram/` now holds only `engram.toml` (API keys, defaults) and
  `models/` (ONNX artifacts shared across projects). `engram server` aborts on
  startup when it finds a legacy `~/.engram/engram.db` but no per-project
  database — see `engram migrate` below to import.
- **HyDE is now opt-in** (ADR 2026-05-05). The new `embedding.hyde_threshold`
  config field defaults to `0` (disabled). Set it to a positive integer `N`
  to enable HyDE for queries shorter than `N` words. The embedding cache is
  now keyed by the original query (not the generated hypothesis), so repeated
  identical queries hit the cache without an extra LLM call.

### New features
- **engram-tui** — terminal dashboard built on ratatui with five tabs
  (Status, Memories, Search, Q-Learning, Models), interactive init wizard,
  and contextual hints. Connects to the server via unix socket and reads
  SQLite directly.
- **`engram migrate` CLI** — import memories from a legacy global
  `~/.engram/engram.db` into the current project's `.engram/engram.db`.
  Default filter: rows whose `project` field matches the cwd basename
  (`--all` includes NULL/mismatched rows; `--dry-run` previews only).
- **Tag-based filtering on `memory_search`** — new optional `tags` array
  parameter restricts results to memories carrying all specified tags.
- **Documentation site** — bilingual EN/RU docs scaffolded with Next.js +
  Fumadocs, deployed to GitHub Pages with a 3D landing page.

### Improvements
- **Concurrency hardening** (ADR 2026-04-24). Replaced `Mutex<IndexSet>` with
  `RwLock<IndexSet>` so parallel `memory_search` calls no longer serialize on
  the index. Removed `Mutex<Embedder>` for the same reason. LLM clients are
  cached in `ServerState` to reuse HTTP connections instead of rebuilding per
  request.
- **Schema migrations are idempotent at startup** (ADR 2026-05-01).
  Introduced `schema_meta` table with `tags_format = json_array_v1` flag;
  `memories.tags` are normalized to canonical JSON array on first run.
- **Reliability** — added 10s HTTP timeout to OpenAI and Voyage clients;
  bumped socket request timeout to 30s; sanitized FTS5 queries to prevent
  syntax errors on special characters; added `max_tokens` to OpenAI HyDE
  requests to prevent runaway responses.
- **Hardened MCP server lifecycle** with socket-client tests and CI coverage.
- **Opt-in latency harness** (`hyperfine`-style p99 measurements) for
  diagnosing search latency under reindex.

### Fixes
- Vector search results were silently discarded in hybrid scoring under some
  conditions — now correctly merged.
- `engram-tui` per-project discovery now matches the server's discovery rules
  (ADR 2026-04-22).
- Init wizard correctly detects API keys on the status screen.
- `init_handler::execute_with_dirs` no longer triggers the interactive
  `engram-tui init` wizard; the TTY-attached check moved up to `execute()`.
  Previously the wizard ran inside `execute_with_dirs`, which made every
  init test hang under any TTY-attached test runner (e.g. `lefthook
  pre-push`) by spawning an interactive subprocess.

### Documentation
- README, AGENT.md, and `website/content/docs/{en,ru}` brought into sync with
  current code (per-project layout, `engram migrate`, HyDE opt-in,
  `embedding.hyde_threshold`, `tags` filter).
- Landing page hero stats updated (9 crates, 470+ tests) and a TUI Dashboard
  feature card added.
- Default `engram.toml` template (written by `engram init`) now documents
  fallback semantics for `database.path` / `server.socket_path` and includes
  `hyde_threshold = 0`.

### Tooling / build
- Added [lefthook](https://github.com/evilmartians/lefthook) for pre-commit
  (`cargo fmt`, `clippy`, typecheck) and pre-push (`cargo test`, vitest,
  `npm run build`) hooks.
- All Rust sources formatted with `cargo fmt`.
- Internal crate dependencies now declare `version` alongside `path` so
  the workspace is publishable to crates.io.
- Bumped CI Node to 22; fixed tailwind native binding error in deploy.

## 0.1.0 (2026-04-05)

Initial release.

### Features
- HNSW approximate nearest neighbor search (engram-hnsw)
- Q-Learning adaptive router (engram-router)
- SQLite storage with FTS5 full-text search (engram-storage)
- LLM client with Voyage, OpenAI, and local ONNX support (engram-llm-client)
- Three-field embedding with HyDE (engram-embeddings)
- Heuristic and LLM-based quality scoring (engram-judge)
- Memory consolidation: dedup, analysis, apply (engram-consolidate)
- CLI and unix socket server (engram-core)
- MCP server for AI agent integration (@engramm/engram-mcp-server)
- Self-learning trainer with clustering, temporal analysis, LoRA fine-tuning (engram-trainer)
