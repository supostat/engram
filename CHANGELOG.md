# Changelog

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
- MCP server for AI agent integration (@engram/mcp-server)
- Self-learning trainer with clustering, temporal analysis, LoRA fine-tuning (engram-trainer)
