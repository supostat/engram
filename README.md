# Engram

Memory system for AI agents. Stores decisions, patterns, and bugfixes with semantic search, automatic deduplication, and self-learning optimization.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Claude Desktop / Claude Code / Cursor              │
│  (MCP client)                                       │
└──────────────┬──────────────────────────────────────┘
               │ MCP protocol (stdio)
┌──────────────▼──────────────────────────────────────┐
│  @engramm/engram-mcp-server          (TypeScript)           │
│  Thin translation layer: MCP ↔ JSON-RPC             │
└──────────────┬──────────────────────────────────────┘
               │ Unix socket (JSON, newline-delimited)
┌──────────────▼──────────────────────────────────────┐
│  engram-core                 (Rust)                 │
│  Server mode: tokio, dispatch, background tasks     │
│  CLI mode: self-contained, no running server needed │
├─────────────────────────────────────────────────────┤
│  engram-storage    SQLite + FTS5, WAL mode          │
│  engram-hnsw       HNSW vector index (cosine)       │
│  engram-embeddings Voyage/OpenAI + HyDE             │
│  engram-judge      Heuristic + LLM scoring          │
│  engram-consolidate Dedup preview/analyze/apply     │
│  engram-router     Q-Learning (4 levels)            │
│  engram-llm-client API + local ONNX inference       │
└─────────────────────────────────────────────────────┘
               │ stdout JSON Lines
┌──────────────▼──────────────────────────────────────┐
│  engram-trainer              (Python)               │
│  Clustering, temporal analysis, ONNX model export   │
└─────────────────────────────────────────────────────┘
```

Two-process model:
- **Rust core** — long-lived unix socket server OR self-contained CLI
- **TypeScript MCP server** — thin translator, manages Rust core lifecycle

## Quick Start

### CLI

```bash
cargo install engram-memory --locked
engram init
engram store --context "Refactored auth module" --action "Split into middleware + handler" --result "Reduced coupling, easier to test"
engram search --query "auth architecture"
engram status
```

### TUI Dashboard

```bash
cargo install engram-tui --locked
engram-tui
```

Terminal dashboard with 5 tabs: Status, Memories, Search, Q-Learning, Models. Includes an interactive init wizard for first-time setup.

### MCP Server

Add to your Claude Desktop / Claude Code config:

```json
{
  "mcpServers": {
    "engram": {
      "command": "npx",
      "args": ["-y", "@engramm/engram-mcp-server"]
    }
  }
}
```

### Trainer

```bash
pip install engram-trainer

# Direct invocation (point --database at the per-project DB)
python -m engram_trainer --database .engram/engram.db --models-path ~/.engram/models/
python -m engram_trainer --database .engram/engram.db --models-path ~/.engram/models/ --deep

# Via CLI (requires engram-core server running)
engram train
engram train --deep  # LoRA fine-tuning (requires torch)
```

## Features

- **Hybrid search** — HNSW vector (cosine) + BM25 sparse (FTS5), alpha-weighted
- **HyDE** — opt-in via `embedding.hyde_threshold > 0` (disabled by default). When enabled, LLM generates a hypothetical memory and embeds the hypothesis instead of the raw query. Cache is keyed by the original query, so repeated calls hit the cache instantly.
- **Automatic deduplication** — cosine similarity > 0.95 at write time
- **Consolidation** — preview → LLM analysis → apply (merge/delete/archive)
- **Q-Learning router** — 4 levels: search strategy, LLM selection, contextualization, proactivity
- **Three learning loops** — fast (Q-Learning per call), medium (trainer daily/weekly), deep (LoRA fine-tune)
- **Cross-project transfer** — project-scoped search with score multiplier, insights are project-agnostic
- **Graceful degradation** — every API dependency has a local fallback (FTS for search, heuristics for judge)
- **Local inference** — optional ONNX runtime for text generation (feature-gated)

## Self-Learning Models

Trainer produces three ONNX models stored in `~/.engram/models/`:

| Model | Size | Algorithm | Purpose |
|-------|------|-----------|---------|
| `mode_classifier.onnx` | 13 KB | TF-IDF + LogisticRegression | Classifies query type (query/research/brainstorm/debugging) for Q-Learning router |
| `ranking_model.onnx` | 23 KB | GradientBoosting | Re-ranks search results by score, usage, recency, length, and tags |
| `text_generator.onnx` | ~312 MB | DistilGPT2 + LoRA | Local text generation replacing API calls for HyDE and routine operations |

The first two models are trained during regular `engram train` runs. The text generator requires `engram train --deep` with PyTorch installed.

## Memory Types

| Type | Purpose |
|------|---------|
| `decision` | Architecture and design decisions |
| `pattern` | Recurring solutions and approaches |
| `bugfix` | Bug diagnoses and fixes |
| `context` | Project context and setup knowledge |
| `antipattern` | What NOT to do, with reasoning |
| `insight` | Derived knowledge from trainer analysis |

## Project Structure

```
crates/
  engram-hnsw/          HNSW approximate nearest neighbor search
  engram-router/        Hierarchical Q-Learning router
  engram-storage/       SQLite storage with FTS5
  engram-llm-client/    LLM provider traits (Voyage, OpenAI, local ONNX)
  engram-embeddings/    Three-field embedding with HyDE
  engram-judge/         Heuristic and LLM-based scoring
  engram-consolidate/   Memory consolidation pipeline
  engram-core/          CLI + server binary
  engram-tui/           Terminal dashboard (ratatui)
mcp-server/             TypeScript MCP server
trainer/                Python self-learning trainer
website/                Documentation site (Fumadocs)
```

## Configuration

Engram splits state across two locations:

- **Global** (`~/.engram/`) — `engram.toml` (API keys, defaults), `models/` (ONNX artifacts shared across projects).
- **Per-project** (`<project>/.engram/`) — `engram.db` (SQLite), `engram.sock` (Unix socket). Discovery walks up from cwd looking for `.engram/`, similar to `.git`.

`~/.engram/engram.toml`:

```toml
[database]
# Fallback only — runtime prefers per-project <project>/.engram/engram.db
# (or ENGRAM_DB_PATH override). This value is used only when no .engram/
# marker is found while walking up from cwd.
path = "~/.engram/memories.db"

[embedding]
provider = "voyage"        # voyage | deterministic
model = "voyage-4"
# output_dimension is optional. Omit to use the Voyage API default (1024,
# which matches [hnsw].dimension below). Set to one of 256/512/1024/2048
# for Voyage-4 Matryoshka truncation — must match [hnsw].dimension.
hyde_threshold = 0          # 0 = HyDE disabled (default); N>0 = enable for queries shorter than N words

[llm]
provider = "openai"         # openai | local
model = "gpt-4o-mini"

[server]
# Fallback only — runtime prefers <project>/.engram/engram.sock
# (or ENGRAM_SOCKET_PATH override) when a project .engram/ exists.
socket_path = "~/.engram/engram.sock"
reindex_interval_secs = 3600

[hnsw]
max_connections = 16
ef_construction = 200
ef_search = 40
dimension = 1024

[consolidation]
stale_days = 90
min_score = 0.3

[trainer]
trainer_binary = "engram-trainer"
trainer_timeout_secs = 300
models_path = "~/.engram/models"
```

Environment variables override config: `ENGRAM_VOYAGE_API_KEY`, `ENGRAM_OPENAI_API_KEY`, `ENGRAM_DB_PATH`, `ENGRAM_SOCKET_PATH`, `ENGRAM_EMBEDDING_MODEL`, `ENGRAM_LLM_MODEL`, `ENGRAM_TRAINER_BINARY`, `ENGRAM_TRAINER_TIMEOUT`, `ENGRAM_MODELS_PATH`.

### Migrating from a previous global install

If you have memories in `~/.engram/engram.db` from before per-project layout, import them with:

```bash
cd /path/to/project
engram migrate --dry-run   # preview what would be imported
engram migrate             # default: only rows whose project field matches the cwd basename
engram migrate --all       # also import NULL-project / mismatched rows
```

`engram server` aborts on startup if it finds a legacy `~/.engram/engram.db` but no `<project>/.engram/engram.db` — the error message points at this command.

### Upgrading 0.2.x → 0.3.x (embedding model)

Engram 0.3.0 switches the default embedding model from `voyage-code-3` to `voyage-4`. Existing databases were embedded with the old model and refuse to boot until they are recomputed:

```bash
engram server     # fails: [6020] EmbeddingModelMismatch
engram reembed    # recomputes every memory under the active provider
engram server     # now starts cleanly
```

`engram reembed` walks every row, calls the active embedding provider, replaces the vector in HNSW, writes the new bytes back to SQLite, and records the model in `schema_meta.embedding_model`. A failed run (some memories the provider rejected) leaves the marker stale so the daemon keeps refusing to start until reembed finishes cleanly — rerun it to retry only the leftovers (`indexed=0`).

If you want to keep the old behaviour, set `embedding.model = "voyage-code-3"` in `engram.toml` before the first 0.3 boot — the guard checks the configured model against `schema_meta`, not the bundled default.

Dimension stays at `1024` because `voyage-4` API default matches; HNSW geometry survives the migration. Within the voyage-4 family (`voyage-4-large`, `voyage-4`, `voyage-4-lite`, `voyage-4-nano`) embeddings share one space, so future switches inside the family do not require another reembed.

## Testing

```bash
cargo test --all              # 470+ Rust tests
cd mcp-server && npm test     # 7 vitest unit tests + typecheck
cd trainer && pytest           # Python tests (pip install -e ".[dev]")
cargo bench --all             # Criterion benchmarks
```

[Lefthook](https://github.com/evilmartians/lefthook) is configured for git hooks: `pre-commit` runs `cargo fmt`, `clippy`, and `typecheck`; `pre-push` runs `cargo test`, `vitest`, and `npm run build`.

## Documentation

Full documentation available at [supostat.github.io/engram](https://supostat.github.io/engram/):
- [Russian](https://supostat.github.io/engram/ru/docs)
- [English](https://supostat.github.io/engram/en/docs)

To run locally:

```bash
cd website && npm install && npm run dev
```

## License

MIT
