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
│  @engram/mcp-server          (TypeScript)           │
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
cargo install engram-core --locked
engram init
engram store --context "Refactored auth module" --action "Split into middleware + handler" --result "Reduced coupling, easier to test"
engram search --query "auth architecture"
engram status
```

### MCP Server

Add to your Claude Desktop / Claude Code config:

```json
{
  "mcpServers": {
    "engram": {
      "command": "npx",
      "args": ["-y", "@engram/mcp-server"]
    }
  }
}
```

### Trainer

```bash
pip install engram-trainer

# Direct invocation
python -m engram_trainer --database ~/.engram/memories.db --models-path ~/.engram/models/
python -m engram_trainer --database ~/.engram/memories.db --models-path ~/.engram/models/ --deep

# Via CLI
engram train
engram train --deep  # LoRA fine-tuning (requires torch)
```

## Features

- **Hybrid search** — HNSW vector (cosine) + BM25 sparse (FTS5), alpha-weighted
- **HyDE** — LLM generates hypothetical memory, embeds hypothesis instead of raw query
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
mcp-server/             TypeScript MCP server
trainer/                Python self-learning trainer
```

## Configuration

`~/.engram/engram.toml`:

```toml
[database]
path = "~/.engram/memories.db"

[embedding]
provider = "voyage"     # voyage | deterministic
model = "voyage-code-3"

[llm]
provider = "openai"     # openai | local
model = "gpt-4o-mini"

[server]
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

## Testing

```bash
cargo test --all              # 350 Rust tests
cd mcp-server && npm test     # 7 vitest unit tests + typecheck
cd trainer && pytest           # Python tests (pip install -e ".[dev]")
cargo bench --all             # Criterion benchmarks
```

## Documentation

Full documentation available at the website:
- Russian: /ru/docs
- English: /en/docs

To run locally:

```bash
cd website && npm install && npm run dev
```

## License

MIT
