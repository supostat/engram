# Engram — Agent Instructions

Engram is a memory system for AI agents. Use it to store decisions, patterns, and bugfixes, then search them to inform future work.

## MCP Tools

### Core Operations

**memory_store** — Save a new memory.
```json
{
  "context": "Situation where the action occurred",
  "action": "Decision or action taken",
  "result": "Outcome or result",
  "memory_type": "decision",
  "tags": "auth,refactor",
  "project": "my-project"
}
```
**Выбор memory_type:**

| Тип | Когда использовать | Пример |
|-----|-------------------|--------|
| `decision` | Архитектурный/дизайн-выбор с обоснованием | "Выбрали REST вместо GraphQL потому что..." |
| `pattern` | Переиспользуемое решение, применимое к будущим задачам | "Svelte 5 stores через .svelte.ts с $state runes" |
| `bugfix` | Диагностика и исправление конкретного бага | "Токены истекали молча → добавили middleware" |
| `context` | Факт о проекте, статус фазы, конфигурация | "Phase B завершена, 124 теста, 17 файлов" |
| `antipattern` | Что НЕ делать и почему | "Не мокать БД в интеграционных тестах — прод миграция сломалась" |

**memory_search** — Find relevant memories using hybrid vector + sparse search.
```json
{
  "query": "authentication middleware architecture",
  "limit": 10,
  "project": "my-project"
}
```

**memory_judge** — Rate a memory's quality. Feeds the Q-Learning router to improve future search.
```json
{
  "memory_id": "uuid",
  "query": "the query that found this memory",
  "score": 0.8
}
```

**memory_status** — Get system health: memory count, index size, pending judgments.
```json
{}
```

**memory_config** — Read current configuration.
```json
{
  "action": "get"
}
```

### Import / Export

**memory_export** — Export all active memories as portable JSON (excludes embeddings).
```json
{}
```

**memory_import** — Import memories from exported JSON. Merge mode: skips duplicates by ID.
```json
{
  "version": 1,
  "memories": [...]
}
```

### Consolidation

**memory_consolidate_preview** — Find deduplication candidates without changes.
```json
{
  "stale_days": 30,
  "min_score": 0.3
}
```

**memory_consolidate** — Analyze candidates with LLM. Returns merge/keep recommendations.
```json
{
  "stale_days": 30,
  "min_score": 0.3
}
```

**memory_consolidate_apply** — Apply recommendations: merge, delete, or archive.
```json
{
  "stale_days": 30,
  "min_score": 0.3
}
```

### Insights

**memory_insights** — List, generate, or delete derived knowledge.
```json
{
  "action": "list"
}
```
Actions: `list` (show insights), `generate` (run trainer analysis), `delete` (remove by ID with `"id": "uuid"`).

## Usage Patterns

### Store after completing work

After solving a bug, making an architecture decision, or discovering a pattern:

```
memory_store({
  context: "User auth tokens expired silently, no error in logs",
  action: "Added explicit token validation middleware with structured error logging",
  result: "Auth failures now surface immediately with clear error codes",
  memory_type: "bugfix",
  tags: "auth,middleware,logging",
  project: "api-server"
})
```

### Search before starting work

Before implementing a feature or debugging:

```
memory_search({ query: "authentication middleware patterns", limit: 5 })
```

### Feedback loop

After search returns results, judge the ones you used:

```
memory_judge({ memory_id: "abc-123", query: "auth middleware", score: 0.9 })
```

This trains the Q-Learning router to improve ranking for future queries.

### Consolidation workflow

Periodically clean up duplicate and stale memories:

```
memory_consolidate_preview({})           → see candidates
memory_consolidate({})                   → get LLM recommendations
memory_consolidate_apply({})             → apply merges/deletes
```

## Search Behavior

- **Hybrid search**: 70% vector similarity (HNSW cosine) + 30% sparse match (BM25 via FTS5)
- **HyDE**: for complex queries, LLM generates a hypothetical memory, then embeds the hypothesis
- **Cross-project**: searches current project first, applies score multiplier for other projects
- **Graceful degradation**: if embedding API is unavailable, falls back to FTS5-only search
