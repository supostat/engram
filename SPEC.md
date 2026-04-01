# Engram — Спецификация

## 1. Обзор проекта

**Engram** — библиотека долгосрочной памяти для автоматизации разработки. Хранит опыт между сессиями и предоставляет его AI-агентам и внешним системам.

Основной сценарий: AI-агент (Claude Desktop, Claude Code, Cursor) обращается к Engram за прошлым опытом при решении новой задачи. Engram находит релевантные записи, возвращает их как контекст, а после выполнения задачи сохраняет новый опыт и обновляет оценки.

**Три точки входа:**

- **MCP (HTTP+SSE)** — для AI-агентов (Claude Desktop, Cursor, Claude Code). Основной интерфейс для агентов.
- **CLI** — самодостаточный, для скриптов, CI/CD, бэкенд-интеграций (Node, Go, PHP, bash). Каждая команда возвращает результат в выбранном формате (json, text, jsonl). Не требует запущенного ядра — работает напрямую с базой.
- **Unix socket** — для кастомных клиентов, максимальная производительность. JSON-протокол, описан в разделе 2.2.

---

## 2. Архитектура

### 2.1 Общая схема

Система состоит из двух процессов:

- **Rust-ядро** — хранилище, HNSW-индекс, Q-Learning роутер, логика оценки и консолидации. Долгоживущий процесс, слушает unix socket.
- **TypeScript MCP-сервер** — тонкий слой, реализует MCP-протокол через HTTP+SSE. Транслирует MCP-вызовы в команды Rust-ядру через unix socket.

```
MCP-клиент (Claude Desktop / Cursor / Claude Code)
        │
        │  HTTP + SSE (MCP-протокол)
        ▼
TypeScript MCP-сервер (@engram/mcp-server)
        │
        │  Unix socket + JSON
        ▼
Rust-ядро (engram-core)
        │
        ├── engram-storage (SQLite + FTS5)
        ├── engram-hnsw (самописный индекс)
        ├── engram-router (иерархический Q-Learning)
        ├── engram-judge (оценка результатов)
        ├── engram-consolidate (очистка и анализ)
        ├── engram-embeddings (генерация векторов)
        └── engram-llm-client (общий клиент для LLM API)
```

### 2.2 Коммуникация между процессами

- **Протокол:** Unix socket (`engram.sock`)
- **Формат:** JSON, каждое сообщение — одна JSON-строка, разделитель `\n`
- **Паттерн:** request-response, Rust-ядро — сервер, TypeScript — клиент

Формат запроса:
```json
{
  "id": "uuid",
  "method": "memory_search",
  "params": { }
}
```

Формат ответа:
```json
{
  "id": "uuid",
  "result": { }
}
```

Формат ошибки:
```json
{
  "id": "uuid",
  "error": {
    "code": 1001,
    "message": "описание ошибки",
    "recoverable": true
  }
}
```

**Коды ошибок:**

| Код | Категория | Описание | Recoverable |
|-----|-----------|----------|-------------|
| 1001 | Storage | Database unavailable | fatal |
| 1002 | Storage | Record not found | recoverable |
| 1003 | Storage | Duplicate detected | recoverable |
| 1004 | Storage | Migration required | fatal |
| 2001 | API | Embedding API unavailable | recoverable (degraded mode) |
| 2002 | API | LLM API unavailable | recoverable (degraded mode) |
| 2003 | API | Rate limit exceeded (after all retries) | recoverable |
| 2004 | API | Invalid API key | fatal |
| 2005 | API | HyDE generation failed | recoverable (search without HyDE) |
| 3001 | HNSW | Index corrupted | recoverable (rebuild) |
| 3002 | HNSW | Dimension mismatch | fatal |
| 3003 | HNSW | Index rebuild required | recoverable |
| 4001 | Router | Unknown mode | recoverable (fallback to default) |
| 4002 | Router | Mode detection failed | recoverable (use default) |
| 5001 | Consolidation | No candidates found | recoverable |
| 5002 | Consolidation | LLM analysis failed | recoverable |
| 5003 | Consolidation | Apply conflict | recoverable |
| 6001 | Training | Trainer not installed | recoverable |
| 6002 | Training | Training failed | recoverable |
| 6003 | Training | Invalid artifact | recoverable |

Recoverable ошибки: система продолжает работать, возможно в degraded mode. Fatal ошибки: операция невозможна, нужно вмешательство.

### 2.3 Жизненный цикл процессов

MCP-сервер управляет жизненным циклом Rust-ядра:

1. При запуске `@engram/mcp-server` проверяет наличие unix socket (`engram.sock`)
2. Если socket не существует — MCP-сервер запускает Rust-ядро (`engram-core`) как дочерний процесс
3. Ждёт появления socket (с таймаутом)
4. Подключается и начинает обслуживать MCP-запросы
5. При завершении MCP-сервера — отправляет команду graceful shutdown в Rust-ядро
6. Rust-ядро завершает текущие операции, сбрасывает данные на диск, удаляет socket файл

При аварийном завершении:
- Rust-ядро: SQLite в WAL mode гарантирует целостность данных. Записи с `indexed = false` доиндексируются при следующем старте. Stale socket файл удаляется при следующем запуске.
- MCP-сервер: Rust-ядро продолжает работать как сиротский процесс. При следующем подключении MCP-сервер обнаруживает живой socket и переподключается.

### 2.4 Логирование

- Библиотека: `tracing`
- Формат: plain text с timestamp и уровнем
- Путь: `~/.engram/logs/engram.log`
- Ротация: по размеру 10 МБ, хранить последние 5 файлов
- Уровни: ERROR (сбои), WARN (rate limit, fallback), INFO (запуск, shutdown, consolidation), DEBUG (каждый MCP-вызов, решения роутера), TRACE (содержимое запросов/ответов)
- Уровень по умолчанию: INFO, настраивается через `engram.toml` или `ENGRAM_LOG_LEVEL`
- `engram check` и `engram status` читают лог для диагностики последних ошибок

### 2.5 Graceful degradation

При недоступности внешних API система продолжает работать в ограниченном режиме:

- **`memory_store`**: запись сохраняется с `indexed = false` и пустым embedding. FTS-индекс обновляется. Ответ содержит `degraded: true`. Фоновая задача доиндексирует при восстановлении API.
- **`memory_search`**: embedding API недоступен → fallback на SQLite FTS5 полнотекстовый поиск. HyDE LLM недоступен → поиск без HyDE обычным эмбеддингом запроса. Ответ содержит `degraded: true`, `search_method`, `hyde_used`.
- **`memory_judge`** (режим LLM): автоматический fallback на эвристики с `degraded: true`.
- **`memory_consolidate_preview`**: работает нормально — не использует API.
- **`memory_consolidate`** (анализ): возвращает ошибку (без LLM анализ бессмыслен). Preview данные сохраняются.
- **`memory_consolidate_apply`**: не зависит от API, работает всегда.

### 2.6 Ограничения текущей версии

> **TODO: Конкурентный доступ.** Текущая архитектура рассчитана на одного MCP-клиента. Несколько клиентов (Claude Desktop + Cursor одновременно) могут подключиться к Rust-ядру, но потокобезопасность HNSW-индекса и корректность конкурентной записи не гарантированы. Требует RwLock на индекс и WAL mode в SQLite. Отложено до появления реальной потребности.

> **TODO: Размер контекста.** При memory_search результаты целиком попадают в промпт LLM-клиента. Нет контроля суммарного размера возвращаемых записей. В будущем: ограничение на размер записи при store, суммаризация при возврате, параметр max_tokens в search.

> **TODO: Кросс-пользовательское обучение.** В v1 шаринг опыта между разработчиками — через export/import инсайтов. В будущем: общий проектный Engram (уровень 2) и федеративное обучение без раскрытия сырых данных (уровень 3).

> **TODO: Безопасность.** API-ключи в `engram.toml` и env vars. База содержит код и решения — потенциально чувствительная информация. В будущем: шифрование данных at rest (SQLite SEE или application-level encryption), ограничение доступа к unix socket (file permissions), keychain-интеграция для API-ключей.

> **TODO: Миграции схемы.** При обновлении версии Engram схема БД может измениться. В будущем: версионирование схемы в таблице `schema_version`, автоматический `ALTER TABLE` при запуске, откат при ошибке.

---

## 3. CLI

Самодостаточный CLI. Каждый вызов — отдельный процесс, работает напрямую с базой, не требует запущенного Rust-ядра. Все команды поддерживают флаг `--format` (json | text | jsonl), по умолчанию json.

### 3.1 Bootstrap

**`engram init`** — первичная настройка. Создаёт `engram.toml`, `~/.engram/`, запрашивает API-ключ Voyage, создаёт SQLite базу с FTS-индексом. Выводит сниппет для MCP-клиента.

Холодный старт — выбор пресетов:
```
engram init
> Выберите пресет (или пропустите):
> 1. node-backend
> 2. react-frontend
> 3. rust-cli
> 4. python-backend
> 5. Пропустить
> Можно указать несколько через запятую: 1,2
```

Или неинтерактивно: `engram init --preset node-backend,react-frontend`

Без сети — пресеты пропускаются, система работает с пустой базой.

**`engram check`** — диагностика. Конфиг, база, API-ключи, socket. Последние ошибки из лога.

**`engram status`** — инфраструктура + содержимое. Pid, uptime, размер базы, количество записей по типам, avg score, pending indexing, router stats.

**`engram version`** — версия.

### 3.2 Работа с памятью

Зеркалит MCP-инструменты. Выход — JSON (или text/jsonl при соответствующем --format).

**store:**
```
engram store --type bugfix --context "..." --action "..." --result "..." [--score 0.8] [--tags tag1,tag2] [--project myapp] [--mode debug] [--format json]
```

**search:**
```
engram search "cors bug" [--mode debug] [--type bugfix] [--project myapp] [--top-k 5] [--min-score 0.3] [--format json]
```

**judge:**
```
engram judge <memory_id> --score 0.8 [--mode debug] [--format json]
engram judge <memory_id> --outcome "тест прошёл" [--mode llm] [--format json]
```

### 3.3 Конфигурация и экспорт

**config:**
```
engram config get [key]
engram config set <key> <value>
```

**export:**
```
engram export [--format json|sqlite] [--project myapp] [--path ./backup.json]
```

**import:**
```
engram import <path> [--strategy merge|replace]
```

### 3.4 Консолидация

```
engram consolidate preview [--similarity-threshold 0.95] [--stale-days 90]
engram consolidate analyze [--ids id1,id2,id3]
engram consolidate apply --actions '<json>'
```

### 3.5 Инсайты и обучение

```
engram insights list [--type cluster|temporal|causal] [--format json]
engram insights generate [--type bugfix] [--format json]
engram insights delete <id>
engram train [--format json]
engram train --deep [--format json]
```

### 3.6 Персистентный HNSW

CLI работает как самодостаточный процесс, поэтому три HNSW-индекса (context, action, result) сериализуются на диск в директорию `~/.engram/indexes/`.

При CLI-вызове:
1. Проверить наличие файлов индексов
2. Сравнить timestamp с последней записью в SQLite
3. Если актуальные — загрузить с диска (быстро)
4. Если устарели — доиндексировать записи с `indexed = false`
5. Если нет файлов — полный rebuild всех трёх индексов
6. После модифицирующих операций (store, import, consolidate apply) — сохранить обновлённые индексы на диск

Это требует от `engram-hnsw` обязательной поддержки сериализации/десериализации на диск.

### 3.7 Пресеты (холодный старт)

Пресеты решают проблему пустой базы. Содержат общие паттерны, антипаттерны и инсайты для конкретного стека.

**Реестр:** отдельный GitHub-репозиторий `engram-presets`. На первое время только официальные пресеты.

**Формат пресета** — JSON, совместимый с `memory_export`/`memory_import`:
```json
{
  "name": "node-backend",
  "version": "1.0.0",
  "description": "Node.js backend (Express/Fastify)",
  "memories": [
    {
      "memory_type": "pattern",
      "context": "...",
      "action": "...",
      "result": "...",
      "tags": ["express", "architecture"]
    }
  ]
}
```

Эмбеддинги не хранятся в пресете — генерируются при импорте (зависят от провайдера).

**Содержимое пресета (три слоя):**
- Общие инсайты — не зависят от стека (тестирование, git-практики, логирование)
- Стек-специфичные паттерны — структура кода, типичные решения
- Стек-специфичные антипаттерны — распространённые ошибки

**Работа с реестром:**
- `engram init` скачивает `index.json` со списком доступных пресетов
- Пользователь выбирает интерактивно или через `--preset`
- Скачивается конкретный файл пресета
- Импортируется через стандартную логику memory_import
- Без сети — пресеты пропускаются

---

## 4. Модульная структура (Cargo workspace)

### 4.1 Граф зависимостей

```
engram-hnsw          (ни от чего)
engram-router        (ни от чего)
engram-storage       (rusqlite, serde)
engram-llm-client    (reqwest, serde — трейты + реализации)
engram-embeddings    (engram-llm-client)
engram-judge         (engram-llm-client)
engram-consolidate   (engram-storage, engram-hnsw, engram-llm-client)
engram-core          (все крейты выше — склейка, unix socket сервер, CLI)
```

### 4.2 Крейты

#### engram-hnsw

Самописная реализация Hierarchical Navigable Small World графа.

- Структура данных для approximate nearest neighbor поиска
- Операции: insert, search (top-k), delete
- Метрика: cosine similarity
- **Обязательная** сериализация/десериализация на диск (требуется для самодостаточного CLI)
- При запуске в режиме ядра: доиндексирует записи с `indexed = false` из SQLite
- При запуске в режиме CLI: загружается с диска, доиндексирует если устарел
- Не зависит от остальных крейтов, может использоваться отдельно
- `engram-core` создаёт три инстанса: context_index, action_index, result_index — по одному для каждого поля записи

#### engram-router

Иерархический Q-Learning роутер. Четыре уровня принятия решений.

**Режимы (верхний уровень иерархии):**

| Режим | Описание | Типичный сценарий |
|-------|----------|-------------------|
| `debug` | Поиск и исправление ошибок | Разбор стектрейса, чтение логов |
| `architecture` | Проектирование и выбор технологий | Выбор библиотек, ADR |
| `coding` | Написание и реализация кода | Имплементация фичи |
| `review` | Ревью и рефакторинг | Код ревью |
| `plan` | Планирование, оценка, анализ рисков | Оценка задачи, risk assessment |
| `routine` | Простые операции | Обновление зависимостей |

**Определение режима (комбинированный подход):**
- Агент может передать mode явно в параметрах MCP-инструмента
- Если mode не передан — автоопределение по ключевым словам запроса
- Q-Learning учится точности автоопределения и корректирует со временем

**Уровень 1: Стратегия поиска.** Какие типы памяти приоритизировать, порог similarity, количество результатов.

| Режим | Приоритет типов (по умолчанию) | Порог similarity | top_k |
|-------|-------------------------------|-----------------|-------|
| debug | bugfix, pattern, context | 0.8 | 3-5 |
| architecture | decision, context, pattern | 0.5 | 5-10 |
| coding | pattern, context, bugfix | 0.7 | 3-5 |
| review | pattern, decision, bugfix | 0.7 | 5-7 |
| plan | decision, bugfix, context, pattern | 0.5 | 7-10 |
| routine | context | 0.8 | 1-3 |

**Уровень 2: Выбор LLM.** Какую модель для judge и consolidate (дешёвая / дорогая / эвристики).

**Уровень 3: Контекстуализация результатов.** `raw` (как есть) или `summarize` (через LLM).

**Уровень 4: Проактивность.** `passive` (просто сохранить) или `proactive` (найти противоречия и предупредить).

**Реализация:**
- Четыре независимых Q-таблицы, каждая для своего уровня
- Режим (mode) — часть state во всех четырёх таблицах
- State: `{mode}:{дополнительный_контекст}`
- Каждая таблица имеет маленькое пространство actions (3-4 варианта)
- Не зависит от остальных крейтов, может использоваться отдельно

#### engram-storage

SQLite-обёртка, схема данных, CRUD-операции.

- Хранит записи памяти всех типов
- Хранит Q-Learning таблицы (4 уровня)
- FTS5-индекс для полнотекстового поиска (fallback при недоступности embedding API)

Схема:
```sql
CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    memory_type TEXT NOT NULL,       -- 'decision' | 'pattern' | 'bugfix' | 'context' | 'antipattern' | 'insight'
    context TEXT NOT NULL,
    action TEXT NOT NULL,
    result TEXT NOT NULL,
    score REAL DEFAULT 0.0,
    embedding_context BLOB,          -- вектор поля context
    embedding_action BLOB,           -- вектор поля action
    embedding_result BLOB,           -- вектор поля result
    indexed BOOLEAN DEFAULT FALSE,   -- true когда все эмбеддинги вычислены и запись во всех HNSW-индексах
    tags TEXT,                       -- JSON-массив
    project TEXT,
    parent_id TEXT,                  -- ссылка на предыдущее событие в причинно-следственной цепочке
    source_ids TEXT,                 -- JSON-массив id (для insight — записи из которых извлечён)
    insight_type TEXT,               -- 'cluster' | 'temporal' | 'causal' (только для insight)
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    used_count INTEGER DEFAULT 0,
    last_used_at TEXT,
    superseded_by TEXT,
    FOREIGN KEY (superseded_by) REFERENCES memories(id),
    FOREIGN KEY (parent_id) REFERENCES memories(id)
);

CREATE VIRTUAL TABLE memories_fts USING fts5(
    context, action, result,
    content='memories',
    content_rowid='rowid'
);

CREATE TABLE q_table (
    router_level TEXT NOT NULL,      -- 'search_strategy' | 'llm_selection' | 'contextualization' | 'proactivity'
    state TEXT NOT NULL,
    action TEXT NOT NULL,
    value REAL DEFAULT 0.0,
    update_count INTEGER DEFAULT 0,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (router_level, state, action)
);

CREATE TABLE consolidation_log (
    id TEXT PRIMARY KEY,
    action TEXT NOT NULL,
    memory_ids TEXT NOT NULL,
    reason TEXT,
    performed_at TEXT NOT NULL,
    performed_by TEXT NOT NULL        -- 'auto' | 'user'
);

CREATE TABLE feedback_tracking (
    memory_id TEXT NOT NULL,
    searched_at TEXT NOT NULL,
    judged BOOLEAN DEFAULT FALSE,
    judged_at TEXT,
    FOREIGN KEY (memory_id) REFERENCES memories(id)
);

CREATE TABLE recommendations (
    id TEXT PRIMARY KEY,
    key TEXT NOT NULL,               -- параметр конфига (dot-notation)
    current_value TEXT,
    suggested_value TEXT NOT NULL,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL,
    status TEXT DEFAULT 'pending'    -- 'pending' | 'accepted' | 'rejected'
);

CREATE TABLE metrics (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,              -- название метрики
    value REAL NOT NULL,
    period_start TEXT NOT NULL,      -- ISO 8601
    period_end TEXT NOT NULL,
    created_at TEXT NOT NULL
);
```

#### engram-llm-client

Общий клиент для взаимодействия с LLM API.

Два трейта:

```
trait EmbeddingProvider:
    async fn embed(text) -> Result<Vec<f32>>
    async fn embed_batch(texts) -> Result<Vec<Vec<f32>>>
    fn dimension() -> usize

trait TextGenerator:
    async fn generate(prompt, system) -> Result<String>
```

Реализации:
- **VoyageCodeProvider** — Voyage AI `voyage-code-3` (основной и единственный для эмбеддингов в v1)
- **OpenAIProvider** — OpenAI `gpt-4o-mini` (для TextGenerator: judge, consolidate)
- **LocalProvider** — ONNX Runtime (фоллбэк, на будущее)

> В v1 эмбеддинги только через Voyage Code. Трейт заложен для будущей замены. При смене провайдера потребуется пересчёт всех эмбеддингов.

**Rate limiting:**
- Retry с exponential backoff: 429/5xx — повтор через 1s, 2s, 4s, максимум 3 попытки
- Batch API для эмбеддингов
- Очередь с настраиваемым concurrency для LLM-вызовов

#### engram-embeddings

Обёртка над `engram-llm-client` для генерации эмбеддингов.

- При store: генерирует три эмбеддинга — отдельно для context, action, result
- Кэширование (не пересчитывать для неизменившегося текста)
- Подготовка текста для каждого поля отдельно

**HyDE (Hypothetical Document Embeddings):**
- При search: автоматическое определение нужен ли HyDE — по наличию вопросительных слов (как, почему, зачем, what, how, why), вопросительного знака, фраз типа «что делать», «как лучше»
- Если HyDE активен: LLM генерирует гипотетическую запись в формате context/action/result, эмбеддится она вместо оригинального запроса
- Агент может передать `hyde: true/false` явно и перекрыть эвристику
- При недоступности LLM для HyDE — fallback на обычный эмбеддинг запроса

#### engram-judge

Оценка результатов. Два режима: эвристический и LLM-as-Judge. При недоступности LLM — автоматический fallback на эвристики.

#### engram-consolidate

Консолидация базы. Preview (без LLM) → Analyze (с LLM) → Apply (без LLM). Автоматическая дедупликация при store и сборка мусора по расписанию.

#### engram-core

Точка входа. Склеивает модули. Два режима работы:

**Режим сервера** (для MCP):
- Unix socket сервер (tokio)
- Маршрутизация JSON-команд
- HNSW-индекс в памяти
- Фоновые задачи: garbage collection, доиндексация `indexed = false` (таймер раз в 5 минут)

**Режим CLI** (самодостаточный):
- Загрузка HNSW с диска, доиндексация если устарел
- Выполнение одной команды
- Сохранение обновлённого индекса при модифицирующих операциях
- Вывод в формате json / text / jsonl

Общее:
- Конфигурация (`engram.toml` + env vars)
- Bootstrap: `init`, `check`, `version`
- `status` — совмещает инфраструктуру и содержимое

---

## 5. TypeScript MCP-сервер (@engram/mcp-server)

### 5.1 Зависимости

- `@modelcontextprotocol/sdk` — официальный MCP SDK
- Стандартный `net` модуль для подключения к unix socket

### 5.2 Роль

Тонкий транслятор: MCP tool calls → JSON-команды через unix socket → ответ клиенту. Не содержит бизнес-логики.

### 5.3 Управление Rust-ядром

- Запуск `engram-core` как дочернего процесса при старте
- Мониторинг health через периодический ping
- Graceful shutdown при завершении
- Переподключение к существующему процессу если socket уже существует

### 5.4 Обнаружение клиентами

`engram init` выводит готовый сниппет для MCP-клиента.

**Claude Desktop** (`claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "engram": {
      "command": "npx",
      "args": ["@engram/mcp-server"],
      "env": {
        "ENGRAM_VOYAGE_API_KEY": "...",
        "ENGRAM_LLM_API_KEY": "..."
      }
    }
  }
}
```

**Claude Code:**
```
claude mcp add engram -- npx @engram/mcp-server
```

**Cursor** (`.cursor/mcp.json`):
```json
{
  "mcpServers": {
    "engram": {
      "command": "npx",
      "args": ["@engram/mcp-server"],
      "env": {
        "ENGRAM_VOYAGE_API_KEY": "...",
        "ENGRAM_LLM_API_KEY": "..."
      }
    }
  }
}
```

---

## 6. MCP-инструменты

### 6.1 memory_store

Сохранить новый опыт.

**Параметры:**
| Поле | Тип | Обязательное | Описание |
|------|-----|-------------|----------|
| memory_type | string | да | `decision` / `pattern` / `bugfix` / `context` / `antipattern` / `insight` |
| context | string | да | Описание ситуации |
| action | string | да | Что было сделано |
| result | string | да | Что получилось |
| score | number | нет | Начальная оценка (0.0-1.0), по умолчанию 0.5 |
| tags | string[] | нет | Теги |
| project | string | нет | Идентификатор проекта |
| mode | string | нет | Режим. Влияет на проактивность (уровень 4) |
| parent_id | string | нет | ID предыдущего события в причинно-следственной цепочке |

**Поведение:**
1. Сгенерировать три эмбеддинга: отдельно для context, action, result
2. Проверить дедупликацию (max cosine similarity по трём индексам > 0.95)
3. Если дубль — обновить существующую (score = max)
4. Если нет — вставить с `indexed = false`, затем HNSW insert во все три индекса, затем `indexed = true`
5. Обновить FTS-индекс
6. Роутер решает о проактивности (уровень 4)
7. Если `proactive` — поиск похожего опыта, вернуть предупреждения

**Degradation:** API недоступен → запись с `indexed = false`, FTS обновляется, `degraded: true`

**Возвращает:** id, `deduplicated: bool`, `degraded: bool`, `warnings[]` (при proactive)

### 6.2 memory_search

Найти релевантный опыт.

**Параметры:**
| Поле | Тип | Обязательное | Описание |
|------|-----|-------------|----------|
| query | string | да | Текстовый запрос |
| mode | string | нет | `debug` / `architecture` / `coding` / `review` / `plan` / `routine` |
| memory_type | string | нет | Явный фильтр (перекрывает роутер) |
| project | string | нет | Фильтр по проекту |
| top_k | number | нет | Количество результатов (перекрывает роутер) |
| min_score | number | нет | Минимальный score |
| hyde | bool | нет | Явное включение/выключение HyDE. null = автоопределение по эвристике |

**Поведение (полный search pipeline):**
1. Определить mode (явный или автоопределение)
2. Роутер выбирает стратегию поиска (уровень 1)
3. **HyDE:** если `hyde=true` или (hyde=null и эвристика детектировала вопрос) → LLM генерирует гипотетическую запись в формате context/action/result → эмбеддится она вместо оригинального запроса
4. **Мультивекторный поиск:** эмбеддинг запроса (или HyDE-гипотезы) сравнивается с тремя HNSW-индексами (context, action, result). Для каждой записи: `vector_score = max(sim_context, sim_action, sim_result)`
5. **Гибридный скоринг:** параллельно BM25 поиск по FTS5. Финальный: `final_score = alpha * vector_score + (1 - alpha) * bm25_score`
6. Фильтрация по memory_type, project, min_score
7. Роутер решает о контекстуализации (уровень 3)
8. Обновить used_count, last_used_at
9. Записать в feedback_tracking

**Degradation:** Embedding API недоступен → fallback на FTS5-only. HyDE LLM недоступен → поиск без HyDE обычным эмбеддингом.

**Feedback loop:** Ответ включает `pending_judgments` когда: 5+ неоценённых записей ИЛИ 10+ минут без judge. Не чаще раз в 3 вызова.

**Возвращает:** Список записей (id, memory_type, context, action, result, score, similarity, best_match_field, tags, created_at), mode, `degraded: bool`, `search_method: "hybrid" | "fts"`, `hyde_used: bool`, `pending_judgments[]`

### 6.3 memory_judge

Оценить результат.

**Параметры:**
| Поле | Тип | Обязательное | Описание |
|------|-----|-------------|----------|
| memory_id | string | да | ID записи |
| score | number | нет | Явная оценка (0.0-1.0) |
| outcome | string | нет | Описание результата (для LLM-as-Judge) |
| mode | string | нет | Режим, для обучения роутера |

**Поведение:**
1. Если передан score → manual, иначе роутер (уровень 2) выбирает heuristic/llm
2. Вычислить score
3. Обновить базу
4. Обновить все Q-таблицы роутера (reward = score)
5. Отметить в feedback_tracking

**Degradation:** LLM недоступен → fallback на heuristic

**Возвращает:** score, метод (manual/heuristic/llm), обоснование (если llm), `degraded: bool`

### 6.4 memory_status

Статистика содержимого памяти (не инфраструктура — для этого `engram status` CLI).

**Возвращает:** total_memories, by_type, by_project, avg_score, hnsw_index_size, pending_indexing, pending_judgments, last_consolidation, router_stats, providers, а также операционные метрики качества (search_hit_rate, judge_avg_score, judge_positive_rate, feedback_completion_rate, antipattern_prevention_rate, insight_usage_rate — см. раздел 15.1)

### 6.5 memory_config

Чтение и изменение конфигурации через MCP.

**Параметры:**
| Поле | Тип | Обязательное | Описание |
|------|-----|-------------|----------|
| action | string | да | `get` / `set` |
| key | string | нет | Ключ в dot-notation. Обязателен для `set` |
| value | any | нет | Новое значение. Обязательно для `set` |

**Возвращает:** Значение (get) или подтверждение (set) с `requires_restart: bool`

### 6.6 memory_export

Экспорт базы для бэкапа или переноса.

**Параметры:**
| Поле | Тип | Обязательное | Описание |
|------|-----|-------------|----------|
| format | string | нет | `json` (по умолчанию) / `sqlite` |
| project | string | нет | Только записи проекта |
| path | string | нет | Путь. По умолчанию `~/.engram/export/` |

**Возвращает:** Путь к файлу, количество записей

### 6.7 memory_import

Импорт из экспорта.

**Параметры:**
| Поле | Тип | Обязательное | Описание |
|------|-----|-------------|----------|
| path | string | да | Путь к файлу |
| strategy | string | нет | `merge` (по умолчанию) / `replace` |

**Возвращает:** Количество импортированных, количество дубликатов (при merge)

### 6.8 memory_consolidate_preview

Найти кандидатов на ревью. Без LLM.

**Параметры:** similarity_threshold (0.95), stale_days (90), min_score_threshold (0.1) — все опциональные.

**Возвращает:** duplicates[], garbage[], contradictions[], stale[], total_candidates

### 6.9 memory_consolidate

Анализ кандидатов через LLM. Структурированный отчёт с рекомендациями.

**Параметры:** candidate_ids (опционально, иначе все из preview)

**Возвращает:** recommendations[] (ids, type, description, suggested_action), summary

### 6.10 memory_consolidate_apply

Применить решения пользователя.

**Параметры:** actions[] (ids, action: merge/delete/archive/keep, keep_id для merge)

**Поведение:** merge/delete/archive/keep. Логируется в consolidation_log. Не зависит от API.

**Возвращает:** Количество действий, ошибки

### 6.11 memory_insights

Управление инсайтами.

**Параметры:**
| Поле | Тип | Обязательное | Описание |
|------|-----|-------------|----------|
| action | string | да | `list` — показать все инсайты, `generate` — запустить генерацию, `delete` — удалить инсайт |
| memory_type | string | нет | Для `generate` — сгенерировать инсайты только из кластеров этого типа |
| id | string | нет | Для `delete` — id инсайта |

**Поведение:**
- `list` — возвращает все инсайты с source_ids и insight_type
- `generate` — сканирует кластеры, генерирует инсайты через LLM для тех где порог превышен. Не дублирует уже существующие инсайты (проверка по source_ids)
- `delete` — удаляет инсайт по id

**Возвращает:** Для `list` — массив инсайтов. Для `generate` — количество новых инсайтов, их id. Для `delete` — подтверждение.

---

## 7. Типы памяти

### 7.1 decision — решения и подходы
Архитектурные решения, выбор технологий, обоснования.

### 7.2 pattern — паттерны кода
Стиль кода, конвенции проекта, типичные структуры.

### 7.3 bugfix — ошибки и фиксы
Баги, их причины, способы решения.

### 7.4 context — контекст проекта
Структура репозитория, зависимости, особенности инфраструктуры.

### 7.5 antipattern — что точно не стоит делать
Стратегические ошибки, неудачные подходы. Отдельный тип потому что антипаттерн должен перевешивать обычные записи при поиске — совпадение с антипаттерном важнее чем совпадение с pattern.

Дополнительные поля (в tags или отдельные поля context/action/result):
- **cost** — сколько стоила ошибка (время, деньги, описание последствий)
- **alternative** — что использовали вместо, на что заменили

### 7.6 insight — выведенное знание
Обобщение из кластера записей. Мета-память — память о памяти. Генерируется автоматически или по команде.

Дополнительные поля:
- **source_ids** — JSON-массив id записей, из которых извлечён инсайт
- **insight_type** — `cluster` (обобщение кластера), `temporal` (временной паттерн), `causal` (причинно-следственная цепочка)

Инсайты ищутся без привязки к project — обобщённое знание универсально. Имеют повышенный приоритет при proactive-поиске.

---

## 8. Автоматические процессы

### 8.1 Дедупликация при записи

При каждом `memory_store`: cosine similarity > 0.95 с существующими → обновить вместо создания. Score = max(existing, new).

### 8.2 Сборка мусора

По расписанию (по умолчанию раз в сутки): score < порог AND used_count = 0 AND возраст > порог → удалить. Логируется с performed_by = "auto".

### 8.3 Фоновая доиндексация

Таймер раз в 5 минут: найти `indexed = false` → попытка embed → HNSW insert → `indexed = true`. Если API недоступен — тихий пропуск.

### 8.4 Транзакционность записи (write-ahead)

1. Запись в SQLite с `indexed = false` (атомарная транзакция)
2. Вставка в HNSW-индекс
3. Обновление `indexed = true`

Сбой между 1 и 3 → запись сохранена, фоновая доиндексация подхватит. Source of truth — SQLite, HNSW — кэш.

### 8.5 Генерация инсайтов

По порогу: когда в кластере похожих записей одного типа (similarity > 0.75) накапливается N штук — система генерирует insight через LLM.

Настраиваемые пороги с дефолтами:

| Тип | Порог (записей в кластере) |
|-----|---------------------------|
| bugfix | 5 |
| pattern | 7 |
| decision | 5 |
| context | 10 |
| antipattern | 3 |

Также доступен ручной вызов: `engram insights generate` (CLI) или MCP-инструмент `memory_insights`.

### 8.6 Детекция причинно-следственных цепочек

Автоматическая (в engram-trainer, средний контур): записи в одном проекте, с высокой similarity, созданные в пределах короткого временного окна (часы) — вероятно связаны. Тренер строит цепочки постфактум через `parent_id`.

Явная: агент при `memory_store` передаёт `parent_id` когда знает что это продолжение предыдущего события.

После 3+ повторений схожей цепочки — генерируется insight с `insight_type: "causal"`.

---

## 9. Самообучение

### 9.1 Три контура обучения

**Быстрый контур (каждый вызов).** Q-Learning роутер обновляет таблицы после каждого judge. Мгновенная адаптация.

**Средний контур (ежедневно/еженедельно).** Запускается engram-trainer:
- Переобучение классификатора mode на накопленных данных
- Переоценка кластеров для генерации инсайтов
- Детекция временных паттернов (циклические, деградационные)
- Построение причинно-следственных цепочек
- Обновление модели ранжирования результатов поиска
- Мета-анализ эффективности системы

**Глубокий контур (по команде, `engram train --deep`):**
- Fine-tune эмбеддинговой модели на парах «похожие / не похожие записи» (из данных judge)
- Fine-tune маленькой генеративной модели (Phi, Qwen-0.5B) на стиле и решениях пользователя
- Результат заменяет API-вызовы для рутинных операций (judge, контекстуализация)

### 9.2 engram-trainer (Python)

Отдельный pip-пакет в монорепо. Опциональный — Engram полностью работает без него. При вызове `engram train` без установленного тренера система выводит `pip install engram-trainer`.

**Python — опция, не зависимость:**
- Базовый Engram (память, поиск, роутер, консолидация) работает без Python
- Обучение — надстройка для тех кому нужно
- Путь к Python настраивается: `[training] python_path = "python3"`

**Запуск:**
```
engram train [--format json]           # средний контур
engram train --deep [--format json]    # глубокий контур
```

Rust вызывает:
```
python3 -m engram_trainer --config '{"db_path": "...", "models_path": "...", "thresholds": {...}, ...}'
python3 -m engram_trainer --config '{...}' --deep
```

Конфигурация передаётся одним JSON-аргументом. Тренер не читает `engram.toml` — Rust формирует JSON из загруженного конфига.

**Принцип: тренер read-only к SQLite, не пишет ни в базу ни в конфиг.**

**Протокол вывода (stdout, JSON Lines):**
```
{"type": "progress", "step": "clustering", "pct": 30, "message": "Анализ кластеров bugfix..."}
{"type": "progress", "step": "insights", "pct": 60, "message": "Генерация 3 инсайтов..."}
{"type": "artifact", "path": "~/.engram/models/mode_classifier.onnx", "size_bytes": 245000}
{"type": "meta_auto", "key": "insights.thresholds.context", "old": 10, "new": 12, "reason": "context insights used 0 times in 30 days"}
{"type": "meta_recommend", "key": "router.modes.debug.default_judge", "suggested": "llm", "reason": "heuristic judge accuracy 45% in debug mode"}
{"type": "insight", "id": "ins_xxx", "insight_type": "cluster", "memory_type": "bugfix", "context": "...", "action": "...", "result": "...", "source_ids": ["id1", "id2", ...]}
{"type": "metric", "name": "judge_avg_score_weekly", "value": 0.73, "period_start": "2025-01-06T00:00:00Z", "period_end": "2025-01-12T23:59:59Z"}
{"type": "complete", "duration_sec": 45, "artifacts": 2, "insights": 3, "auto_adjustments": 1, "recommendations": 1, "metrics": 6}
```

Ошибки — в stderr. Exit code != 0 → Rust показывает ошибку.

**Rust обрабатывает каждую строку:**
- `progress` → при `--format text` выводит прогресс, при `--format json` пробрасывает
- `artifact` → логирует
- `meta_auto` → применяет автокорректировку, пишет в `engram.toml`
- `meta_recommend` → вставляет в таблицу `recommendations` в SQLite
- `insight` → вставляет новый insight в таблицу `memories`, генерирует эмбеддинг, обновляет HNSW
- `metric` → вставляет в таблицу `metrics`
- `complete` → выводит итог пользователю

**Артефакты (тренер сохраняет напрямую в `~/.engram/models/`):**
- `mode_classifier.onnx` — классификатор mode
- `ranking_model.onnx` — модель ранжирования результатов поиска
- `embeddings_adapter.onnx` — LoRA-адаптер для эмбеддингов (глубокий контур)
- `gen_adapter.gguf` — LoRA-адаптер для генеративной модели (глубокий контур)

Rust-ядро периодически проверяет `~/.engram/models/` и подгружает обновлённые модели.

### 9.3 Мета-обучение

Система наблюдает за собственной эффективностью и корректирует параметры.

**Автоматические корректировки (безопасные, < 20% за цикл):**
- Пороги генерации инсайтов (повысить для бесполезных типов, понизить для ценных)
- Epsilon роутера
- Множитель кросс-проектного поиска

**Через подтверждение (pending_recommendations в memory_status):**
- Изменение дефолтных приоритетов типов памяти для режимов
- Отключение типа из поиска для режима
- Переключение judge между heuristic и llm
- Рекомендация создать инсайт из конкретного кластера

### 9.4 Кросс-проектный трансфер

Трёхуровневый поиск:
1. Текущий проект — полная релевантность
2. Другие проекты — score × множитель (по умолчанию 0.7, настраиваемый)
3. Инсайты — всегда без привязки к проекту

Роутер уровня 1 может включать/выключать межпроектный поиск. Множитель корректируется мета-обучением.

---

## 10. Конфигурация

Файл `engram.toml`:
```toml
[server]
socket_path = "/tmp/engram.sock"

[storage]
db_path = "~/.engram/engram.db"
hnsw_path = "~/.engram/indexes/"    # директория для трёх HNSW-индексов (context, action, result)

[logging]
level = "info"
path = "~/.engram/logs/engram.log"
max_size_mb = 10
max_files = 5

[embedding]
provider = "voyage"
model = "voyage-code-3"

[llm]
provider = "openai"
model = "gpt-4o-mini"
max_concurrency = 3
retry_max_attempts = 3
retry_base_delay_ms = 1000

[router]
learning_rate = 0.1
epsilon = 0.1
default_mode = "routine"
mode_detection = "auto"

[router.modes.debug]
priority_types = ["bugfix", "pattern", "context"]
default_similarity_threshold = 0.8
default_top_k = 5
default_judge = "heuristic"
default_contextualization = "raw"
default_proactivity = "passive"

[router.modes.architecture]
priority_types = ["decision", "context", "pattern"]
default_similarity_threshold = 0.5
default_top_k = 7
default_judge = "llm"
default_contextualization = "summarize"
default_proactivity = "proactive"

[router.modes.coding]
priority_types = ["pattern", "context", "bugfix"]
default_similarity_threshold = 0.7
default_top_k = 5
default_judge = "heuristic"
default_contextualization = "raw"
default_proactivity = "passive"

[router.modes.review]
priority_types = ["pattern", "decision", "bugfix"]
default_similarity_threshold = 0.7
default_top_k = 5
default_judge = "llm"
default_contextualization = "raw"
default_proactivity = "passive"

[router.modes.plan]
priority_types = ["decision", "bugfix", "context", "pattern"]
default_similarity_threshold = 0.5
default_top_k = 10
default_judge = "llm"
default_contextualization = "summarize"
default_proactivity = "proactive"

[router.modes.routine]
priority_types = ["context"]
default_similarity_threshold = 0.8
default_top_k = 3
default_judge = "heuristic"
default_contextualization = "raw"
default_proactivity = "passive"

[feedback]
max_pending = 5
idle_minutes = 10
min_calls_between = 3

[consolidate]
auto_dedup_threshold = 0.95
garbage_min_score = 0.1
garbage_min_age_days = 90
garbage_schedule = "daily"

[search]
hybrid_alpha = 0.7                 # баланс dense vs sparse (1.0 = только vector, 0.0 = только BM25)
hyde_auto = true                   # автоопределение HyDE по эвристике

[hnsw]
m = 16
ef_construction = 200
ef_search = 50

[reindex]
interval_minutes = 5

[cross_project]
enabled = true
score_multiplier = 0.7             # множитель score для записей из других проектов

[insights]
cluster_similarity = 0.75          # порог similarity для детекции кластера
thresholds.bugfix = 5              # записей в кластере для генерации инсайта
thresholds.pattern = 7
thresholds.decision = 5
thresholds.context = 10
thresholds.antipattern = 3
causal_chain_min_repetitions = 3   # повторений цепочки для генерации causal insight
temporal_window_hours = 24         # окно для детекции связанных событий

[training]
python_path = "python3"            # путь к Python-интерпретатору
models_path = "~/.engram/models/"
meta_auto_adjust_max_pct = 20     # макс. процент автокорректировки за цикл
```

Все параметры перекрываются переменными окружения с префиксом `ENGRAM_`.

---

## 11. Структура проекта

```
engram/
├── Cargo.toml                     # workspace
├── engram.toml                    # конфигурация по умолчанию
├── crates/
│   ├── engram-hnsw/
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   └── lib.rs
│   │   ├── tests/
│   │   │   └── integration.rs
│   │   └── benches/
│   │       └── hnsw_bench.rs
│   ├── engram-router/
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── q_table.rs
│   │   │   ├── mode.rs
│   │   │   └── levels.rs
│   │   ├── tests/
│   │   │   └── integration.rs
│   │   └── benches/
│   │       └── router_bench.rs
│   ├── engram-storage/
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── schema.rs
│   │   │   └── migrations/
│   │   ├── tests/
│   │   │   └── integration.rs
│   │   └── benches/
│   │       └── storage_bench.rs
│   ├── engram-llm-client/
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── voyage.rs
│   │   │   ├── openai.rs
│   │   │   └── local.rs
│   │   └── tests/
│   │       └── integration.rs
│   ├── engram-embeddings/
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   └── lib.rs
│   │   └── tests/
│   │       └── integration.rs
│   ├── engram-judge/
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   └── lib.rs
│   │   └── tests/
│   │       └── integration.rs
│   ├── engram-consolidate/
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   └── lib.rs
│   │   └── tests/
│   │       └── integration.rs
│   └── engram-core/
│       ├── Cargo.toml
│       ├── src/
│       │   ├── main.rs
│       │   ├── cli.rs              # полный CLI (bootstrap + команды памяти)
│       │   ├── config.rs
│       │   ├── server.rs           # unix socket сервер (режим сервера)
│       │   ├── dispatch.rs         # маршрутизация команд
│       │   └── output.rs           # форматирование вывода (json, text, jsonl)
│       ├── tests/
│       │   └── integration.rs
│       └── benches/
│           └── core_bench.rs
├── tests/
│   └── e2e/
│       ├── full_cycle.rs
│       ├── router_learning.rs
│       └── mcp_integration.rs
├── mcp-server/
│   ├── package.json
│   ├── tsconfig.json
│   └── src/
│       ├── index.ts
│       ├── socket-client.ts
│       ├── lifecycle.ts
│       └── tools.ts
├── trainer/
│   ├── pyproject.toml             # engram-trainer pip-пакет
│   ├── src/
│   │   └── engram_trainer/
│   │       ├── __init__.py
│   │       ├── data.py            # чтение из SQLite
│   │       ├── insights.py        # генерация инсайтов, кластеризация
│   │       ├── temporal.py        # детекция временных паттернов
│   │       ├── causal.py          # причинно-следственные цепочки
│   │       ├── classifier.py      # обучение классификатора mode
│   │       ├── ranker.py          # модель ранжирования
│   │       ├── finetune.py        # fine-tune эмбеддингов и генеративной модели
│   │       └── meta.py            # мета-анализ, автокорректировки
│   └── tests/
│       ├── test_data.py
│       ├── test_insights.py
│       ├── test_temporal.py
│       ├── test_causal.py
│       ├── test_classifier.py
│       ├── test_ranker.py
│       ├── test_meta.py
│       ├── test_protocol.py       # интеграционный: stdout JSON Lines протокол
│       ├── test_artifacts.py      # валидация ONNX-артефактов
│       └── conftest.py            # фикстуры: mock SQLite, mock LLM
├── AGENT.md                       # инструкции для агентов (system prompt)
└── README.md
```

---

## 12. Порядок реализации

Каждая фаза: TDD (сначала тесты, потом код, потом бенчмарки). Крейт завершён когда тесты зелёные, бенчмарки в целевых показателях, clippy zero warnings.

### Фаза 1: Фундамент
1. `engram-hnsw` — тесты insert/search/delete/recall/serialize/deserialize → реализация → бенчмарки
2. `engram-router` — тесты Q-таблицы, mode detection, 4 уровня → реализация → бенчмарки
3. `engram-storage` — тесты CRUD, миграции, FTS, write-ahead → реализация → бенчмарки

### Фаза 2: Внешние интеграции
4. `engram-llm-client` — тесты с mock, обработка ошибок, retry → реализация
5. `engram-embeddings` — тесты кэширования, подготовки текста → реализация
6. `engram-judge` — тесты эвристик, парсинг LLM, fallback → реализация

### Фаза 3: Склейка
7. `engram-core` (режим сервера) — тесты socket протокола, маршрутизации, фоновых задач → реализация → бенчмарки
8. `engram-consolidate` — тесты preview/analyze/apply → реализация
9. `engram-core` (CLI) — тесты всех команд, форматирование вывода (json/text/jsonl), персистентный HNSW → реализация

### Фаза 4: MCP
10. `@engram/mcp-server` — тесты lifecycle, socket-client, tools → реализация
11. E2E тесты — полный цикл через MCP и CLI, degradation, feedback loop

### Фаза 5: Самообучение (средний контур)
12. `engram-trainer` — классификатор mode, ранжирование, генерация инсайтов
13. Детекция временных паттернов и причинно-следственных цепочек
14. Мета-анализ: автокорректировки + pending_recommendations
15. Интеграция: Rust-ядро подхватывает ONNX-модели из `~/.engram/models/`

### Фаза 6: Самообучение (глубокий контур)
16. Fine-tune эмбеддинговой модели на данных пользователя
17. Fine-tune генеративной модели (LoRA-адаптер)
18. Замена API-вызовов на локальные модели для рутинных операций

### Фаза 7: Полировка
19. Документация, примеры конфигурации
20. CI pipeline с авто-бенчмарками
21. Публикация crates.io + npm + PyPI

---

## 13. Разработка: TDD и бенчмарки

### 13.1 Строгий TDD

Red → Green → Refactor. Никакой продакшен-код без предварительного теста. Это требование.

**Unit-тесты** — `#[cfg(test)] mod tests` в каждом крейте. Каждая публичная функция, каждый edge case.

**Интеграционные тесты** — `tests/` в каждом крейте:
- storage + hnsw: запись → индексация → поиск
- storage FTS: запись → FTS поиск (без эмбеддингов)
- hnsw: serialize → deserialize → поиск даёт те же результаты
- core (сервер): полный цикл через unix socket
- core (CLI): каждая команда возвращает валидный JSON, text, jsonl
- core (CLI): персистентный HNSW — store → файл обновлён → search в новом процессе находит запись
- core: фоновая доиндексация `indexed = false` → таймер → `indexed = true`
- consolidate + storage + hnsw: preview → analyze (mock) → apply

**E2E-тесты** — `tests/e2e/`:
- Полный цикл store → search → judge → consolidate (через MCP)
- Полный цикл store → search → judge → consolidate (через CLI)
- CLI вывод парсится корректно из Node.js / bash
- Полный цикл store → search → judge → consolidate
- Роутер меняет стратегию после серии judge
- Дедупликация при store
- Проактивность в plan-режиме
- Graceful degradation: API отключен → store с degraded → search через FTS
- Feedback loop: search без judge → pending_judgments заполняется

### 13.2 Бенчмарки

`criterion` для статистически корректных измерений.

**engram-hnsw:**

| Бенчмарк | Целевой показатель |
|----------|--------------------|
| insert_1k | < 100ms |
| insert_10k | < 2s |
| search_in_1k (top-5) | < 1ms |
| search_in_10k (top-5) | < 5ms |
| search_in_100k (top-5) | < 50ms |
| recall@10 vs brute force | > 95% |
| build_from_sqlite (10k) | < 500ms |
| serialize_to_disk (10k) | < 200ms |
| deserialize_from_disk (10k) | < 100ms |

**engram-router:**

| Бенчмарк | Целевой показатель |
|----------|--------------------|
| choose_action | < 1us |
| update_and_choose (1000x) | < 1ms |
| convergence_rate | < 100 итераций |

**engram-storage:**

| Бенчмарк | Целевой показатель |
|----------|--------------------|
| insert_memory | < 1ms |
| batch_insert_1k | < 500ms |
| search_with_filter (10k) | < 5ms |
| fts_search (10k) | < 10ms |
| full_table_scan (10k) | < 100ms |

**engram-core (e2e через socket):**

| Бенчмарк | Целевой показатель |
|----------|--------------------|
| memory_search_e2e (без API) | < 10ms |
| memory_store_e2e (без API) | < 15ms |
| socket_throughput | > 1000 rps |

**Качество поиска (запускаются отдельно, на подготовленном датасете):**

| Бенчмарк | Что измеряет |
|----------|-------------|
| single_vector_recall | recall@5 при одном общем эмбеддинге (baseline) |
| multi_vector_recall | recall@5 при трёх отдельных эмбеддингах (context/action/result) |
| hybrid_recall | recall@5 при multi-vector + BM25 |
| hyde_recall | recall@5 при hybrid + HyDE |
| alpha_sweep | оптимальный alpha для hybrid_alpha при разных типах запросов |

Ожидание: каждый следующий уровень даёт прирост recall. Если нет — упрощаем pipeline.

**Правила:**
- Бенчмарки с API не входят в CI
- Остальные запускаются на каждый PR
- Деградация > 10% — PR не проходит
- Результаты сохраняются для трендов

### 13.3 Тестирование engram-trainer (Python, pytest)

**Unit-тесты (mock SQLite с тестовыми данными):**
- `data.py` — in-memory SQLite с тестовыми записями: корректно читает все 6 типов, фильтрует, группирует
- `insights.py` — 10 записей с высокой similarity → кластер детектируется. 10 разных записей → кластер не детектируется. Mock LLM для генерации текста
- `temporal.py` — записи с повторяющимся паттерном (bugfix каждые 2 недели после dependency update) → паттерн найден
- `causal.py` — записи с close timestamps и high similarity → связаны. Далёкие по времени → не связаны
- `classifier.py` — размеченные данные (query → mode), обучить, accuracy выше baseline
- `ranker.py` — проверка что модель ранжирования учитывает score, recency, used_count корректно
- `meta.py` — статистика с явной деградацией (insight type X never used) → корректная автокорректировка и рекомендация

**Интеграционный тест stdout-протокола:**
- Запуск тренера на тестовой базе
- Каждая строка stdout — валидный JSON
- Тип каждого сообщения из допустимого набора (progress/artifact/meta_auto/meta_recommend/insight/complete)
- Последнее сообщение — `complete`
- `pct` в progress монотонно растёт
- `meta_auto` содержит обязательные поля (key, old, new, reason)
- `insight` содержит обязательные поля (id, insight_type, source_ids, context, action, result)

**Валидация ONNX-артефактов:**
- После запуска тренера ONNX-файл существует
- Загружается через `onnxruntime`
- Принимает тензор правильной размерности
- Возвращает валидный output (правильный shape, значения в ожидаемом диапазоне)

### 13.4 CI pipeline

Rust:
```
cargo test --workspace
cargo bench --workspace
cargo clippy --workspace
cargo fmt --check
```

TypeScript:
```
npm test
npm run typecheck
npm run lint
```

Python (engram-trainer):
```
cd trainer
pytest
pytest --integration           # интеграционные тесты с stdout-протоколом и ONNX-валидацией
```

---

## 14. Инструкции для агентов

### 14.1 AGENT.md (system prompt)

Файл `AGENT.md` в корне проекта. Подключается как system instruction при работе агента с Engram. Язык — английский. Агент отвечает на языке пользователя.

Содержание:

**Overview:**
- Engram is your long-term memory. Always search before starting work, always store after completing.
- Respond in the user's language.

**Search discipline:**
- Call `memory_search` BEFORE starting any task
- Set `mode` matching your current activity (debug/coding/plan/architecture/review/routine)
- Be specific in queries — include error codes, library names, file paths, function names
- Pay attention to antipatterns in results — they are more important than regular patterns
- If `pending_judgments` is returned — evaluate those memories before continuing
- If search returns no results — proceed without context, but store your experience after

**Store discipline:**
- After completing a task — call `memory_store` with the result
- Do NOT store trivial things (formatting, variable renaming, simple typo fixes)
- Decisions with reasoning — always store as `decision`
- Found and fixed bug — always store as `bugfix`
- Failed approach — store as `antipattern` with cost and alternative
- Set `parent_id` when current action is a consequence of a previous one
- Include relevant tags for better discoverability

**Judge discipline:**
- After confirming the result works — call `memory_judge` with high score
- If the result turned out to be wrong — call `memory_judge` with low score
- Do NOT postpone judging — the router learns only from feedback
- Prefer explicit score when you are certain, use LLM mode for ambiguous cases

**Mode selection guide:**
- `debug`: investigating a bug, reading logs, analyzing stack traces
- `architecture`: designing structure, choosing technology, writing ADR
- `coding`: implementing a feature, writing a function, adding a module
- `review`: reviewing code, refactoring, improving existing code
- `plan`: planning a task, estimating effort, analyzing risks
- `routine`: minor operations, updating dependencies, small config changes

**Consolidation (periodic):**
- Run `memory_consolidate_preview` periodically to check memory health
- Review contradictions manually — only you know which approach is current
- Let the system auto-deduplicate and garbage-collect

### 14.2 Tool descriptions

Каждый MCP-инструмент имеет короткое description видимое агенту:

- `memory_store`: "Save experience to long-term memory. Call after completing any non-trivial task."
- `memory_search`: "Search past experience. Call BEFORE starting any task. Set mode matching your activity."
- `memory_judge`: "Rate the quality of a past experience. Call after confirming results work (or don't)."
- `memory_status`: "View memory statistics, pending judgments, and system health."
- `memory_config`: "Read or change Engram configuration."
- `memory_export`: "Export memory database for backup or transfer."
- `memory_import`: "Import memory from backup or preset."
- `memory_consolidate_preview`: "Scan memory for duplicates, contradictions, and stale entries. No LLM cost."
- `memory_consolidate`: "Analyze consolidation candidates with LLM. Returns structured recommendations."
- `memory_consolidate_apply`: "Apply user-approved consolidation actions (merge/delete/archive)."
- `memory_insights`: "List, generate, or delete insights (derived knowledge from memory clusters)."

---

## 15. Метрики качества

### 15.1 Операционные метрики (вычисляются на лету)

Доступны через `memory_status`, агрегируются из существующих таблиц:

| Метрика | Описание | Источник |
|---------|----------|----------|
| search_hit_rate | % search с хотя бы одним результатом similarity > порог | memories, feedback_tracking |
| judge_avg_score | Средний score по всем judge за период | memories |
| judge_positive_rate | % judge со score > 0.7 | memories |
| feedback_completion_rate | % search по которым потом был judge | feedback_tracking |
| antipattern_prevention_rate | Сколько раз antipattern найден при search до начала работы | memories (type=antipattern, used_count) |
| insight_usage_rate | % инсайтов попавших в результаты search хоть раз | memories (type=insight, used_count) |
| hyde_improvement | Разница в judge score между поисками с HyDE и без | feedback_tracking + memories |
| cross_project_hit_rate | Как часто межпроектные результаты получают высокий judge | memories, feedback_tracking |

### 15.2 Трендовые метрики (engram-trainer)

Считаются тренером, передаются через stdout-протокол (тип `metric`), Rust записывает в таблицу `metrics`.

| Метрика | Описание |
|---------|----------|
| judge_avg_score_weekly | Средний score по неделям — растёт ли качество |
| search_hit_rate_weekly | Hit rate по неделям — улучшается ли recall |
| feedback_loop_time_avg | Среднее время от search до judge — сокращается ли цикл |
| router_accuracy | Точность автоопределения mode (сравнение с явно переданными) |
| memory_growth_rate | Записей в неделю по типам |
| insight_generation_rate | Инсайтов сгенерировано за период |

### 15.3 Хранение

Трендовые метрики хранятся в таблице `metrics` (см. схему в разделе 4.2 engram-storage). Операционные метрики вычисляются на лету при каждом вызове `memory_status`.
