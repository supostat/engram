import sqlite3
import tempfile
import os

import pytest

CREATE_MEMORIES = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    memory_type TEXT NOT NULL CHECK(memory_type IN ('decision','pattern','bugfix','context','antipattern','insight')),
    context TEXT NOT NULL,
    action TEXT NOT NULL,
    result TEXT NOT NULL,
    score REAL DEFAULT 0.0,
    embedding_context BLOB,
    embedding_action BLOB,
    embedding_result BLOB,
    indexed BOOLEAN DEFAULT FALSE,
    tags TEXT,
    project TEXT,
    parent_id TEXT,
    source_ids TEXT,
    insight_type TEXT CHECK(insight_type IS NULL OR insight_type IN ('cluster','temporal','causal')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    used_count INTEGER DEFAULT 0,
    last_used_at TEXT,
    superseded_by TEXT,
    FOREIGN KEY (superseded_by) REFERENCES memories(id),
    FOREIGN KEY (parent_id) REFERENCES memories(id)
)
"""

CREATE_Q_TABLE = """
CREATE TABLE IF NOT EXISTS q_table (
    router_level TEXT NOT NULL,
    state TEXT NOT NULL,
    action TEXT NOT NULL,
    value REAL DEFAULT 0.0,
    update_count INTEGER DEFAULT 0,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (router_level, state, action)
)
"""

CREATE_FEEDBACK_TRACKING = """
CREATE TABLE IF NOT EXISTS feedback_tracking (
    memory_id TEXT NOT NULL,
    searched_at TEXT NOT NULL,
    judged BOOLEAN DEFAULT FALSE,
    judged_at TEXT,
    FOREIGN KEY (memory_id) REFERENCES memories(id)
)
"""

SAMPLE_MEMORIES = [
    (
        "mem-001", "decision", "Choosing database for project",
        "Selected SQLite for embedded storage", "Fast queries, zero config",
        0.85, None, None, None, False, "database,sqlite", "engram",
        None, None, None,
        "2025-01-10T12:00:00Z", "2025-01-10T12:00:00Z", 3, "2025-01-15T08:00:00Z", None,
    ),
    (
        "mem-002", "pattern", "Handling concurrent writes",
        "Use WAL mode for SQLite", "Improved write throughput by 40%",
        0.92, None, None, None, True, "concurrency,sqlite", "engram",
        None, None, None,
        "2025-01-11T10:00:00Z", "2025-01-12T10:00:00Z", 7, "2025-02-01T09:00:00Z", None,
    ),
    (
        "mem-003", "bugfix", "Memory leak in connection pool",
        "Close connections on drop", "Eliminated OOM crashes",
        0.78, b"\x01\x02\x03", b"\x04\x05\x06", b"\x07\x08\x09", True,
        "bugfix,memory", "engram", None, None, None,
        "2025-01-15T14:00:00Z", "2025-01-15T14:00:00Z", 1, None, None,
    ),
    (
        "mem-004", "insight", "Cluster of database-related decisions",
        "Database patterns converge on embedded solutions",
        "Teams prefer embedded databases for CLI tools",
        0.60, None, None, None, False, "meta,database", None,
        None, "mem-001,mem-002", "cluster",
        "2025-02-01T09:00:00Z", "2025-02-01T09:00:00Z", 0, None, None,
    ),
    (
        "mem-005", "antipattern", "Global mutable state in handlers",
        "Avoid shared mutable state without synchronization",
        "Race conditions in production",
        0.45, None, None, None, False, None, "engram",
        None, None, None,
        "2025-02-10T16:00:00Z", "2025-02-10T16:00:00Z", 2, "2025-03-01T12:00:00Z", None,
    ),
]

SAMPLE_Q_TABLE_ENTRIES = [
    ("semantic", "high_score", "boost", 1.5, 10, "2025-01-20T12:00:00Z"),
    ("semantic", "low_score", "penalize", -0.5, 3, "2025-01-21T12:00:00Z"),
    ("keyword", "exact_match", "prefer", 2.0, 15, "2025-01-22T12:00:00Z"),
]

SAMPLE_FEEDBACK_ENTRIES = [
    ("mem-001", "2025-01-12T10:00:00Z", True, "2025-01-12T11:00:00Z"),
    ("mem-002", "2025-01-13T14:00:00Z", False, None),
    ("mem-003", "2025-01-16T09:00:00Z", True, "2025-01-16T10:00:00Z"),
]

MEMORY_TYPES_CYCLE = [
    "decision", "pattern", "bugfix", "context", "antipattern",
]

TRAINING_CONTEXTS = {
    "decision": [
        "Choosing database engine for the project",
        "Selecting authentication strategy",
        "Picking message queue for async processing",
        "Deciding on API versioning approach",
    ],
    "pattern": [
        "Handling database connections with pooling",
        "Retry logic for transient network failures",
        "Structured logging across all services",
        "Circuit breaker for external dependencies",
    ],
    "bugfix": [
        "Memory leak in connection pool detected",
        "Race condition in concurrent handler access",
        "Null pointer in deserialization path",
        "Deadlock in transaction manager",
    ],
    "context": [
        "Setting up local development environment",
        "Configuring CI pipeline for Rust project",
        "Docker compose setup for integration tests",
        "Performance baseline measurements",
    ],
    "antipattern": [
        "Global mutable state without synchronization",
        "Catching all exceptions silently",
        "Hard-coded configuration values in source",
        "Circular dependency between modules",
    ],
}

TRAINING_ACTIONS = {
    "decision": [
        "Selected SQLite for embedded storage",
        "Chose JWT with refresh token rotation",
        "Adopted RabbitMQ with dead letter exchange",
        "Implemented URL-path versioning",
    ],
    "pattern": [
        "Use connection pool with max lifetime",
        "Exponential backoff with jitter",
        "Log context propagation via middleware",
        "Hystrix-style circuit breaker wrapper",
    ],
    "bugfix": [
        "Close connections on drop in all paths",
        "Added mutex around shared state",
        "Added null check before deserialization",
        "Reordered lock acquisition to prevent deadlock",
    ],
    "context": [
        "Documented all env vars in .env.example",
        "Added cargo test to CI with caching",
        "Created docker-compose.test.yml",
        "Ran benchmarks with criterion",
    ],
    "antipattern": [
        "Refactored to pass state explicitly",
        "Added typed error handling with enums",
        "Moved config to environment variables",
        "Extracted shared types into core crate",
    ],
}

TRAINING_RESULTS = {
    "decision": "Improved throughput and reduced complexity",
    "pattern": "Increased reliability and reduced incidents",
    "bugfix": "Eliminated production crashes",
    "context": "Streamlined developer onboarding",
    "antipattern": "Reduced coupling and improved testability",
}


def _build_training_memories(count: int) -> list[tuple]:
    memories = []
    for i in range(count):
        memory_type = MEMORY_TYPES_CYCLE[i % len(MEMORY_TYPES_CYCLE)]
        variant = (i // len(MEMORY_TYPES_CYCLE)) % 4
        context = TRAINING_CONTEXTS[memory_type][variant]
        action = TRAINING_ACTIONS[memory_type][variant]
        result = TRAINING_RESULTS[memory_type]
        score = round(0.3 + (i % 7) * 0.1, 2)
        used_count = i % 5
        tags = f"{memory_type},training" if i % 2 == 0 else None
        day = 10 + (i % 20)
        last_used_day = day + 3 if used_count > 0 else None
        last_used_at = (
            f"2025-01-{last_used_day:02d}T08:00:00Z"
            if last_used_day and last_used_day <= 31
            else None
        )
        memories.append((
            f"train-{i:03d}", memory_type, context, action, result,
            score, None, None, None, False, tags, "engram",
            None, None, None,
            f"2025-01-{day:02d}T12:00:00Z",
            f"2025-01-{day:02d}T12:00:00Z",
            used_count, last_used_at, None,
        ))
    return memories


def _build_training_feedback(count: int) -> list[tuple]:
    feedback = []
    for i in range(count):
        memory_id = f"train-{i:03d}"
        judged = i % 3 == 0
        judged_at = "2025-01-20T10:00:00Z" if judged else None
        feedback.append((
            memory_id, "2025-01-18T10:00:00Z", judged, judged_at,
        ))
    return feedback


@pytest.fixture
def test_database_path():
    temp_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    temp_file.close()
    database_path = temp_file.name

    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(CREATE_MEMORIES)
    connection.execute(CREATE_Q_TABLE)
    connection.execute(CREATE_FEEDBACK_TRACKING)

    for memory in SAMPLE_MEMORIES:
        connection.execute(
            """INSERT INTO memories (
                id, memory_type, context, action, result, score,
                embedding_context, embedding_action, embedding_result,
                indexed, tags, project, parent_id, source_ids, insight_type,
                created_at, updated_at, used_count, last_used_at, superseded_by
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            memory,
        )

    for entry in SAMPLE_Q_TABLE_ENTRIES:
        connection.execute(
            """INSERT INTO q_table (router_level, state, action, value, update_count, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)""",
            entry,
        )

    for entry in SAMPLE_FEEDBACK_ENTRIES:
        connection.execute(
            """INSERT INTO feedback_tracking (memory_id, searched_at, judged, judged_at)
            VALUES (?, ?, ?, ?)""",
            entry,
        )

    connection.commit()
    connection.close()

    yield database_path

    os.unlink(database_path)


@pytest.fixture
def empty_database_path():
    temp_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    temp_file.close()
    database_path = temp_file.name

    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(CREATE_MEMORIES)
    connection.execute(CREATE_Q_TABLE)
    connection.execute(CREATE_FEEDBACK_TRACKING)
    connection.commit()
    connection.close()

    yield database_path

    os.unlink(database_path)


TRAINING_MEMORY_COUNT = 25


@pytest.fixture
def training_database_path():
    temp_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    temp_file.close()
    database_path = temp_file.name

    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(CREATE_MEMORIES)
    connection.execute(CREATE_Q_TABLE)
    connection.execute(CREATE_FEEDBACK_TRACKING)

    training_memories = _build_training_memories(TRAINING_MEMORY_COUNT)
    for memory in training_memories:
        connection.execute(
            """INSERT INTO memories (
                id, memory_type, context, action, result, score,
                embedding_context, embedding_action, embedding_result,
                indexed, tags, project, parent_id, source_ids, insight_type,
                created_at, updated_at, used_count, last_used_at, superseded_by
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            memory,
        )

    for entry in SAMPLE_Q_TABLE_ENTRIES:
        connection.execute(
            """INSERT INTO q_table (router_level, state, action, value, update_count, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)""",
            entry,
        )

    training_feedback = _build_training_feedback(TRAINING_MEMORY_COUNT)
    for entry in training_feedback:
        connection.execute(
            """INSERT INTO feedback_tracking (memory_id, searched_at, judged, judged_at)
            VALUES (?, ?, ?, ?)""",
            entry,
        )

    connection.commit()
    connection.close()

    yield database_path

    os.unlink(database_path)
