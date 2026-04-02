import sqlite3
from dataclasses import dataclass


@dataclass
class Memory:
    id: str
    memory_type: str
    context: str
    action: str
    result: str
    score: float
    embedding_context: bytes | None
    embedding_action: bytes | None
    embedding_result: bytes | None
    indexed: bool
    tags: str | None
    project: str | None
    parent_id: str | None
    source_ids: str | None
    insight_type: str | None
    created_at: str
    updated_at: str
    used_count: int
    last_used_at: str | None
    superseded_by: str | None


@dataclass
class QTableEntry:
    router_level: str
    state: str
    action: str
    value: float
    update_count: int


@dataclass
class FeedbackEntry:
    memory_id: str
    searched_at: str
    judged: bool


MEMORIES_COLUMNS = (
    "id", "memory_type", "context", "action", "result", "score",
    "embedding_context", "embedding_action", "embedding_result",
    "indexed", "tags", "project", "parent_id", "source_ids", "insight_type",
    "created_at", "updated_at", "used_count", "last_used_at", "superseded_by",
)


def _row_to_memory(row: tuple) -> Memory:
    return Memory(
        id=row[0],
        memory_type=row[1],
        context=row[2],
        action=row[3],
        result=row[4],
        score=row[5],
        embedding_context=row[6],
        embedding_action=row[7],
        embedding_result=row[8],
        indexed=bool(row[9]),
        tags=row[10],
        project=row[11],
        parent_id=row[12],
        source_ids=row[13],
        insight_type=row[14],
        created_at=row[15],
        updated_at=row[16],
        used_count=row[17],
        last_used_at=row[18],
        superseded_by=row[19],
    )


class DataReader:
    def __init__(self, database_path: str):
        uri = f"file:{database_path}?mode=ro"
        self.connection = sqlite3.connect(uri, uri=True)

    def read_memories(
        self, memory_types: list[str] | None = None,
    ) -> list[Memory]:
        columns = ", ".join(MEMORIES_COLUMNS)

        if memory_types is None:
            query = f"SELECT {columns} FROM memories"
            cursor = self.connection.execute(query)
        else:
            placeholders = ", ".join("?" for _ in memory_types)
            query = (
                f"SELECT {columns} FROM memories "
                f"WHERE memory_type IN ({placeholders})"
            )
            cursor = self.connection.execute(query, memory_types)

        return [_row_to_memory(row) for row in cursor.fetchall()]

    def read_q_table(self) -> list[QTableEntry]:
        cursor = self.connection.execute(
            "SELECT router_level, state, action, value, update_count "
            "FROM q_table",
        )
        return [
            QTableEntry(
                router_level=row[0],
                state=row[1],
                action=row[2],
                value=row[3],
                update_count=row[4],
            )
            for row in cursor.fetchall()
        ]

    def read_feedback(self) -> list[FeedbackEntry]:
        cursor = self.connection.execute(
            "SELECT memory_id, searched_at, judged FROM feedback_tracking",
        )
        return [
            FeedbackEntry(
                memory_id=row[0],
                searched_at=row[1],
                judged=bool(row[2]),
            )
            for row in cursor.fetchall()
        ]

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()
