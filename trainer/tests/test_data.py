import sqlite3

import pytest

from engram_trainer.data import DataReader, Memory, QTableEntry, FeedbackEntry


class TestReadMemories:
    def test_read_memories_all(self, test_database_path):
        reader = DataReader(test_database_path)
        memories = reader.read_memories()
        reader.close()

        assert len(memories) == 5
        first = next(m for m in memories if m.id == "mem-001")
        assert first.memory_type == "decision"
        assert first.context == "Choosing database for project"
        assert first.action == "Selected SQLite for embedded storage"
        assert first.result == "Fast queries, zero config"
        assert first.score == pytest.approx(0.85)
        assert first.tags == "database,sqlite"
        assert first.project == "engram"
        assert first.used_count == 3
        assert first.last_used_at == "2025-01-15T08:00:00Z"
        assert first.superseded_by is None

    def test_read_memories_filtered(self, test_database_path):
        with DataReader(test_database_path) as reader:
            decisions = reader.read_memories(memory_types=["decision"])
            assert len(decisions) == 1
            assert decisions[0].memory_type == "decision"

            patterns_and_bugs = reader.read_memories(memory_types=["pattern", "bugfix"])
            assert len(patterns_and_bugs) == 2
            types = {m.memory_type for m in patterns_and_bugs}
            assert types == {"pattern", "bugfix"}

    def test_read_memories_empty_database(self, empty_database_path):
        with DataReader(empty_database_path) as reader:
            memories = reader.read_memories()
            assert memories == []

    def test_read_memories_with_embeddings(self, test_database_path):
        with DataReader(test_database_path) as reader:
            memories = reader.read_memories()
            with_embeddings = next(m for m in memories if m.id == "mem-003")
            assert with_embeddings.embedding_context == b"\x01\x02\x03"
            assert with_embeddings.embedding_action == b"\x04\x05\x06"
            assert with_embeddings.embedding_result == b"\x07\x08\x09"

            without_embeddings = next(m for m in memories if m.id == "mem-001")
            assert without_embeddings.embedding_context is None
            assert without_embeddings.embedding_action is None
            assert without_embeddings.embedding_result is None

    def test_read_memories_insight_fields(self, test_database_path):
        with DataReader(test_database_path) as reader:
            memories = reader.read_memories(memory_types=["insight"])
            assert len(memories) == 1
            insight = memories[0]
            assert insight.insight_type == "cluster"
            assert insight.source_ids == "mem-001,mem-002"


class TestReadQTable:
    def test_read_q_table(self, test_database_path):
        with DataReader(test_database_path) as reader:
            entries = reader.read_q_table()

        assert len(entries) == 3
        boost = next(e for e in entries if e.action == "boost")
        assert boost.router_level == "semantic"
        assert boost.state == "high_score"
        assert boost.value == pytest.approx(1.5)
        assert boost.update_count == 10

    def test_read_q_table_empty(self, empty_database_path):
        with DataReader(empty_database_path) as reader:
            entries = reader.read_q_table()
            assert entries == []


class TestReadFeedback:
    def test_read_feedback(self, test_database_path):
        with DataReader(test_database_path) as reader:
            entries = reader.read_feedback()

        assert len(entries) == 3
        judged = next(e for e in entries if e.memory_id == "mem-001")
        assert judged.judged is True
        assert judged.searched_at == "2025-01-12T10:00:00Z"

        not_judged = next(e for e in entries if e.memory_id == "mem-002")
        assert not_judged.judged is False

    def test_read_feedback_empty(self, empty_database_path):
        with DataReader(empty_database_path) as reader:
            entries = reader.read_feedback()
            assert entries == []


class TestContextManager:
    def test_context_manager_opens_and_closes(self, test_database_path):
        with DataReader(test_database_path) as reader:
            memories = reader.read_memories()
            assert len(memories) == 5

    def test_context_manager_closes_on_exception(self, test_database_path):
        with pytest.raises(RuntimeError):
            with DataReader(test_database_path) as reader:
                raise RuntimeError("forced error")


class TestReadOnlyMode:
    def test_insert_raises_error(self, test_database_path):
        with DataReader(test_database_path) as reader:
            with pytest.raises(sqlite3.OperationalError):
                reader.connection.execute(
                    "INSERT INTO memories (id, memory_type, context, action, result, score, created_at, updated_at) "
                    "VALUES ('x', 'decision', 'c', 'a', 'r', 0.0, '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
                )
