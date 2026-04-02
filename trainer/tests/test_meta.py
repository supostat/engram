from engram_trainer.data import Memory, QTableEntry
from engram_trainer.meta import MetaAnalyzer, MetaResult


def _make_memory(
    memory_id: str,
    score: float = 0.5,
    memory_type: str = "decision",
) -> Memory:
    return Memory(
        id=memory_id,
        memory_type=memory_type,
        context="Some context",
        action="Some action",
        result="Some result",
        score=score,
        embedding_context=None,
        embedding_action=None,
        embedding_result=None,
        indexed=False,
        tags=None,
        project=None,
        parent_id=None,
        source_ids=None,
        insight_type=None,
        created_at="2025-01-01T00:00:00Z",
        updated_at="2025-01-01T00:00:00Z",
        used_count=0,
        last_used_at=None,
        superseded_by=None,
    )


def _make_q_table_entry(
    value: float = 1.0, update_count: int = 5,
) -> QTableEntry:
    return QTableEntry(
        router_level="semantic",
        state="high_score",
        action="boost",
        value=value,
        update_count=update_count,
    )


class TestAnalyzeReturnsMetrics:
    def test_non_empty_memories_produce_metrics(self):
        memories = [
            _make_memory("mem-001", score=0.8),
            _make_memory("mem-002", score=0.6),
            _make_memory("mem-003", score=0.4),
        ]
        q_table = [_make_q_table_entry()]
        analyzer = MetaAnalyzer()

        meta_result = analyzer.analyze(memories, q_table)

        assert isinstance(meta_result, MetaResult)
        metric_names = {m.name for m in meta_result.metrics}
        assert "memory_count" in metric_names
        assert "avg_score" in metric_names

        memory_count_metric = next(
            m for m in meta_result.metrics if m.name == "memory_count"
        )
        assert memory_count_metric.value == 3.0

        avg_score_metric = next(
            m for m in meta_result.metrics if m.name == "avg_score"
        )
        assert abs(avg_score_metric.value - 0.6) < 0.001


class TestAnalyzeEmpty:
    def test_no_memories_returns_empty_metrics(self):
        analyzer = MetaAnalyzer()

        meta_result = analyzer.analyze([], [])

        assert isinstance(meta_result, MetaResult)
        assert len(meta_result.metrics) == 0
        assert len(meta_result.recommendations) == 0


class TestRecommendationsLowScore:
    def test_low_score_memories_get_archive_recommendation(self):
        memories = [
            _make_memory("mem-low-1", score=0.1),
            _make_memory("mem-low-2", score=0.2),
            _make_memory("mem-ok", score=0.8),
        ]
        q_table = [_make_q_table_entry()]
        analyzer = MetaAnalyzer()

        meta_result = analyzer.analyze(memories, q_table)

        archive_recommendations = [
            r for r in meta_result.recommendations
            if r.action == "archive"
        ]
        archived_ids = {r.target_id for r in archive_recommendations}
        assert "mem-low-1" in archived_ids
        assert "mem-low-2" in archived_ids
        assert "mem-ok" not in archived_ids
