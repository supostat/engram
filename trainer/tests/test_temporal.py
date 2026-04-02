from engram_trainer.data import Memory
from engram_trainer.temporal import TemporalAnalyzer
from engram_trainer.types import Insight


def _make_memory(
    memory_id: str,
    memory_type: str = "bugfix",
    created_at: str = "2025-01-01T00:00:00Z",
    tags: str | None = None,
) -> Memory:
    return Memory(
        id=memory_id,
        memory_type=memory_type,
        context="ctx",
        action="act",
        result="res",
        score=0.5,
        embedding_context=None,
        embedding_action=None,
        embedding_result=None,
        indexed=False,
        tags=tags,
        project=None,
        parent_id=None,
        source_ids=None,
        insight_type=None,
        created_at=created_at,
        updated_at=created_at,
        used_count=0,
        last_used_at=None,
        superseded_by=None,
    )


class TestFindPatterns:
    def test_recurring_pattern_detected(self):
        memories = [
            _make_memory("m1", created_at="2025-01-01T00:00:00Z"),
            _make_memory("m2", created_at="2025-01-02T00:00:00Z"),
            _make_memory("m3", created_at="2025-01-03T00:00:00Z"),
            _make_memory("m4", created_at="2025-01-04T00:00:00Z"),
            _make_memory("m5", created_at="2025-01-05T00:00:00Z"),
        ]

        analyzer = TemporalAnalyzer(min_occurrences=3, window_hours=168)
        patterns = analyzer.find_patterns(memories)

        assert len(patterns) >= 1
        pattern = patterns[0]
        assert pattern.memory_type == "bugfix"
        assert pattern.occurrences >= 3
        assert len(pattern.memory_ids) >= 3

    def test_below_threshold_no_patterns(self):
        memories = [
            _make_memory("m1", created_at="2025-01-01T00:00:00Z"),
            _make_memory("m2", created_at="2025-01-02T00:00:00Z"),
        ]

        analyzer = TemporalAnalyzer(min_occurrences=3)
        patterns = analyzer.find_patterns(memories)

        assert patterns == []

    def test_mixed_types_no_cross_type_patterns(self):
        memories = [
            _make_memory("m1", memory_type="bugfix", created_at="2025-01-01T00:00:00Z"),
            _make_memory("m2", memory_type="decision", created_at="2025-01-02T00:00:00Z"),
            _make_memory("m3", memory_type="bugfix", created_at="2025-01-03T00:00:00Z"),
            _make_memory("m4", memory_type="decision", created_at="2025-01-04T00:00:00Z"),
            _make_memory("m5", memory_type="bugfix", created_at="2025-01-05T00:00:00Z"),
        ]

        analyzer = TemporalAnalyzer(min_occurrences=3)
        patterns = analyzer.find_patterns(memories)

        for pattern in patterns:
            assert pattern.memory_type in ("bugfix", "decision")
            ids_in_pattern = set(pattern.memory_ids)
            bugfix_ids = {"m1", "m3", "m5"}
            decision_ids = {"m2", "m4"}
            is_pure_bugfix = ids_in_pattern.issubset(bugfix_ids)
            is_pure_decision = ids_in_pattern.issubset(decision_ids)
            assert is_pure_bugfix or is_pure_decision

    def test_empty_memories_no_patterns(self):
        analyzer = TemporalAnalyzer()
        patterns = analyzer.find_patterns([])

        assert patterns == []


class TestGenerateInsights:
    def test_generates_insight_from_pattern(self):
        memories = [
            _make_memory("m1", created_at="2025-01-01T00:00:00Z"),
            _make_memory("m2", created_at="2025-01-02T00:00:00Z"),
            _make_memory("m3", created_at="2025-01-03T00:00:00Z"),
        ]

        analyzer = TemporalAnalyzer(min_occurrences=3)
        patterns = analyzer.find_patterns(memories)
        insights = analyzer.generate_insights(patterns)

        assert len(insights) >= 1
        insight = insights[0]
        assert isinstance(insight, Insight)
        assert insight.insight_type == "temporal"
        assert len(insight.source_ids) >= 3
        assert insight.id
