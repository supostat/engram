import json

from engram_trainer.protocol import (
    emit_progress,
    emit_insight,
    emit_recommendation,
    emit_metric,
    emit_artifact,
    emit_complete,
)


class TestEmitProgress:
    def test_emit_progress(self, capsys):
        emit_progress("training", 42.5)
        output = capsys.readouterr().out
        parsed = json.loads(output.strip())
        assert parsed == {"type": "progress", "stage": "training", "percent": 42.5}


class TestEmitInsight:
    def test_emit_insight(self, capsys):
        emit_insight(
            id="ins-001",
            context="Pattern cluster found",
            action="Group related decisions",
            result="3 decisions share common theme",
            insight_type="cluster",
            tags="meta,database",
            source_ids="mem-001,mem-002,mem-003",
        )
        output = capsys.readouterr().out
        parsed = json.loads(output.strip())
        assert parsed["type"] == "insight"
        assert parsed["id"] == "ins-001"
        assert parsed["context"] == "Pattern cluster found"
        assert parsed["action"] == "Group related decisions"
        assert parsed["result"] == "3 decisions share common theme"
        assert parsed["insight_type"] == "cluster"
        assert parsed["tags"] == "meta,database"
        assert parsed["source_ids"] == "mem-001,mem-002,mem-003"

    def test_emit_insight_optional_fields(self, capsys):
        emit_insight(
            id="ins-002",
            context="Temporal pattern",
            action="Detect trend",
            result="Bug fixes increasing",
            insight_type="temporal",
        )
        output = capsys.readouterr().out
        parsed = json.loads(output.strip())
        assert parsed["tags"] is None
        assert parsed["source_ids"] is None


class TestEmitRecommendation:
    def test_emit_recommendation(self, capsys):
        emit_recommendation(
            target_id="mem-001",
            action="increase_score",
            reasoning="Frequently accessed with positive feedback",
        )
        output = capsys.readouterr().out
        parsed = json.loads(output.strip())
        assert parsed == {
            "type": "recommendation",
            "target_id": "mem-001",
            "action": "increase_score",
            "reasoning": "Frequently accessed with positive feedback",
        }


class TestEmitMetric:
    def test_emit_metric(self, capsys):
        emit_metric("average_score", 0.73)
        output = capsys.readouterr().out
        parsed = json.loads(output.strip())
        assert parsed == {"type": "metric", "name": "average_score", "value": 0.73}


class TestEmitArtifact:
    def test_emit_artifact(self, capsys):
        emit_artifact("/models/router.onnx", 524288)
        output = capsys.readouterr().out
        parsed = json.loads(output.strip())
        assert parsed == {
            "type": "artifact",
            "path": "/models/router.onnx",
            "size_bytes": 524288,
        }


class TestEmitComplete:
    def test_emit_complete(self, capsys):
        emit_complete(12, 45.3)
        output = capsys.readouterr().out
        parsed = json.loads(output.strip())
        assert parsed == {
            "type": "complete",
            "insights_generated": 12,
            "duration_secs": 45.3,
        }


class TestOutputFormat:
    def test_all_emitters_produce_single_line(self, capsys):
        emit_progress("init", 0.0)
        emit_insight("i1", "c", "a", "r", "cluster")
        emit_recommendation("t1", "a", "reason")
        emit_metric("m", 1.0)
        emit_artifact("/p", 100)
        emit_complete(1, 0.1)

        output = capsys.readouterr().out
        lines = output.strip().split("\n")
        assert len(lines) == 6

        for line in lines:
            parsed = json.loads(line)
            assert "type" in parsed
