import os

import onnx

from engram_trainer.classifier import ModeClassifier
from engram_trainer.data import FeedbackEntry, Memory
from engram_trainer.ranker import FEATURE_COUNT, RankingModel


MEMORY_TYPES_CYCLE = [
    "decision", "pattern", "bugfix", "context", "antipattern",
]


def _make_memory(
    memory_id: str,
    memory_type: str = "decision",
    context: str = "Some context",
    action: str = "Some action",
    score: float = 0.5,
    used_count: int = 1,
    tags: str | None = "tag",
) -> Memory:
    return Memory(
        id=memory_id,
        memory_type=memory_type,
        context=context,
        action=action,
        result="Some result",
        score=score,
        embedding_context=None,
        embedding_action=None,
        embedding_result=None,
        indexed=False,
        tags=tags,
        project=None,
        parent_id=None,
        source_ids=None,
        insight_type=None,
        created_at="2025-01-10T12:00:00Z",
        updated_at="2025-01-10T12:00:00Z",
        used_count=used_count,
        last_used_at="2025-01-15T08:00:00Z",
        superseded_by=None,
    )


def _make_diverse_memories(count: int) -> list[Memory]:
    contexts = {
        "decision": "Choosing architecture for the system",
        "pattern": "Handling database connections properly",
        "bugfix": "Memory leak in connection pool detected",
        "context": "Setting up development environment locally",
        "antipattern": "Global mutable state causing race conditions",
    }
    memories = []
    for i in range(count):
        memory_type = MEMORY_TYPES_CYCLE[i % len(MEMORY_TYPES_CYCLE)]
        memories.append(
            _make_memory(
                memory_id=f"mem-{i:03d}",
                memory_type=memory_type,
                context=f"{contexts[memory_type]} variant {i}",
                action=f"Action for {memory_type} number {i}",
                score=0.3 + (i % 5) * 0.15,
                used_count=i % 7,
                tags="tag" if i % 2 == 0 else None,
            ),
        )
    return memories


def _make_training_data(
    count: int,
) -> tuple[list[Memory], list[FeedbackEntry]]:
    memories = _make_diverse_memories(count)
    feedback = [
        FeedbackEntry(
            memory_id=f"mem-{i:03d}",
            searched_at="2025-01-12T10:00:00Z",
            judged=(i % 3 == 0),
        )
        for i in range(count)
    ]
    return memories, feedback


class TestOnnxClassifierLoadable:
    def test_exported_model_passes_onnx_validation(self, tmp_path):
        models_path = str(tmp_path / "models")
        classifier = ModeClassifier(models_path)
        memories = _make_diverse_memories(25)

        training_result = classifier.train(memories)

        loaded_model = onnx.load(training_result.model_path)
        onnx.checker.check_model(loaded_model)


class TestOnnxRankerLoadable:
    def test_exported_model_passes_onnx_validation(self, tmp_path):
        models_path = str(tmp_path / "models")
        ranker = RankingModel(models_path)
        memories, feedback = _make_training_data(25)

        training_result = ranker.train(memories, feedback)

        loaded_model = onnx.load(training_result.model_path)
        onnx.checker.check_model(loaded_model)


class TestOnnxClassifierInputShape:
    def test_input_dimension_matches_tfidf_features(self, tmp_path):
        models_path = str(tmp_path / "models")
        classifier = ModeClassifier(models_path)
        memories = _make_diverse_memories(25)

        classifier.train(memories)

        onnx_path = os.path.join(models_path, "mode_classifier.onnx")
        loaded_model = onnx.load(onnx_path)
        graph_inputs = loaded_model.graph.input
        assert len(graph_inputs) >= 1

        first_input = graph_inputs[0]
        input_shape = first_input.type.tensor_type.shape
        feature_dimension = input_shape.dim[1].dim_value
        assert feature_dimension > 0
        assert feature_dimension <= 500


class TestOnnxRankerInputShape:
    def test_input_dimension_matches_feature_count(self, tmp_path):
        models_path = str(tmp_path / "models")
        ranker = RankingModel(models_path)
        memories, feedback = _make_training_data(25)

        ranker.train(memories, feedback)

        onnx_path = os.path.join(models_path, "ranking_model.onnx")
        loaded_model = onnx.load(onnx_path)
        graph_inputs = loaded_model.graph.input
        assert len(graph_inputs) >= 1

        first_input = graph_inputs[0]
        input_shape = first_input.type.tensor_type.shape
        feature_dimension = input_shape.dim[1].dim_value
        assert feature_dimension == FEATURE_COUNT
