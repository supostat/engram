import os

from engram_trainer.classifier import ClassifierResult, ModeClassifier
from engram_trainer.data import Memory


def _make_memory(
    memory_id: str,
    memory_type: str = "decision",
    context: str = "Some context",
    action: str = "Some action",
) -> Memory:
    return Memory(
        id=memory_id,
        memory_type=memory_type,
        context=context,
        action=action,
        result="Some result",
        score=0.5,
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


MEMORY_TYPES = ["decision", "pattern", "bugfix", "context", "antipattern"]

CONTEXTS_BY_TYPE = {
    "decision": "Choosing architecture for the system",
    "pattern": "Handling database connections properly",
    "bugfix": "Memory leak in connection pool detected",
    "context": "Setting up development environment locally",
    "antipattern": "Global mutable state causing race conditions",
}


def _make_diverse_memories(count: int) -> list[Memory]:
    memories = []
    for i in range(count):
        memory_type = MEMORY_TYPES[i % len(MEMORY_TYPES)]
        context = CONTEXTS_BY_TYPE[memory_type]
        memories.append(
            _make_memory(
                memory_id=f"mem-{i:03d}",
                memory_type=memory_type,
                context=f"{context} variant {i}",
                action=f"Action for {memory_type} number {i}",
            ),
        )
    return memories


class TestTrainSufficientData:
    def test_returns_classifier_result(self, tmp_path):
        models_path = str(tmp_path / "models")
        classifier = ModeClassifier(models_path)
        memories = _make_diverse_memories(25)

        training_result = classifier.train(memories)

        assert isinstance(training_result, ClassifierResult)
        assert training_result.accuracy >= 0.0
        assert training_result.accuracy <= 1.0
        assert len(training_result.classes) > 0
        assert os.path.isfile(training_result.model_path)


class TestTrainInsufficientData:
    def test_returns_none_below_threshold(self, tmp_path):
        models_path = str(tmp_path / "models")
        classifier = ModeClassifier(models_path)
        memories = _make_diverse_memories(5)

        training_result = classifier.train(memories)

        assert training_result is None


class TestOnnxFileCreated:
    def test_onnx_file_exists_after_training(self, tmp_path):
        models_path = str(tmp_path / "models")
        classifier = ModeClassifier(models_path)
        memories = _make_diverse_memories(25)

        classifier.train(memories)

        onnx_path = os.path.join(models_path, "mode_classifier.onnx")
        assert os.path.isfile(onnx_path)
        assert os.path.getsize(onnx_path) > 0


class TestClassifierHandlesSingleClass:
    def test_single_class_still_trains(self, tmp_path):
        models_path = str(tmp_path / "models")
        classifier = ModeClassifier(models_path)
        memories = [
            _make_memory(
                memory_id=f"mem-{i:03d}",
                memory_type="decision",
                context=f"Decision context variant {i}",
                action=f"Decision action variant {i}",
            )
            for i in range(15)
        ]

        training_result = classifier.train(memories)

        assert isinstance(training_result, ClassifierResult)
        assert training_result.classes == ["decision"]
