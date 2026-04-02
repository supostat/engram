import os

from engram_trainer.data import FeedbackEntry, Memory
from engram_trainer.ranker import RankerResult, RankingModel


def _make_memory(
    memory_id: str,
    score: float = 0.5,
    used_count: int = 1,
    tags: str | None = "tag",
) -> Memory:
    return Memory(
        id=memory_id,
        memory_type="decision",
        context="Some context for ranking",
        action="Some action for ranking",
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


def _make_feedback(memory_id: str, judged: bool) -> FeedbackEntry:
    return FeedbackEntry(
        memory_id=memory_id,
        searched_at="2025-01-12T10:00:00Z",
        judged=judged,
    )


def _make_training_data(
    count: int,
) -> tuple[list[Memory], list[FeedbackEntry]]:
    memories = []
    feedback = []
    for i in range(count):
        memory_id = f"mem-{i:03d}"
        memories.append(
            _make_memory(
                memory_id=memory_id,
                score=0.3 + (i % 5) * 0.15,
                used_count=i % 7,
                tags="tag" if i % 2 == 0 else None,
            ),
        )
        feedback.append(_make_feedback(memory_id, judged=(i % 3 == 0)))
    return memories, feedback


class TestTrainSufficientData:
    def test_returns_ranker_result(self, tmp_path):
        models_path = str(tmp_path / "models")
        ranker = RankingModel(models_path)
        memories, feedback = _make_training_data(25)

        training_result = ranker.train(memories, feedback)

        assert isinstance(training_result, RankerResult)
        assert training_result.accuracy >= 0.0
        assert training_result.accuracy <= 1.0
        assert os.path.isfile(training_result.model_path)


class TestTrainInsufficientData:
    def test_returns_none_below_threshold(self, tmp_path):
        models_path = str(tmp_path / "models")
        ranker = RankingModel(models_path)
        memories, feedback = _make_training_data(5)

        training_result = ranker.train(memories, feedback)

        assert training_result is None


class TestOnnxFileCreated:
    def test_onnx_file_exists_after_training(self, tmp_path):
        models_path = str(tmp_path / "models")
        ranker = RankingModel(models_path)
        memories, feedback = _make_training_data(25)

        ranker.train(memories, feedback)

        onnx_path = os.path.join(models_path, "ranking_model.onnx")
        assert os.path.isfile(onnx_path)
        assert os.path.getsize(onnx_path) > 0
