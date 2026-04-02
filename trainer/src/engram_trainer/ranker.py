from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from engram_trainer.data import FeedbackEntry, Memory
from engram_trainer.onnx_export import export_model_to_onnx
from engram_trainer.types import parse_timestamp


MINIMUM_TRAINING_SAMPLES = 10
ONNX_FILENAME = "ranking_model.onnx"
FEATURE_COUNT = 5


@dataclass
class RankerResult:
    model_path: str
    accuracy: float


class RankingModel:
    def __init__(self, models_path: str):
        self._models_path = models_path

    def train(
        self,
        memories: list[Memory],
        feedback: list[FeedbackEntry],
    ) -> RankerResult | None:
        if len(memories) < MINIMUM_TRAINING_SAMPLES:
            return None

        feedback_by_memory = _build_feedback_lookup(feedback)
        feature_matrix = _build_feature_matrix(memories)
        labels = _build_labels(memories, feedback_by_memory)

        model = _fit_model(feature_matrix, labels)
        accuracy = _compute_accuracy(model, feature_matrix, labels)

        onnx_path = _export_to_onnx(model, self._models_path)

        return RankerResult(model_path=onnx_path, accuracy=accuracy)


def _build_feedback_lookup(
    feedback: list[FeedbackEntry],
) -> dict[str, bool]:
    lookup: dict[str, bool] = {}
    for entry in feedback:
        if entry.judged:
            lookup[entry.memory_id] = True
    return lookup


def _build_feature_matrix(memories: list[Memory]) -> np.ndarray:
    features = [_extract_features(memory) for memory in memories]
    return np.array(features, dtype=np.float32)


def _extract_features(memory: Memory) -> list[float]:
    recency_days = _compute_recency_days(memory)
    text_length = float(len(memory.context) + len(memory.action))
    has_tags = 1.0 if memory.tags else 0.0

    return [
        memory.score,
        float(memory.used_count),
        recency_days,
        text_length,
        has_tags,
    ]


def _compute_recency_days(memory: Memory) -> float:
    if memory.last_used_at is None:
        created = parse_timestamp(memory.created_at)
        updated = parse_timestamp(memory.updated_at)
        delta = updated - created
        return max(delta.total_seconds() / 86400.0, 0.0)

    last_used = parse_timestamp(memory.last_used_at)
    created = parse_timestamp(memory.created_at)
    delta = last_used - created
    return max(delta.total_seconds() / 86400.0, 0.0)


def _build_labels(
    memories: list[Memory],
    feedback_by_memory: dict[str, bool],
) -> np.ndarray:
    labels = [
        1 if memory.id in feedback_by_memory else 0
        for memory in memories
    ]
    return np.array(labels, dtype=np.int32)


def _fit_model(
    feature_matrix: np.ndarray, labels: np.ndarray,
) -> GradientBoostingClassifier:
    model = GradientBoostingClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
    )
    return model.fit(feature_matrix, labels)


def _compute_accuracy(
    model: GradientBoostingClassifier,
    feature_matrix: np.ndarray,
    labels: np.ndarray,
) -> float:
    predictions = model.predict(feature_matrix)
    correct = int(np.sum(predictions == labels))
    return correct / len(labels)


def _export_to_onnx(
    model: GradientBoostingClassifier,
    models_path: str,
) -> str:
    return export_model_to_onnx(
        model, FEATURE_COUNT, models_path, ONNX_FILENAME,
    )
