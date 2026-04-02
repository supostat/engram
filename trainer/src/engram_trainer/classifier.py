from dataclasses import dataclass

import numpy as np
from scipy.sparse import vstack as sparse_vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from engram_trainer.data import Memory
from engram_trainer.onnx_export import export_model_to_onnx


MINIMUM_TRAINING_SAMPLES = 10
ONNX_FILENAME = "mode_classifier.onnx"
SYNTHETIC_LABEL = "__synthetic__"


@dataclass
class ClassifierResult:
    model_path: str
    accuracy: float
    classes: list[str]


class ModeClassifier:
    def __init__(self, models_path: str):
        self._models_path = models_path

    def train(self, memories: list[Memory]) -> ClassifierResult | None:
        if len(memories) < MINIMUM_TRAINING_SAMPLES:
            return None

        texts = _extract_texts(memories)
        labels = _extract_labels(memories)
        real_classes = sorted(set(labels))

        vectorizer = TfidfVectorizer(max_features=500)
        feature_matrix = vectorizer.fit_transform(texts)

        train_matrix, train_labels = _ensure_two_classes(
            feature_matrix, labels,
        )
        model = _fit_model(train_matrix, train_labels)
        accuracy = _compute_accuracy(model, feature_matrix, labels)

        onnx_path = _export_to_onnx(
            model, vectorizer, self._models_path,
        )

        return ClassifierResult(
            model_path=onnx_path,
            accuracy=accuracy,
            classes=real_classes,
        )


def _extract_texts(memories: list[Memory]) -> list[str]:
    return [f"{m.context} {m.action}" for m in memories]


def _extract_labels(memories: list[Memory]) -> list[str]:
    return [m.memory_type for m in memories]


def _ensure_two_classes(feature_matrix, labels: list[str]):
    unique = set(labels)
    if len(unique) >= 2:
        return feature_matrix, labels

    from scipy.sparse import csr_matrix
    zero_row = csr_matrix((1, feature_matrix.shape[1]))
    augmented_matrix = sparse_vstack([feature_matrix, zero_row])
    augmented_labels = labels + [SYNTHETIC_LABEL]
    return augmented_matrix, augmented_labels


def _fit_model(feature_matrix, labels: list[str]) -> LogisticRegression:
    model = LogisticRegression(max_iter=200, solver="lbfgs")
    return model.fit(feature_matrix, labels)


def _compute_accuracy(
    model: LogisticRegression,
    feature_matrix: np.ndarray,
    labels: list[str],
) -> float:
    predictions = model.predict(feature_matrix)
    correct = sum(
        predicted == actual
        for predicted, actual in zip(predictions, labels)
    )
    return correct / len(labels)


def _export_to_onnx(
    model: LogisticRegression,
    vectorizer: TfidfVectorizer,
    models_path: str,
) -> str:
    feature_count = len(vectorizer.get_feature_names_out())
    return export_model_to_onnx(
        model, feature_count, models_path, ONNX_FILENAME,
    )
