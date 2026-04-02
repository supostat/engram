import uuid
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

from engram_trainer.data import Memory
from engram_trainer.types import Insight, decode_embedding


@dataclass
class Cluster:
    memory_ids: list[str]
    centroid_context: str
    centroid_action: str
    centroid_result: str
    size: int


def _average_embedding(embeddings: list[np.ndarray]) -> np.ndarray:
    return np.mean(embeddings, axis=0)


class ClusterAnalyzer:
    def __init__(self, similarity_threshold: float = 0.85):
        self._distance_threshold = 1.0 - similarity_threshold

    def find_clusters(self, memories: list[Memory]) -> list[Cluster]:
        embedded_memories = _filter_memories_with_embeddings(memories)
        if len(embedded_memories) < 2:
            return []

        embedding_matrix = _build_embedding_matrix(embedded_memories)
        labels = _run_agglomerative_clustering(
            embedding_matrix, self._distance_threshold,
        )
        return _build_clusters_from_labels(embedded_memories, labels)

    def generate_insights(self, clusters: list[Cluster]) -> list[Insight]:
        return [_cluster_to_insight(cluster) for cluster in clusters]


def _filter_memories_with_embeddings(
    memories: list[Memory],
) -> list[Memory]:
    return [
        memory for memory in memories
        if memory.embedding_context is not None
    ]


def _build_embedding_matrix(memories: list[Memory]) -> np.ndarray:
    vectors = [
        decode_embedding(memory.embedding_context)
        for memory in memories
    ]
    return np.array(vectors)


def _run_agglomerative_clustering(
    embedding_matrix: np.ndarray, distance_threshold: float,
) -> np.ndarray:
    similarity_matrix = cosine_similarity(embedding_matrix)
    distance_matrix = 1.0 - similarity_matrix
    np.fill_diagonal(distance_matrix, 0.0)
    distance_matrix = np.maximum(distance_matrix, 0.0)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="precomputed",
        linkage="average",
    )
    return clustering.fit_predict(distance_matrix)


def _build_clusters_from_labels(
    memories: list[Memory], labels: np.ndarray,
) -> list[Cluster]:
    label_to_memories: dict[int, list[Memory]] = {}
    for memory, label in zip(memories, labels):
        label_to_memories.setdefault(int(label), []).append(memory)

    clusters = []
    for group in label_to_memories.values():
        if len(group) < 2:
            continue
        clusters.append(_build_single_cluster(group))

    return clusters


def _build_single_cluster(group: list[Memory]) -> Cluster:
    return Cluster(
        memory_ids=[memory.id for memory in group],
        centroid_context=group[0].context,
        centroid_action=group[0].action,
        centroid_result=group[0].result,
        size=len(group),
    )


def _cluster_to_insight(cluster: Cluster) -> Insight:
    return Insight(
        id=str(uuid.uuid4()),
        context=cluster.centroid_context,
        action=cluster.centroid_action,
        result=cluster.centroid_result,
        insight_type="cluster",
        source_ids=list(cluster.memory_ids),
        tags=None,
    )
