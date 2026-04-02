import numpy as np

from engram_trainer.data import Memory
from engram_trainer.insights import Cluster, ClusterAnalyzer
from engram_trainer.types import Insight


def _make_memory(
    memory_id: str,
    embedding: bytes | None = None,
    context: str = "ctx",
    action: str = "act",
    result: str = "res",
    memory_type: str = "decision",
) -> Memory:
    return Memory(
        id=memory_id,
        memory_type=memory_type,
        context=context,
        action=action,
        result=result,
        score=0.5,
        embedding_context=embedding,
        embedding_action=embedding,
        embedding_result=embedding,
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


def _make_embedding(seed: int, dimension: int = 1024) -> bytes:
    vector = np.random.default_rng(seed).random(dimension, dtype=np.float32)
    return vector.tobytes()


def _make_similar_embedding(seed: int, noise_scale: float = 0.01) -> bytes:
    rng = np.random.default_rng(seed)
    base = rng.random(1024, dtype=np.float32)
    noise = np.random.default_rng(seed + 1000).random(1024, dtype=np.float32)
    similar = base + noise * noise_scale
    return similar.astype(np.float32).tobytes()


class TestFindClusters:
    def test_groups_similar_memories(self):
        emb_a1 = _make_embedding(42)
        emb_a2 = _make_similar_embedding(42, noise_scale=0.001)
        emb_b1 = _make_embedding(99)
        emb_b2 = _make_similar_embedding(99, noise_scale=0.001)

        memories = [
            _make_memory("m1", embedding=emb_a1, context="database setup"),
            _make_memory("m2", embedding=emb_a2, context="database config"),
            _make_memory("m3", embedding=emb_b1, context="auth flow"),
            _make_memory("m4", embedding=emb_b2, context="auth system"),
        ]

        analyzer = ClusterAnalyzer(similarity_threshold=0.85)
        clusters = analyzer.find_clusters(memories)

        assert len(clusters) == 2
        cluster_id_sets = [set(c.memory_ids) for c in clusters]
        assert {"m1", "m2"} in cluster_id_sets
        assert {"m3", "m4"} in cluster_id_sets

    def test_skips_memories_without_embeddings(self):
        emb = _make_embedding(42)
        memories = [
            _make_memory("m1", embedding=emb),
            _make_memory("m2", embedding=None),
            _make_memory("m3", embedding=emb),
        ]

        analyzer = ClusterAnalyzer(similarity_threshold=0.85)
        clusters = analyzer.find_clusters(memories)

        all_ids = {mid for c in clusters for mid in c.memory_ids}
        assert "m2" not in all_ids

    def test_single_memory_returns_no_clusters(self):
        memories = [_make_memory("m1", embedding=_make_embedding(42))]

        analyzer = ClusterAnalyzer(similarity_threshold=0.85)
        clusters = analyzer.find_clusters(memories)

        assert clusters == []

    def test_all_different_embeddings_no_clusters(self):
        memories = [
            _make_memory("m1", embedding=_make_embedding(1)),
            _make_memory("m2", embedding=_make_embedding(2)),
            _make_memory("m3", embedding=_make_embedding(3)),
            _make_memory("m4", embedding=_make_embedding(4)),
        ]

        analyzer = ClusterAnalyzer(similarity_threshold=0.99)
        clusters = analyzer.find_clusters(memories)

        assert clusters == []


class TestGenerateInsights:
    def test_generates_insights_from_clusters(self):
        clusters = [
            Cluster(
                memory_ids=["m1", "m2"],
                centroid_context="database",
                centroid_action="configure db",
                centroid_result="fast queries",
                size=2,
            ),
            Cluster(
                memory_ids=["m3", "m4"],
                centroid_context="auth",
                centroid_action="setup auth",
                centroid_result="secure access",
                size=2,
            ),
        ]

        analyzer = ClusterAnalyzer()
        insights = analyzer.generate_insights(clusters)

        assert len(insights) == 2
        for insight in insights:
            assert isinstance(insight, Insight)
            assert insight.insight_type == "cluster"
            assert len(insight.source_ids) >= 2
            assert insight.id  # uuid not empty

        source_id_sets = [set(i.source_ids) for i in insights]
        assert {"m1", "m2"} in source_id_sets
        assert {"m3", "m4"} in source_id_sets

    def test_generates_empty_insights_from_empty_clusters(self):
        analyzer = ClusterAnalyzer()
        insights = analyzer.generate_insights([])

        assert insights == []
