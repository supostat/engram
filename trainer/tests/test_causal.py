import numpy as np

from engram_trainer.causal import CausalAnalyzer
from engram_trainer.types import Insight
from engram_trainer.data import Memory


def _make_memory(
    memory_id: str,
    parent_id: str | None = None,
    embedding: bytes | None = None,
    created_at: str = "2025-01-01T00:00:00Z",
    memory_type: str = "decision",
) -> Memory:
    return Memory(
        id=memory_id,
        memory_type=memory_type,
        context="ctx",
        action="act",
        result="res",
        score=0.5,
        embedding_context=embedding,
        embedding_action=embedding,
        embedding_result=embedding,
        indexed=False,
        tags=None,
        project=None,
        parent_id=parent_id,
        source_ids=None,
        insight_type=None,
        created_at=created_at,
        updated_at=created_at,
        used_count=0,
        last_used_at=None,
        superseded_by=None,
    )


def _make_embedding(seed: int) -> bytes:
    vector = np.random.default_rng(seed).random(1024, dtype=np.float32)
    return vector.tobytes()


def _make_similar_embedding(seed: int, noise_scale: float = 0.001) -> bytes:
    rng = np.random.default_rng(seed)
    base = rng.random(1024, dtype=np.float32)
    noise = np.random.default_rng(seed + 1000).random(1024, dtype=np.float32)
    similar = base + noise * noise_scale
    return similar.astype(np.float32).tobytes()


class TestBuildChainsExplicit:
    def test_parent_id_chain(self):
        memories = [
            _make_memory("a", created_at="2025-01-01T00:00:00Z"),
            _make_memory("b", parent_id="a", created_at="2025-01-01T01:00:00Z"),
            _make_memory("c", parent_id="b", created_at="2025-01-01T02:00:00Z"),
        ]

        analyzer = CausalAnalyzer()
        chains = analyzer.build_chains(memories)

        assert len(chains) == 1
        assert chains[0].chain_length == 3
        assert chains[0].memory_ids == ["a", "b", "c"]

    def test_no_parent_id_no_embedding_no_chains(self):
        memories = [
            _make_memory("a", created_at="2025-01-01T00:00:00Z"),
            _make_memory("b", created_at="2025-01-01T01:00:00Z"),
        ]

        analyzer = CausalAnalyzer()
        chains = analyzer.build_chains(memories)

        assert chains == []


class TestBuildChainsImplicit:
    def test_implicit_chain_by_similarity_and_time(self):
        emb1 = _make_embedding(42)
        emb2 = _make_similar_embedding(42, noise_scale=0.001)
        emb3 = _make_similar_embedding(42, noise_scale=0.002)

        memories = [
            _make_memory(
                "x", embedding=emb1,
                created_at="2025-01-01T00:00:00Z",
            ),
            _make_memory(
                "y", embedding=emb2,
                created_at="2025-01-01T02:00:00Z",
            ),
            _make_memory(
                "z", embedding=emb3,
                created_at="2025-01-01T04:00:00Z",
            ),
        ]

        analyzer = CausalAnalyzer(
            time_window_hours=6, similarity_threshold=0.7,
        )
        chains = analyzer.build_chains(memories)

        assert len(chains) >= 1
        chain_ids = chains[0].memory_ids
        assert len(chain_ids) >= 2

    def test_empty_memories_no_chains(self):
        analyzer = CausalAnalyzer()
        chains = analyzer.build_chains([])

        assert chains == []


class TestGenerateInsights:
    def test_generates_insight_from_chain(self):
        memories = [
            _make_memory("a", created_at="2025-01-01T00:00:00Z"),
            _make_memory("b", parent_id="a", created_at="2025-01-01T01:00:00Z"),
            _make_memory("c", parent_id="b", created_at="2025-01-01T02:00:00Z"),
        ]

        analyzer = CausalAnalyzer()
        chains = analyzer.build_chains(memories)
        insights = analyzer.generate_insights(chains)

        assert len(insights) == 1
        insight = insights[0]
        assert isinstance(insight, Insight)
        assert insight.insight_type == "causal"
        assert len(insight.source_ids) == 3
        assert insight.id
