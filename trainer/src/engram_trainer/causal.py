import uuid
from dataclasses import dataclass

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from engram_trainer.data import Memory
from engram_trainer.types import (
    Insight,
    decode_embedding,
    hours_between,
    parse_timestamp,
)


@dataclass
class CausalChain:
    memory_ids: list[str]
    chain_length: int
    root_type: str


class CausalAnalyzer:
    def __init__(
        self,
        time_window_hours: int = 6,
        similarity_threshold: float = 0.7,
    ):
        self._time_window_hours = time_window_hours
        self._similarity_threshold = similarity_threshold

    def build_chains(self, memories: list[Memory]) -> list[CausalChain]:
        if not memories:
            return []

        explicit_chains = _build_explicit_chains(memories)
        chained_ids = {
            mid for chain in explicit_chains for mid in chain.memory_ids
        }

        unchained = [m for m in memories if m.id not in chained_ids]
        implicit_chains = _build_implicit_chains(
            unchained, self._time_window_hours, self._similarity_threshold,
        )

        return explicit_chains + implicit_chains

    def generate_insights(self, chains: list[CausalChain]) -> list[Insight]:
        return [_chain_to_insight(chain) for chain in chains]


def _build_explicit_chains(memories: list[Memory]) -> list[CausalChain]:
    memory_by_id = {m.id: m for m in memories}
    children_by_parent: dict[str, list[str]] = {}
    for memory in memories:
        if memory.parent_id is not None:
            children_by_parent.setdefault(
                memory.parent_id, [],
            ).append(memory.id)

    roots = [
        m for m in memories
        if m.parent_id is None and m.id in children_by_parent
    ]

    chains = []
    for root in roots:
        chain_ids = _walk_chain(root.id, children_by_parent)
        if len(chain_ids) >= 2:
            chains.append(CausalChain(
                memory_ids=chain_ids,
                chain_length=len(chain_ids),
                root_type=root.memory_type,
            ))
    return chains


def _walk_chain(
    root_id: str, children_by_parent: dict[str, list[str]],
) -> list[str]:
    chain = [root_id]
    current = root_id
    while current in children_by_parent:
        children = children_by_parent[current]
        next_child = children[0]
        chain.append(next_child)
        current = next_child
    return chain


def _build_implicit_chains(
    memories: list[Memory],
    time_window_hours: int,
    similarity_threshold: float,
) -> list[CausalChain]:
    embedded = [
        m for m in memories if m.embedding_context is not None
    ]
    if len(embedded) < 2:
        return []

    sorted_memories = sorted(
        embedded, key=lambda m: parse_timestamp(m.created_at),
    )

    vectors = np.array([
        decode_embedding(m.embedding_context) for m in sorted_memories
    ])
    similarity_matrix = cosine_similarity(vectors)

    visited: set[int] = set()
    chains: list[CausalChain] = []

    for i in range(len(sorted_memories)):
        if i in visited:
            continue
        chain_indices = _grow_implicit_chain(
            i, sorted_memories, similarity_matrix,
            time_window_hours, similarity_threshold, visited,
        )
        if len(chain_indices) >= 2:
            visited.update(chain_indices)
            chain_memories = [sorted_memories[idx] for idx in chain_indices]
            chains.append(CausalChain(
                memory_ids=[m.id for m in chain_memories],
                chain_length=len(chain_memories),
                root_type=chain_memories[0].memory_type,
            ))

    return chains


def _grow_implicit_chain(
    start_index: int,
    sorted_memories: list[Memory],
    similarity_matrix: np.ndarray,
    time_window_hours: int,
    similarity_threshold: float,
    visited: set[int],
) -> list[int]:
    chain = [start_index]
    current = start_index

    for candidate in range(current + 1, len(sorted_memories)):
        if candidate in visited:
            continue
        time_gap = hours_between(
            parse_timestamp(sorted_memories[current].created_at),
            parse_timestamp(sorted_memories[candidate].created_at),
        )
        if time_gap > time_window_hours:
            break
        if similarity_matrix[current, candidate] >= similarity_threshold:
            chain.append(candidate)
            current = candidate

    return chain


def _chain_to_insight(chain: CausalChain) -> Insight:
    return Insight(
        id=str(uuid.uuid4()),
        context=f"Causal chain of {chain.chain_length} memories",
        action=f"Root type: {chain.root_type}",
        result=f"Chain: {' -> '.join(chain.memory_ids)}",
        insight_type="causal",
        source_ids=list(chain.memory_ids),
        tags=None,
    )
