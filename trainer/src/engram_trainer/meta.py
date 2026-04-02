from dataclasses import dataclass

from engram_trainer.data import Memory, QTableEntry


LOW_SCORE_THRESHOLD = 0.3


@dataclass
class MetricEntry:
    name: str
    value: float


@dataclass
class RecommendationEntry:
    target_id: str
    action: str
    reasoning: str


@dataclass
class MetaResult:
    metrics: list[MetricEntry]
    recommendations: list[RecommendationEntry]


class MetaAnalyzer:
    def __init__(self, config: dict | None = None):
        self._low_score_threshold = LOW_SCORE_THRESHOLD
        if config and "low_score_threshold" in config:
            self._low_score_threshold = config["low_score_threshold"]

    def analyze(
        self,
        memories: list[Memory],
        q_table: list[QTableEntry],
    ) -> MetaResult:
        if not memories:
            return MetaResult(metrics=[], recommendations=[])

        metrics = _compute_metrics(memories, q_table)
        recommendations = _generate_recommendations(
            memories, self._low_score_threshold,
        )

        return MetaResult(
            metrics=metrics,
            recommendations=recommendations,
        )


def _compute_metrics(
    memories: list[Memory],
    q_table: list[QTableEntry],
) -> list[MetricEntry]:
    metrics = [
        MetricEntry(name="memory_count", value=float(len(memories))),
        MetricEntry(name="avg_score", value=_average_score(memories)),
    ]

    type_distribution = _compute_type_distribution(memories)
    for memory_type, fraction in type_distribution.items():
        metrics.append(
            MetricEntry(
                name=f"type_fraction_{memory_type}",
                value=fraction,
            ),
        )

    if q_table:
        convergence = _compute_q_table_convergence(q_table)
        metrics.append(
            MetricEntry(name="q_table_convergence", value=convergence),
        )

    return metrics


def _average_score(memories: list[Memory]) -> float:
    total = sum(m.score for m in memories)
    return total / len(memories)


def _compute_type_distribution(
    memories: list[Memory],
) -> dict[str, float]:
    counts: dict[str, int] = {}
    for memory in memories:
        counts[memory.memory_type] = counts.get(memory.memory_type, 0) + 1

    total = len(memories)
    return {
        memory_type: count / total
        for memory_type, count in sorted(counts.items())
    }


def _compute_q_table_convergence(
    q_table: list[QTableEntry],
) -> float:
    total_updates = sum(entry.update_count for entry in q_table)
    if total_updates == 0:
        return 0.0
    average_updates = total_updates / len(q_table)
    return min(average_updates / 100.0, 1.0)


def _generate_recommendations(
    memories: list[Memory],
    low_score_threshold: float,
) -> list[RecommendationEntry]:
    recommendations = []

    for memory in memories:
        if memory.score < low_score_threshold:
            recommendations.append(
                RecommendationEntry(
                    target_id=memory.id,
                    action="archive",
                    reasoning=(
                        f"Score {memory.score:.2f} below threshold "
                        f"{low_score_threshold}"
                    ),
                ),
            )

    return recommendations
