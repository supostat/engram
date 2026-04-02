import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta

from engram_trainer.data import Memory
from engram_trainer.types import Insight, hours_between, parse_timestamp


@dataclass
class TemporalPattern:
    memory_ids: list[str]
    memory_type: str
    interval_hours: float
    occurrences: int


class TemporalAnalyzer:
    def __init__(
        self, min_occurrences: int = 3, window_hours: int = 168,
    ):
        self._min_occurrences = min_occurrences
        self._window_hours = window_hours

    def find_patterns(self, memories: list[Memory]) -> list[TemporalPattern]:
        grouped = _group_by_type(memories)
        patterns = []
        for memory_type, typed_memories in grouped.items():
            pattern = _detect_recurring_pattern(
                typed_memories, memory_type,
                self._min_occurrences, self._window_hours,
            )
            if pattern is not None:
                patterns.append(pattern)
        return patterns

    def generate_insights(
        self, patterns: list[TemporalPattern],
    ) -> list[Insight]:
        return [_pattern_to_insight(pattern) for pattern in patterns]


def _group_by_type(memories: list[Memory]) -> dict[str, list[Memory]]:
    grouped: dict[str, list[Memory]] = defaultdict(list)
    for memory in memories:
        grouped[memory.memory_type].append(memory)
    return grouped


def _detect_recurring_pattern(
    memories: list[Memory],
    memory_type: str,
    min_occurrences: int,
    window_hours: int,
) -> TemporalPattern | None:
    if len(memories) < min_occurrences:
        return None

    sorted_memories = sorted(
        memories, key=lambda m: parse_timestamp(m.created_at),
    )
    timestamps = [
        parse_timestamp(m.created_at) for m in sorted_memories
    ]

    window_memories = _filter_within_window(
        sorted_memories, timestamps, window_hours,
    )
    if len(window_memories) < min_occurrences:
        return None

    window_timestamps = [
        parse_timestamp(m.created_at) for m in window_memories
    ]
    average_interval = _compute_average_interval(window_timestamps)

    return TemporalPattern(
        memory_ids=[m.id for m in window_memories],
        memory_type=memory_type,
        interval_hours=average_interval,
        occurrences=len(window_memories),
    )


def _filter_within_window(
    sorted_memories: list[Memory],
    timestamps: list[datetime],
    window_hours: int,
) -> list[Memory]:
    if not timestamps:
        return []
    latest = timestamps[-1]
    window_start = latest - timedelta(hours=window_hours)
    return [
        memory for memory, timestamp in zip(sorted_memories, timestamps)
        if timestamp >= window_start
    ]


def _compute_average_interval(timestamps: list[datetime]) -> float:
    if len(timestamps) < 2:
        return 0.0
    intervals = [
        hours_between(timestamps[i], timestamps[i + 1])
        for i in range(len(timestamps) - 1)
    ]
    return sum(intervals) / len(intervals)


def _pattern_to_insight(pattern: TemporalPattern) -> Insight:
    return Insight(
        id=str(uuid.uuid4()),
        context=f"Recurring {pattern.memory_type} pattern detected",
        action=f"{pattern.occurrences} occurrences with ~{pattern.interval_hours:.1f}h interval",
        result=f"Temporal pattern in {pattern.memory_type} memories",
        insight_type="temporal",
        source_ids=list(pattern.memory_ids),
        tags=None,
    )
