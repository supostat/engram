from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass
class Insight:
    id: str
    context: str
    action: str
    result: str
    insight_type: str
    source_ids: list[str]
    tags: str | None


def parse_timestamp(timestamp_string: str) -> datetime:
    cleaned = timestamp_string.replace("Z", "+00:00")
    return datetime.fromisoformat(cleaned)


def hours_between(earlier: datetime, later: datetime) -> float:
    delta = later - earlier
    return delta.total_seconds() / 3600.0


def decode_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)
