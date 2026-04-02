import json


def emit_progress(stage: str, percent: float):
    message = {"type": "progress", "stage": stage, "percent": percent}
    print(json.dumps(message), flush=True)


def emit_insight(
    id: str,
    context: str,
    action: str,
    result: str,
    insight_type: str,
    tags: str | None = None,
    source_ids: str | None = None,
):
    message = {
        "type": "insight",
        "id": id,
        "context": context,
        "action": action,
        "result": result,
        "insight_type": insight_type,
        "tags": tags,
        "source_ids": source_ids,
    }
    print(json.dumps(message), flush=True)


def emit_recommendation(
    target_id: str, action: str, reasoning: str,
):
    message = {
        "type": "recommendation",
        "target_id": target_id,
        "action": action,
        "reasoning": reasoning,
    }
    print(json.dumps(message), flush=True)


def emit_metric(name: str, value: float):
    message = {"type": "metric", "name": name, "value": value}
    print(json.dumps(message), flush=True)


def emit_artifact(path: str, size_bytes: int):
    message = {
        "type": "artifact",
        "path": path,
        "size_bytes": size_bytes,
    }
    print(json.dumps(message), flush=True)


def emit_complete(insights_generated: int, duration_secs: float):
    message = {
        "type": "complete",
        "insights_generated": insights_generated,
        "duration_secs": duration_secs,
    }
    print(json.dumps(message), flush=True)
