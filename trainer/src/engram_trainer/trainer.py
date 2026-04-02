import os
import time

from engram_trainer.causal import CausalAnalyzer
from engram_trainer.classifier import ModeClassifier
from engram_trainer.data import DataReader
from engram_trainer.insights import ClusterAnalyzer
from engram_trainer.meta import MetaAnalyzer
from engram_trainer.protocol import (
    emit_artifact,
    emit_complete,
    emit_insight,
    emit_metric,
    emit_progress,
    emit_recommendation,
)
from engram_trainer.ranker import RankingModel
from engram_trainer.temporal import TemporalAnalyzer
from engram_trainer.types import Insight


def run_medium(database_path: str, models_path: str):
    start_time = time.monotonic()
    insights_count = 0

    emit_progress("loading", 0)
    memories, q_table, feedback = _load_data(database_path)

    insights_count += _run_clustering(memories)
    insights_count += _run_temporal(memories)
    insights_count += _run_causal(memories)
    _run_model_training(memories, feedback, models_path)
    _run_meta_analysis(memories, q_table)

    duration = time.monotonic() - start_time
    emit_complete(insights_count, round(duration, 2))


def run_deep(database_path: str, models_path: str):
    run_medium(database_path, models_path)


def _load_data(database_path: str):
    with DataReader(database_path) as reader:
        memories = reader.read_memories()
        q_table = reader.read_q_table()
        feedback = reader.read_feedback()
    return memories, q_table, feedback


def _run_clustering(memories: list) -> int:
    emit_progress("clustering", 20)
    try:
        analyzer = ClusterAnalyzer()
        clusters = analyzer.find_clusters(memories)
        insights = analyzer.generate_insights(clusters)
        _emit_insights(insights)
        return len(insights)
    except Exception as error:
        emit_progress(f"clustering_failed: {error}", 20)
        return 0


def _run_temporal(memories: list) -> int:
    emit_progress("temporal", 40)
    try:
        analyzer = TemporalAnalyzer()
        patterns = analyzer.find_patterns(memories)
        insights = analyzer.generate_insights(patterns)
        _emit_insights(insights)
        return len(insights)
    except Exception as error:
        emit_progress(f"temporal_failed: {error}", 40)
        return 0


def _run_causal(memories: list) -> int:
    emit_progress("causal", 60)
    try:
        analyzer = CausalAnalyzer()
        chains = analyzer.build_chains(memories)
        insights = analyzer.generate_insights(chains)
        _emit_insights(insights)
        return len(insights)
    except Exception as error:
        emit_progress(f"causal_failed: {error}", 60)
        return 0


def _run_model_training(
    memories: list, feedback: list, models_path: str,
):
    emit_progress("training", 70)

    try:
        classifier = ModeClassifier(models_path)
        classifier_result = classifier.train(memories)
        if classifier_result is not None:
            file_size = os.path.getsize(classifier_result.model_path)
            emit_artifact(classifier_result.model_path, file_size)
    except Exception as error:
        emit_progress(f"classifier_training_failed: {error}", 70)

    try:
        ranker = RankingModel(models_path)
        ranker_result = ranker.train(memories, feedback)
        if ranker_result is not None:
            file_size = os.path.getsize(ranker_result.model_path)
            emit_artifact(ranker_result.model_path, file_size)
    except Exception as error:
        emit_progress(f"ranker_training_failed: {error}", 70)


def _run_meta_analysis(memories: list, q_table: list):
    emit_progress("meta", 90)
    try:
        analyzer = MetaAnalyzer()
        meta_result = analyzer.analyze(memories, q_table)
        for metric in meta_result.metrics:
            emit_metric(metric.name, metric.value)
        for recommendation in meta_result.recommendations:
            emit_recommendation(
                recommendation.target_id,
                recommendation.action,
                recommendation.reasoning,
            )
    except Exception as error:
        emit_progress(f"meta_analysis_failed: {error}", 90)


def _emit_insights(insights: list[Insight]):
    for insight in insights:
        source_ids_string = (
            ",".join(insight.source_ids) if insight.source_ids else None
        )
        emit_insight(
            id=insight.id,
            context=insight.context,
            action=insight.action,
            result=insight.result,
            insight_type=insight.insight_type,
            tags=insight.tags,
            source_ids=source_ids_string,
        )
