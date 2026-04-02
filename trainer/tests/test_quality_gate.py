import tempfile

import pytest

from engram_trainer.data import Memory
from engram_trainer.quality_gate import (
    QualityResult,
    validate_text_generation,
    _compute_rouge_l,
)


def _make_memory(
    memory_id: str,
    context: str = "test context",
    action: str = "test action",
    result: str = "test result",
    score: float = 0.8,
) -> Memory:
    return Memory(
        id=memory_id,
        memory_type="decision",
        context=context,
        action=action,
        result=result,
        score=score,
        embedding_context=None,
        embedding_action=None,
        embedding_result=None,
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


class TestComputeRougeLIdentical:
    def test_identical_strings_return_one(self):
        assert _compute_rouge_l("hello world", "hello world") == pytest.approx(1.0)

    def test_identical_multiword(self):
        text = "the quick brown fox jumps over the lazy dog"
        assert _compute_rouge_l(text, text) == pytest.approx(1.0)


class TestComputeRougeLNoOverlap:
    def test_completely_different_returns_zero(self):
        assert _compute_rouge_l("aaa bbb ccc", "ddd eee fff") == pytest.approx(0.0)

    def test_empty_reference_returns_zero(self):
        assert _compute_rouge_l("", "some text") == pytest.approx(0.0)

    def test_empty_hypothesis_returns_zero(self):
        assert _compute_rouge_l("some text", "") == pytest.approx(0.0)

    def test_both_empty_returns_zero(self):
        assert _compute_rouge_l("", "") == pytest.approx(0.0)


class TestComputeRougeLPartial:
    def test_partial_overlap(self):
        reference = "the cat sat on the mat"
        hypothesis = "the cat lay on a mat"
        score = _compute_rouge_l(reference, hypothesis)
        assert 0.0 < score < 1.0

    def test_subset_sequence(self):
        reference = "a b c d e"
        hypothesis = "a c e"
        score = _compute_rouge_l(reference, hypothesis)
        assert 0.0 < score < 1.0

    def test_reordered_hypothesis_scores_lower(self):
        reference = "a b c d e"
        ordered = "a b c"
        reordered = "c a e"
        score_ordered = _compute_rouge_l(reference, ordered)
        score_reordered = _compute_rouge_l(reference, reordered)
        assert score_ordered > score_reordered


class TestValidateNoModelFiles:
    def test_missing_model_directory_returns_not_passed(self):
        memories = [_make_memory(f"mem-{i}") for i in range(15)]
        result = validate_text_generation(memories, "/nonexistent/path")
        assert result.passed is False
        assert result.samples_tested == 0
        assert result.avg_score == pytest.approx(0.0)

    def test_empty_model_directory_returns_not_passed(self):
        with tempfile.TemporaryDirectory() as empty_directory:
            memories = [_make_memory(f"mem-{i}") for i in range(15)]
            result = validate_text_generation(memories, empty_directory)
            assert result.passed is False
            assert result.samples_tested == 0
            assert result.avg_score == pytest.approx(0.0)


class TestQualityResultDataclass:
    def test_fields_accessible(self):
        result = QualityResult(avg_score=0.42, samples_tested=10, passed=True)
        assert result.avg_score == pytest.approx(0.42)
        assert result.samples_tested == 10
        assert result.passed is True

    def test_not_passed_state(self):
        result = QualityResult(avg_score=0.05, samples_tested=5, passed=False)
        assert result.passed is False
        assert result.avg_score == pytest.approx(0.05)

    def test_zero_samples(self):
        result = QualityResult(avg_score=0.0, samples_tested=0, passed=False)
        assert result.samples_tested == 0
        assert result.avg_score == pytest.approx(0.0)
        assert result.passed is False
