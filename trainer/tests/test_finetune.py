import os
import math

import pytest

try:
    import torch
    from transformers import GPT2Config
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from engram_trainer.data import Memory

pytestmark = pytest.mark.skipif(
    not HAS_TORCH, reason="torch not installed",
)

MEMORY_TYPES = ["decision", "pattern", "bugfix", "context", "antipattern"]


def _tiny_gpt2_config() -> "GPT2Config":
    return GPT2Config(
        n_layer=1,
        n_head=1,
        n_embd=32,
        vocab_size=50257,
    )


def _make_memory(
    memory_id: str,
    score: float,
    context: str = "Choosing database engine",
    action: str = "Selected SQLite for embedded storage",
    result: str = "Fast queries, zero config",
) -> Memory:
    return Memory(
        id=memory_id,
        memory_type=MEMORY_TYPES[hash(memory_id) % len(MEMORY_TYPES)],
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
        created_at="2025-01-10T12:00:00Z",
        updated_at="2025-01-10T12:00:00Z",
        used_count=0,
        last_used_at=None,
        superseded_by=None,
    )


def _make_high_quality_memories(count: int) -> list[Memory]:
    contexts = [
        "Choosing database for persistent storage",
        "Selecting authentication mechanism",
        "Designing message queue architecture",
        "Planning API versioning strategy",
        "Evaluating caching layer options",
    ]
    actions = [
        "Use SQLite with WAL mode",
        "Implement JWT with refresh rotation",
        "Adopt RabbitMQ with dead letter exchange",
        "Apply URL-path versioning scheme",
        "Deploy Redis with LRU eviction policy",
    ]
    results = [
        "Reduced latency by 40 percent",
        "Improved security posture significantly",
        "Achieved reliable async processing",
        "Maintained backward compatibility",
        "Cut response time in half",
    ]
    memories = []
    for i in range(count):
        variant = i % len(contexts)
        memories.append(
            _make_memory(
                memory_id=f"hq-{i:03d}",
                score=0.6 + (i % 4) * 0.1,
                context=f"{contexts[variant]} iteration {i}",
                action=f"{actions[variant]} approach {i}",
                result=f"{results[variant]} in scenario {i}",
            ),
        )
    return memories


class TestTrainSufficientData:
    def test_returns_finetune_result(self, tmp_path):
        from engram_trainer.finetune import FinetuneResult, TextLoraTrainer

        models_path = str(tmp_path / "models")
        trainer = TextLoraTrainer(
            models_path, model_config=_tiny_gpt2_config(),
        )
        memories = _make_high_quality_memories(25)

        training_result = trainer.train(memories)

        assert isinstance(training_result, FinetuneResult)
        assert os.path.isfile(training_result.model_path)
        assert training_result.final_loss > 0
        assert training_result.samples_used == 25


class TestTrainInsufficientData:
    def test_returns_none_below_threshold(self, tmp_path):
        from engram_trainer.finetune import TextLoraTrainer

        models_path = str(tmp_path / "models")
        trainer = TextLoraTrainer(
            models_path, model_config=_tiny_gpt2_config(),
        )
        memories = _make_high_quality_memories(5)

        training_result = trainer.train(memories)

        assert training_result is None


class TestTrainLowQualityFiltered:
    def test_returns_none_when_all_filtered(self, tmp_path):
        from engram_trainer.finetune import TextLoraTrainer

        models_path = str(tmp_path / "models")
        trainer = TextLoraTrainer(
            models_path, model_config=_tiny_gpt2_config(),
        )
        memories = [
            _make_memory(f"low-{i:03d}", score=0.3)
            for i in range(30)
        ]

        training_result = trainer.train(memories)

        assert training_result is None


class TestTrainingDataFormat:
    def test_build_training_data_produces_tokenized_dataset(self):
        from transformers import AutoTokenizer
        from engram_trainer.finetune import _build_training_data

        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        tokenizer.pad_token = tokenizer.eos_token
        memories = _make_high_quality_memories(5)

        dataset = _build_training_data(memories, tokenizer)

        assert len(dataset) == 5
        assert "input_ids" in dataset.column_names
        assert "labels" in dataset.column_names
        assert "attention_mask" in dataset.column_names

        first_sample = dataset[0]
        assert len(first_sample["input_ids"]) > 0
        assert len(first_sample["labels"]) == len(first_sample["input_ids"])


class TestOnnxExportCreatesFiles:
    def test_model_and_tokenizer_files_exist(self, tmp_path):
        from engram_trainer.finetune import FinetuneResult, TextLoraTrainer

        models_path = str(tmp_path / "models")
        trainer = TextLoraTrainer(
            models_path, model_config=_tiny_gpt2_config(),
        )
        memories = _make_high_quality_memories(25)

        training_result = trainer.train(memories)

        assert training_result is not None
        assert os.path.isfile(training_result.model_path)
        assert os.path.getsize(training_result.model_path) > 0
        assert os.path.isfile(training_result.tokenizer_path)
        assert os.path.getsize(training_result.tokenizer_path) > 0


class TestResultMetrics:
    def test_final_loss_is_valid(self, tmp_path):
        from engram_trainer.finetune import TextLoraTrainer

        models_path = str(tmp_path / "models")
        trainer = TextLoraTrainer(
            models_path, model_config=_tiny_gpt2_config(),
        )
        memories = _make_high_quality_memories(25)

        training_result = trainer.train(memories)

        assert training_result is not None
        assert training_result.final_loss > 0
        assert math.isfinite(training_result.final_loss)
        assert training_result.samples_used == 25
