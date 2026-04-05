import os
from dataclasses import dataclass

from engram_trainer.data import Memory


QUALITY_THRESHOLD = 0.5
MINIMUM_SAMPLES = 20
ONNX_FILENAME = "text_generator.onnx"
TOKENIZER_FILENAME = "tokenizer.json"

LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["c_attn", "c_proj"]

TRAINING_EPOCHS = 2
TRAINING_BATCH_SIZE = 4
TRAINING_LEARNING_RATE = 2e-4
MAX_SEQUENCE_LENGTH = 256


@dataclass
class FinetuneResult:
    model_path: str
    tokenizer_path: str
    final_loss: float
    samples_used: int


class TextLoraTrainer:
    def __init__(
        self,
        models_path: str,
        base_model: str = "distilgpt2",
        model_config=None,
    ):
        self._models_path = models_path
        self._base_model = base_model
        self._model_config = model_config

    def train(
        self, memories: list[Memory], min_samples: int = MINIMUM_SAMPLES,
    ) -> FinetuneResult | None:
        quality_memories = _filter_by_quality(memories)
        if len(quality_memories) < min_samples:
            return None

        model, tokenizer = _load_model_and_tokenizer(
            self._base_model, self._model_config,
        )
        lora_model = _apply_lora(model)
        dataset = _build_training_data(quality_memories, tokenizer)
        final_loss = _run_training(lora_model, tokenizer, dataset)
        merged_model = _merge_lora_weights(lora_model)
        model_path, tokenizer_path = _export_model(
            merged_model, tokenizer, self._models_path,
        )

        return FinetuneResult(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            final_loss=final_loss,
            samples_used=len(quality_memories),
        )


def _filter_by_quality(memories: list[Memory]) -> list[Memory]:
    return [m for m in memories if m.score > QUALITY_THRESHOLD]


def _load_model_and_tokenizer(base_model: str, model_config=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    if model_config is not None:
        model = AutoModelForCausalLM.from_config(model_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(base_model)

    return model, tokenizer


def _apply_lora(model):
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_config)


def _build_training_data(memories: list[Memory], tokenizer, max_length: int = MAX_SEQUENCE_LENGTH):
    from datasets import Dataset

    texts = _format_training_texts(memories)
    tokenized = _tokenize_texts(texts, tokenizer, max_length)
    return Dataset.from_dict(tokenized)


def _format_training_texts(memories: list[Memory]) -> list[str]:
    return [
        f"Context: {m.context}\nAction: {m.action}\n{m.result}"
        for m in memories
    ]


def _tokenize_texts(texts: list[str], tokenizer, max_length: int) -> dict:
    encoded = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    return {
        "input_ids": encoded["input_ids"].tolist(),
        "attention_mask": encoded["attention_mask"].tolist(),
        "labels": encoded["input_ids"].tolist(),
    }


def _run_training(model, tokenizer, dataset) -> float:
    import shutil
    import torch
    from transformers import Trainer, TrainingArguments
    import tempfile

    training_directory = tempfile.mkdtemp(prefix="engram_lora_")

    try:
        training_arguments = TrainingArguments(
            output_dir=training_directory,
            num_train_epochs=TRAINING_EPOCHS,
            per_device_train_batch_size=TRAINING_BATCH_SIZE,
            learning_rate=TRAINING_LEARNING_RATE,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            disable_tqdm=True,
            use_cpu=not torch.cuda.is_available(),
        )

        trainer = Trainer(
            model=model,
            args=training_arguments,
            train_dataset=dataset,
            processing_class=tokenizer,
        )

        train_output = trainer.train()
        return train_output.training_loss
    finally:
        shutil.rmtree(training_directory, ignore_errors=True)


def _merge_lora_weights(lora_model):
    return lora_model.merge_and_unload()


def _export_model(
    model, tokenizer, output_path: str,
) -> tuple[str, str]:
    import torch

    os.makedirs(output_path, exist_ok=True)
    onnx_path = os.path.join(output_path, ONNX_FILENAME)
    tokenizer_path = os.path.join(output_path, TOKENIZER_FILENAME)

    wrapper = _build_logits_only_wrapper(model)
    wrapper.eval()

    dummy_input = torch.zeros(
        1, MAX_SEQUENCE_LENGTH, dtype=torch.long,
    )

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_input,),
            onnx_path,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "logits": {0: "batch", 1: "sequence"},
            },
        )

    tokenizer.save_pretrained(output_path)
    return onnx_path, tokenizer_path


def _build_logits_only_wrapper(model):
    """Build a torch.nn.Module that returns only logits.

    torch.onnx.export with dynamo cannot handle DynamicCache
    in model output. This wrapper calls the underlying model
    with use_cache=False and return_dict=False, returning
    only the logits tensor.
    """
    import torch

    class _LogitsOnlyModule(torch.nn.Module):
        def __init__(self, inner_model):
            super().__init__()
            self.inner_model = inner_model

        def forward(self, input_ids):
            outputs = self.inner_model(
                input_ids, use_cache=False, return_dict=False,
            )
            return outputs[0]

    return _LogitsOnlyModule(model)
