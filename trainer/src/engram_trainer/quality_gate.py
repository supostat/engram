import os
import random
from dataclasses import dataclass

from engram_trainer.data import Memory


ONNX_FILENAME = "text_generator.onnx"
TOKENIZER_FILENAME = "tokenizer.json"
QUALITY_SCORE_THRESHOLD = 0.5
SAMPLE_COUNT = 10
PASSING_ROUGE_L_THRESHOLD = 0.1


@dataclass
class QualityResult:
    avg_score: float
    samples_tested: int
    passed: bool


def validate_text_generation(
    memories: list[Memory], models_path: str,
) -> QualityResult:
    onnx_path = os.path.join(models_path, ONNX_FILENAME)
    tokenizer_path = os.path.join(models_path, TOKENIZER_FILENAME)

    if not os.path.isfile(onnx_path) or not os.path.isfile(tokenizer_path):
        return QualityResult(avg_score=0.0, samples_tested=0, passed=False)

    quality_memories = [
        m for m in memories if m.score > QUALITY_SCORE_THRESHOLD
    ]
    if not quality_memories:
        return QualityResult(avg_score=0.0, samples_tested=0, passed=False)

    sampled = random.sample(
        quality_memories, min(SAMPLE_COUNT, len(quality_memories)),
    )

    try:
        session, tokenizer = _load_inference_session(
            onnx_path, tokenizer_path,
        )
    except Exception:
        return QualityResult(avg_score=0.0, samples_tested=0, passed=False)

    rouge_scores: list[float] = []
    for memory in sampled:
        try:
            generated = _generate_text(session, tokenizer, memory)
            score = _compute_rouge_l(memory.result, generated)
            rouge_scores.append(score)
        except Exception:
            continue

    if not rouge_scores:
        return QualityResult(avg_score=0.0, samples_tested=0, passed=False)

    avg_score = sum(rouge_scores) / len(rouge_scores)
    return QualityResult(
        avg_score=round(avg_score, 4),
        samples_tested=len(rouge_scores),
        passed=avg_score > PASSING_ROUGE_L_THRESHOLD,
    )


def _load_inference_session(onnx_path: str, tokenizer_path: str):
    import onnxruntime
    from tokenizers import Tokenizer

    session = onnxruntime.InferenceSession(
        onnx_path, providers=["CPUExecutionProvider"],
    )
    tokenizer_directory = os.path.dirname(tokenizer_path)
    tokenizer = Tokenizer.from_file(
        os.path.join(tokenizer_directory, "tokenizer.json"),
    )
    return session, tokenizer


def _generate_text(session, tokenizer, memory: Memory) -> str:
    prompt = f"Context: {memory.context}\nAction: {memory.action}\n"
    encoding = tokenizer.encode(prompt)
    input_ids = encoding.ids

    import numpy as np

    max_new_tokens = 64
    for _ in range(max_new_tokens):
        input_array = np.array([input_ids], dtype=np.int64)
        logits = session.run(None, {"input_ids": input_array})[0]
        next_token_logits = logits[0, -1, :]
        next_token = int(np.argmax(next_token_logits))

        eos_token_id = tokenizer.token_to_id("</s>")
        if eos_token_id is None:
            eos_token_id = tokenizer.token_to_id("<|endoftext|>")
        if next_token == eos_token_id:
            break

        input_ids.append(next_token)

    generated_ids = input_ids[len(encoding.ids):]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def _compute_rouge_l(reference: str, hypothesis: str) -> float:
    reference_tokens = reference.split()
    hypothesis_tokens = hypothesis.split()

    if not reference_tokens or not hypothesis_tokens:
        return 0.0

    lcs_length = _longest_common_subsequence_length(
        reference_tokens, hypothesis_tokens,
    )

    precision = lcs_length / len(hypothesis_tokens)
    recall = lcs_length / len(reference_tokens)

    if precision + recall == 0.0:
        return 0.0

    f_measure = (2.0 * precision * recall) / (precision + recall)
    return f_measure


def _longest_common_subsequence_length(
    sequence_a: list[str], sequence_b: list[str],
) -> int:
    length_a = len(sequence_a)
    length_b = len(sequence_b)

    previous_row = [0] * (length_b + 1)
    current_row = [0] * (length_b + 1)

    for i in range(1, length_a + 1):
        for j in range(1, length_b + 1):
            if sequence_a[i - 1] == sequence_b[j - 1]:
                current_row[j] = previous_row[j - 1] + 1
            else:
                current_row[j] = max(previous_row[j], current_row[j - 1])
        previous_row, current_row = current_row, [0] * (length_b + 1)

    return previous_row[length_b]
