"""GSM8K counterpart of experiments/distillation_by_metrics/mmlu/shared.py.

Same pipeline and training config as the MMLU-Pro runs (LoRA, shuffle, 1024 reasoning rows + 256
answer-format rows per epoch, re-estimated every epoch), with these GSM8K-specific choices:

* pool: the official GSM8K train split (7473), eval: the official test split (1319);
* reasoning traces: the human-written GSM8K solutions (`answer_w_steps`), no teacher generation;
* proxy: Qwen2.5-72B only (Llama-3.3-70B answers with a number in only 27% of rows, so its answer
  entropy is mostly measured on the start of a reasoning chain);
* entropy: the answer is several tokens, so the per-row entropy is aggregated over the answer tokens
  (ENTROPY_AGGREGATION, fixed before the runs), for the student online and for the proxy offline;
* online estimation: the prompt ends with an empty `<think></think>` and the answer is the tokens the
  student generates after it -- the format the answer-format training rows teach;
* 100 epochs, CoT eval with a 2048-token thinking cap only.

Every run first validates answer formatting/grading on the train and test splits
(`validate_gsm8k_pipeline`) and audits the grading of every eval response afterwards
(`audit_gsm8k_eval`).

Usage:  uv run src/experiments/distillation_by_metrics/gsm8k/<arm>/<model>.py [--seed N]
"""

import argparse
from collections.abc import Callable
from pathlib import Path

from transformers import AutoTokenizer

from core.complexity_estimation.entropy.answer_tokens_entropy_with_random_estimator import (
    AnswerTokensEntropyWithRandomEstimator,
)
from core.dataset_samplers.base_sampler import BaseDatasetSampler, BaseDatasetSamplerConfig
from core.dataset_samplers.entropy_gain_proportional_sampler import EntropyGainProportionalSampler
from core.dataset_samplers.random_sampler import RandomSampler
from core.dataset_samplers.student_entropy_proportional_sampler import StudentEntropyProportionalSampler
from core.datasets.causal_dataset_adapter import CausalDatasetAdapter
from core.datasets.gsm8k.gsm8k_empty_thinking_response_dataset import GSM8KEmptyThinkingResponseDataset
from core.datasets.gsm8k.gsm8k_reasoning_response_dataset import GSM8KReasoningResponseDataset
from core.datasets.gsm8k.gsm8k_validation import audit_gsm8k_eval, validate_gsm8k_pipeline
from core.datasets.merged_dataset_adapter import MergedDatasetAdapter
from core.datasets.qa_dataset import QADatasetConfig
from core.datasets.qa_dataset_adapter import QADatasetAdapter
from core.evaluation.multi_checkpoint_evaluator import (
    GenerationConfig,
    MultiCheckpointEvaluator,
    MultiCheckpointEvaluatorConfig,
)
from core.training.lora_trainer import (
    LoRASpecificTrainingArgs,
    LoRATrainingArgs,
    phi4_mini_lora_target_modules,
)
from core.training.resampling_trainer import ModelGenerateConfig, ResamplingTrainer, ResamplingTrainerConfig
from core.training.thinking_tokens import setup_thinking_tokens
from core.utils.datasets import merge_mmlu_on_question_id

REPO_ROOT = Path(__file__).parent.joinpath("../../../..")
TRAIN_PATH = REPO_ROOT / "data/source/gsm8k/gsm8k_train.parquet"
TEST_PATH = REPO_ROOT / "data/source/gsm8k/gsm8k_test.parquet"
PROXY_MODEL = "qwen_72b"
PROXY_ENTROPY_PATH = REPO_ROOT / f"data/out/single_token_entropy_normalized/gsm8k_{PROXY_MODEL}.parquet"

# Answer-token entropy aggregate, for the student (online) and the proxy (offline column
# f"{ENTROPY_AGGREGATION}_entropy"). "max" had the best ROC-AUC vs. Qwen2.5-3B direct-answer errors
# on the train pool (max 0.758 / mean 0.741 / first 0.711). Fixed before the runs; do not tune.
ENTROPY_AGGREGATION = "max"

MODELS = ("qwen_3b", "llama_3b", "phi4_mini")
SAVE_SCHEDULE = [10, 20, 50, 100]
REASONING_TOP_K = 1024
ANSWER_FORMAT_TOP_K = 256
# Longest gold answer is 9 digits; Qwen/Phi tokenize digits one by one, Llama in groups of three.
ESTIMATION_MAX_NEW_TOKENS = 16
MAX_THINKING_TOKENS = 2048
# Tokens left for the answer once the thinking cap forces </think>.
ANSWER_TOKEN_BUDGET = 24

SAMPLERS: dict[str, type[BaseDatasetSampler]] = {
    "random": RandomSampler,
    "student_entropy_proportional": StudentEntropyProportionalSampler,
    "entropy_gain_proportional": EntropyGainProportionalSampler,
}


def run(
    arm: str,
    model_name: str,
    seed: int = 42,
    save_schedule: list[int] | None = None,
    skip_missing_checkpoints: bool = False,
    lora_training_args: LoRASpecificTrainingArgs | None = None,
):
    assert arm in SAMPLERS, f"Unknown arm {arm!r}; expected one of {sorted(SAMPLERS)}"
    assert model_name in MODELS, f"Unknown model {model_name!r}; expected one of {MODELS}"
    save_schedule = save_schedule or SAVE_SCHEDULE

    MODEL_NAME = REPO_ROOT.joinpath(f"artifacts/base_models_v0/{model_name}").as_posix()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    setup_thinking_tokens(tokenizer)

    # Fail before any GPU work if an answer could be formatted or graded wrongly.
    validate_gsm8k_pipeline(tokenizer, [TRAIN_PATH, TEST_PATH])

    suffix = "" if seed == 42 else f"_seed{seed}"
    OUT_PATH = REPO_ROOT.joinpath("artifacts/distillation_by_metrics/gsm8k").joinpath(arm, f"{model_name}{suffix}")

    lora_training_args = lora_training_args or LoRASpecificTrainingArgs()
    lora_training_args.train_thinking_token_embeddings = True
    if model_name == "phi4_mini":
        lora_training_args.target_modules = phi4_mini_lora_target_modules

    proxy_column = f"{PROXY_MODEL}_{ENTROPY_AGGREGATION}_entropy"
    TEACHER_ENTROPY_DATASET_PATH = OUT_PATH.joinpath("teacher_entropy.parquet")
    merged = merge_mmlu_on_question_id(
        main_path=TRAIN_PATH,
        extra_paths=[PROXY_ENTROPY_PATH],
        extra_columns=[{f"{ENTROPY_AGGREGATION}_entropy": proxy_column}],
        aggregation_function=lambda df: df.assign(teacher_entropy=df[proxy_column].astype(float)),
        save_path=TEACHER_ENTROPY_DATASET_PATH,
    )
    assert merged["teacher_entropy"].notna().all(), f"Missing {PROXY_MODEL} entropy for some GSM8K train rows"

    train_dataset_adapter = get_merged_adapter_with_data_mix(SAMPLERS[arm])
    train_dataset_adapter.override_tokenizer(tokenizer)

    trainer = ResamplingTrainer(
        config=ResamplingTrainerConfig(
            training_args=LoRATrainingArgs(
                num_train_epochs=save_schedule[-1], per_device_train_batch_size=2, seed=seed, data_seed=seed
            ),
            lora_training_args=lora_training_args,
            save_schedule=save_schedule,
            shuffle=True,
            out_path=OUT_PATH.as_posix(),
            model_id=MODEL_NAME,
            train_dataset=train_dataset_adapter,
            complexity_evaluation_dataset=QADatasetAdapter(
                dataset=GSM8KEmptyThinkingResponseDataset(
                    config=QADatasetConfig(
                        path=TEACHER_ENTROPY_DATASET_PATH.as_posix(),
                        dataset_id="gsm8k_teacher_entropy",
                    ),
                    tokenizer=tokenizer,
                ),
                add_empty_thinking=True,
            ),
            complexity_estimator=AnswerTokensEntropyWithRandomEstimator(aggregation=ENTROPY_AGGREGATION),
            complexity_estimation_runner_generation_config=ModelGenerateConfig(
                max_new_tokens=ESTIMATION_MAX_NEW_TOKENS
            ),
        ),
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.unload()

    eval_dataset_id = f"gsm8k_test_cap{MAX_THINKING_TOKENS}"
    MultiCheckpointEvaluator(
        config=MultiCheckpointEvaluatorConfig(
            checkpoints_dir=OUT_PATH.as_posix(),
            eval_dataset=QADatasetAdapter(
                dataset=GSM8KReasoningResponseDataset(
                    config=QADatasetConfig(path=TEST_PATH.as_posix(), dataset_id=eval_dataset_id),
                    tokenizer=tokenizer,
                ),
                add_thinking_start_token=True,
            ),
            generation=GenerationConfig(
                max_new_tokens=MAX_THINKING_TOKENS + ANSWER_TOKEN_BUDGET,
                max_thinking_tokens=MAX_THINKING_TOKENS,
                max_batch_size=256,
            ),
            summary_filename=f"summary_reasoning_evals_cap{MAX_THINKING_TOKENS}.json",
            skip_missing_checkpoints=skip_missing_checkpoints,
        ),
        tokenizer=tokenizer,
    ).evaluate_all()

    audit_gsm8k_eval(
        checkpoints_dir=OUT_PATH,
        eval_dataset_id=eval_dataset_id,
        test_path=TEST_PATH,
        tokenizer=tokenizer,
        out_file=OUT_PATH / f"answer_parsing_audit_cap{MAX_THINKING_TOKENS}.json",
    )


def get_merged_adapter_with_data_mix(sampler_cls: type[BaseDatasetSampler]) -> MergedDatasetAdapter:
    return get_merged_adapter_with_data_mix_from_factory(
        lambda top_k: sampler_cls(BaseDatasetSamplerConfig(top_k=top_k))
    )


def get_merged_adapter_with_data_mix_from_factory(
    sampler_factory: Callable[[int], BaseDatasetSampler],
) -> MergedDatasetAdapter:
    return MergedDatasetAdapter(
        [
            CausalDatasetAdapter(
                dataset=GSM8KReasoningResponseDataset(
                    config=QADatasetConfig(
                        path="should be overridden by SetResamplingPathCallback",
                        dataset_id="gsm8k_train_reasoning",
                    ),
                    # Will be overridden
                    tokenizer=None,  # type: ignore
                ),
                dataset_sampler=sampler_factory(REASONING_TOP_K),
            ),
            # Mix in a smaller `<think></think>N` set selected by the same policy so the student keeps
            # answering with a bare number right after </think>, keeping per-epoch entropy estimation
            # (which prompts with an empty <think></think>) measurable.
            CausalDatasetAdapter(
                dataset=GSM8KEmptyThinkingResponseDataset(
                    config=QADatasetConfig(
                        path="should be overridden by SetResamplingPathCallback",
                        dataset_id="gsm8k_train_empty_thinking",
                    ),
                    # Will be overridden
                    tokenizer=None,  # type: ignore
                ),
                dataset_sampler=sampler_factory(ANSWER_FORMAT_TOP_K),
            ),
        ]
    )


def main(arm: str, model_name: str) -> None:
    parser = argparse.ArgumentParser(description=f"GSM8K distillation: {arm} / {model_name}")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(arm=arm, model_name=model_name, seed=args.seed)
