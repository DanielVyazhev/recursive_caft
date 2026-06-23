from pathlib import Path

from transformers import AutoTokenizer

from core.complexity_estimation.complexity_estimation_runner import ModelGenerateConfig
from core.complexity_estimation.entropy.single_token_entropy_estimator import SingleTokenEntropyEstimator
from core.dataset_samplers.entropy_gain_sampler import EntropyGainSampler
from core.dataset_samplers.entropy_ratio_sampler import BaseDatasetSamplerConfig
from core.datasets.causal_dataset_adapter import CausalDatasetAdapter
from core.datasets.merged_dataset_adapter import MergedDatasetAdapter
from core.datasets.mmlu.mmlu_reasoning_response_dataset import MMLUReasoningResponseDataset
from core.datasets.mmlu.mmlu_single_token_response_dataset import MMLUSingleTokenResponseDataset
from core.datasets.qa_dataset import QADatasetConfig
from core.datasets.qa_dataset_adapter import QADatasetAdapter
from core.evaluation.multi_checkpoint_evaluator import (
    GenerationConfig,
    MultiCheckpointEvaluator,
    MultiCheckpointEvaluatorConfig,
)
from core.training.lora_trainer import LoRASpecificTrainingArgs, LoRATrainingArgs
from core.training.resampling_trainer import ResamplingTrainer, ResamplingTrainerConfig
from core.training.thinking_tokens import setup_thinking_tokens
from core.utils.datasets import add_average_column, merge_mmlu_on_question_id

MODEL_NAME = Path(__file__).resolve().parents[5].joinpath("artifacts/base_models_v0/qwen_3b").as_posix()
OUT_PATH = Path(__file__).parent.joinpath(
    "../../../../../artifacts/distillation_by_metrics/mmlu/entropy_gain_direct_reasoning_trace/qwen_3b"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token
setup_thinking_tokens(tokenizer)

TEACHER_ENTROPY_DATASET_PATH = OUT_PATH.joinpath("teacher_entropy.parquet")
merge_mmlu_on_question_id(
    main_path=Path(__file__).parent.joinpath(
        "../../../../../data/out/splits/random/mmlu/train_distilled_deepseek_v4_flash_regenerate_incorrect_w_large.parquet"
    ),
    extra_paths=[
        Path(__file__).parent.joinpath("../../../../../data/out/single_token_entropy/mmlu_llama_70b.parquet"),
        Path(__file__).parent.joinpath("../../../../../data/out/single_token_entropy/mmlu_qwen_72b.parquet"),
    ],
    extra_columns=[
        {"entropy_value": "llama_70b_entropy_value"},
        {"entropy_value": "qwen_72b_entropy_value"},
    ],
    aggregation_function=lambda df: add_average_column(
        df, "llama_70b_entropy_value", "qwen_72b_entropy_value", "teacher_entropy"
    ),
    save_path=TEACHER_ENTROPY_DATASET_PATH,
)

trainer = ResamplingTrainer(
    config=ResamplingTrainerConfig(
        training_args=LoRATrainingArgs(
            num_train_epochs=50,
            per_device_train_batch_size=4,
        ),
        lora_training_args=LoRASpecificTrainingArgs(train_thinking_token_embeddings=True),
        out_path=OUT_PATH.as_posix(),
        model_id=MODEL_NAME,
        # EntropyGainSampler drops non-positive-gain rows each epoch (drop_non_positive defaults to
        # True): rows where the student is already at/below the teacher's entropy carry no signal, so
        # the per-epoch training set shrinks below top_k over time (see estimation_stats.json
        # "selected_samples"). Because __len__ stays at top_k, the cosine LR schedule is sized for the
        # upper bound and won't fully decay — accepted, as the shrinking set is the point.
        train_dataset=MergedDatasetAdapter(
            [
                CausalDatasetAdapter(
                    dataset=MMLUReasoningResponseDataset(
                        config=QADatasetConfig(
                            path="should be overridden by SetResamplingPathCallback",
                            dataset_id="train_distilled_deepseek_v4_flash_regenerate_incorrect_w_large",
                        ),
                        tokenizer=tokenizer,
                    ),
                    dataset_sampler=EntropyGainSampler(BaseDatasetSamplerConfig(top_k=1024)),
                ),
                # Mix in a smaller single-token-answer set (hardest by gain) so the model keeps
                # answering with a single letter, keeping the per-epoch entropy estimation stable.
                CausalDatasetAdapter(
                    dataset=MMLUSingleTokenResponseDataset(
                        config=QADatasetConfig(
                            path="should be overridden by SetResamplingPathCallback",
                            dataset_id="mmlu_single_token_train",
                        ),
                        tokenizer=tokenizer,
                    ),
                    dataset_sampler=EntropyGainSampler(BaseDatasetSamplerConfig(top_k=256)),
                ),
            ]
        ),
        save_schedule=[1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50],
        complexity_evaluation_dataset=QADatasetAdapter(
            dataset=MMLUSingleTokenResponseDataset(
                config=QADatasetConfig(
                    path=TEACHER_ENTROPY_DATASET_PATH.as_posix(),
                    dataset_id="mmlu_teacher_entropy",
                ),
                tokenizer=tokenizer,
            )
        ),
        complexity_estimator=SingleTokenEntropyEstimator(),
        complexity_estimation_runner_generation_config=ModelGenerateConfig(max_new_tokens=1),
    ),
    tokenizer=tokenizer,
)
trainer.train()
trainer.unload()

cot_evaluator = MultiCheckpointEvaluator(
    config=MultiCheckpointEvaluatorConfig(
        checkpoints_dir=OUT_PATH.as_posix(),
        eval_dataset=QADatasetAdapter(
            dataset=MMLUReasoningResponseDataset(
                config=QADatasetConfig(
                    path=Path(__file__)
                    .parent.joinpath("../../../../../data/out/splits/random/mmlu/test.parquet")
                    .as_posix(),
                    dataset_id="mmlu_random_test",
                ),
                tokenizer=tokenizer,
            ),
            add_thinking_start_token=True,
        ),
        generation=GenerationConfig(max_new_tokens=8500, max_thinking_tokens=8192, max_batch_size=256),
        summary_filename="summary_reasoning_evals.json",
    ),
    tokenizer=tokenizer,
)
cot_evaluator.evaluate_all()
