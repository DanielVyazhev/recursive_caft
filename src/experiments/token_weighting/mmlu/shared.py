"""Token-weighted distillation on MMLU reasoning traces: shared runner.

Pilot protocol (one model, one seed): every arm trains on the same train-minus-dev split,
same data order and seed, full 20 epochs with checkpoints at `SAVE_SCHEDULE`, and is
evaluated on the 500-question dev split held out of train (make_mmlu_dev_split.py) with a
2048-token thinking cap. The arms differ only in `token_weighting`:

    uniform  -> Full SFT (stock loss; the reference the other arms are compared against)
    rho      -> reducible-fraction weights (the method)
    discord  -> DISCORD baseline with the same proxy as reference
    kl / eg  -> ablations

Proxy statistics come from one teacher-forced pass of Qwen2.5-72B-Instruct over the traces
(qwen_3b_proxy_pass.py); the cache is keyed by the student tokenization, so it is per
student model.
"""

from pathlib import Path

from transformers import AutoTokenizer

from core.datasets.causal_dataset_adapter import CausalDatasetAdapter
from core.datasets.mmlu.mmlu_reasoning_response_dataset import MMLUReasoningResponseDataset
from core.datasets.qa_dataset import QADatasetConfig
from core.datasets.qa_dataset_adapter import QADatasetAdapter
from core.evaluation.multi_checkpoint_evaluator import (
    GenerationConfig,
    MultiCheckpointEvaluator,
    MultiCheckpointEvaluatorConfig,
)
from core.training.base_trainer import PackingConfig
from core.training.lora_trainer import (
    LoRASpecificTrainingArgs,
    LoRATrainer,
    LoRATrainerConfig,
    LoRATrainingArgs,
    phi4_mini_lora_target_modules,
)
from core.training.packing_budgets import packing_budget
from core.training.thinking_tokens import setup_thinking_tokens
from core.training.token_weighting import TokenWeightingConfig

REPO_ROOT = Path(__file__).resolve().parents[4]
SPLITS_DIR = REPO_ROOT / "data" / "out" / "splits" / "random" / "mmlu"
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "token_weighting" / "mmlu"

TRAIN_DATASET = "train_corrected_answer_deepseek_v4_pro_and_others_head_truncated8192"
DEV_SPLIT = "dev500"
TRAIN_MINUS_DEV = f"{TRAIN_DATASET}_minus_{DEV_SPLIT}"
DEV_DATASET = f"{TRAIN_DATASET}_{DEV_SPLIT}"

PROXY_MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"
PROXY_NICK = "qwen_72b"
PROXY_TOP_K = 32

SAVE_SCHEDULE = [2, 5, 10, 15, 20]
MAX_THINKING_TOKENS = 2048


def base_model_path(model_name: str) -> str:
    return (REPO_ROOT / "artifacts" / "base_models_v0" / model_name).as_posix()


def split_path(dataset_name: str) -> str:
    return (SPLITS_DIR / f"{dataset_name}.parquet").as_posix()


def proxy_cache_path(model_name: str, proxy_nick: str = PROXY_NICK, train_dataset: str = TRAIN_DATASET) -> str:
    return (REPO_ROOT / "data" / "out" / "proxy_cache" / "mmlu" / proxy_nick / model_name / train_dataset).as_posix()


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path(model_name), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    setup_thinking_tokens(tokenizer)
    return tokenizer


def reasoning_adapter(tokenizer, dataset_name: str) -> CausalDatasetAdapter:
    return CausalDatasetAdapter(
        dataset=MMLUReasoningResponseDataset(
            config=QADatasetConfig(path=split_path(dataset_name), dataset_id=f"mmlu_{dataset_name}"),
            tokenizer=tokenizer,
        )
    )


def run(
    model_name: str,
    relative_out_path: str,
    token_weighting: TokenWeightingConfig | None,
    train_dataset: str = TRAIN_MINUS_DEV,
    eval_dataset: str = DEV_DATASET,
    eval_split: str = DEV_SPLIT,
    save_schedule: list[int] | None = None,
    max_thinking_tokens: int = MAX_THINKING_TOKENS,
    skip_missing_checkpoints: bool = True,
    run_evaluation: bool = True,
    lora_training_args: LoRASpecificTrainingArgs | None = None,
):
    save_schedule = save_schedule or SAVE_SCHEDULE
    tokenizer = load_tokenizer(model_name)
    out_path = (ARTIFACTS_DIR / relative_out_path).as_posix()

    lora_training_args = lora_training_args or LoRASpecificTrainingArgs()
    lora_training_args.train_thinking_token_embeddings = True
    if model_name == "phi4_mini":
        lora_training_args.target_modules = phi4_mini_lora_target_modules

    trainer = LoRATrainer(
        config=LoRATrainerConfig(
            out_path=out_path,
            model_id=base_model_path(model_name),
            train_dataset=reasoning_adapter(tokenizer, train_dataset),
            training_args=LoRATrainingArgs(num_train_epochs=save_schedule[-1], per_device_train_batch_size=1),
            lora_training_args=lora_training_args,
            packing=PackingConfig(budget=packing_budget(model_name)),
            save_schedule=save_schedule,
            token_weighting=token_weighting,
        ),
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.unload()

    if not run_evaluation:
        return

    cot_evaluator = MultiCheckpointEvaluator(
        config=MultiCheckpointEvaluatorConfig(
            checkpoints_dir=out_path,
            eval_dataset=QADatasetAdapter(
                dataset=MMLUReasoningResponseDataset(
                    config=QADatasetConfig(
                        path=split_path(eval_dataset),
                        dataset_id=f"mmlu_{eval_split}_cap{max_thinking_tokens}",
                    ),
                    tokenizer=tokenizer,
                ),
                add_thinking_start_token=True,
            ),
            generation=GenerationConfig(
                max_new_tokens=max_thinking_tokens + 10, max_thinking_tokens=max_thinking_tokens, max_batch_size=256
            ),
            summary_filename=f"summary_reasoning_evals_{eval_split}_cap{max_thinking_tokens}.json",
            skip_missing_checkpoints=skip_missing_checkpoints,
        ),
        tokenizer=tokenizer,
    )
    cot_evaluator.evaluate_all()
