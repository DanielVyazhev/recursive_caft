"""Token-weighted distillation on MMLU reasoning traces: shared runner.

Pilot protocol (one model, one seed, one 40 GB GPU): every arm trains on the same fixed
1024-trace subset of train-minus-dev (make_mmlu_dev_split.py), same data order and seed,
no single-token mix, no packing (as the budget20 runs), 20 epochs with checkpoints at
`SAVE_SCHEDULE`, evaluated on the 500-question dev split held out of train with a 2048-token
thinking cap. The arms differ only in `token_weighting`:

    uniform  -> Full SFT (stock loss and collator; the reference the other arms are compared to)
    rho      -> reducible-fraction weights (the method)
    discord  -> DISCORD baseline with the same proxy as reference
    kl / eg  -> ablations

Proxy statistics come from one teacher-forced pass of the proxy over the traces
(qwen_3b_proxy_pass.py); the cache is keyed by the student tokenization, so it is per student
model and per dataset file.

Paths: student models are read from $RECURSIVE_CAFT_MODELS_DIR, else /mnt/data198/LLM/agents
when it exists (dataminer8 layout, dirs named after the HF repos), else artifacts/base_models_v0
(the zhores layout). The proxy is $RECURSIVE_CAFT_PROXY_MODEL, else the local
/mnt/data198/LLM/models/Qwen2.5-14B-Instruct, else the hub id: the largest Qwen2.5 that runs in
bf16 on one 40 GB card.
"""

import importlib.util
import os
from pathlib import Path
from typing import override

from transformers import (
    AutoTokenizer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from core.complexity_estimation.entropy.single_token_entropy_with_random_estimator import (
    SingleTokenEntropyWithRandomEstimator,
)
from core.dataset_samplers.base_sampler import (
    BaseDatasetSampler,
    BaseDatasetSamplerConfig,
)
from core.dataset_samplers.random_sampler import RandomSampler
from core.datasets.causal_dataset_adapter import CausalDatasetAdapter
from core.datasets.merged_dataset_adapter import MergedDatasetAdapter
from core.datasets.mmlu.mmlu_reasoning_response_dataset import (
    MMLUReasoningResponseDataset,
)
from core.datasets.mmlu.mmlu_single_token_response_dataset import (
    MMLUSingleTokenResponseDataset,
)
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
from core.training.resampling_trainer import (
    ModelGenerateConfig,
    ResamplingTrainer,
    ResamplingTrainerConfig,
)
from core.training.thinking_tokens import setup_thinking_tokens
from core.training.token_weighting import TokenWeightingConfig
from core.utils.datasets import add_average_column, merge_mmlu_on_question_id

REPO_ROOT = Path(__file__).resolve().parents[4]
SPLITS_DIR = REPO_ROOT / "data" / "out" / "splits" / "random" / "mmlu"
TEACHER_ENTROPY_DIR = REPO_ROOT / "data" / "out" / "single_token_entropy_normalized"
TEST_DATASET = "test"
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "token_weighting" / "mmlu"
SERVER_AGENTS_DIR = Path("/mnt/data198/LLM/agents")
SERVER_PROXY_DIR = Path("/mnt/data198/LLM/models/Qwen2.5-14B-Instruct")
# Student nick (packing budgets, LoRA target modules) -> directory name under the agents dir.
SERVER_MODEL_DIRS = {
    "qwen_3b": "Qwen2.5-3B-Instruct",
    "phi4_mini": "Phi-4-mini-instruct",
    "llama_3b": "llama_3b",
}

TRAIN_DATASET = "train_corrected_answer_deepseek_v4_pro_and_others_head_truncated8192"
DEV_SPLIT = "dev500"
TRAIN_MINUS_DEV = f"{TRAIN_DATASET}_minus_{DEV_SPLIT}"
TRAIN_FIXED = f"{TRAIN_MINUS_DEV}_fixed1024"
DEV_DATASET = f"{TRAIN_DATASET}_{DEV_SPLIT}"

PROXY_MODEL_ID = os.environ.get(
    "RECURSIVE_CAFT_PROXY_MODEL",
    SERVER_PROXY_DIR.as_posix()
    if SERVER_PROXY_DIR.exists()
    else "Qwen/Qwen2.5-14B-Instruct",
)
PROXY_NICK = Path(PROXY_MODEL_ID).name.lower()
PROXY_TOP_K = 32

SAVE_SCHEDULE = [2, 5, 10, 15, 20]
MAX_THINKING_TOKENS = 2048


def base_model_path(model_name: str) -> str:
    env = os.environ.get("RECURSIVE_CAFT_MODELS_DIR")
    if env:
        return (Path(env) / SERVER_MODEL_DIRS.get(model_name, model_name)).as_posix()
    if SERVER_AGENTS_DIR.exists():
        return (
            SERVER_AGENTS_DIR / SERVER_MODEL_DIRS.get(model_name, model_name)
        ).as_posix()
    return (REPO_ROOT / "artifacts" / "base_models_v0" / model_name).as_posix()


def split_path(dataset_name: str) -> str:
    return (SPLITS_DIR / f"{dataset_name}.parquet").as_posix()


def eval_attn_implementation() -> str:
    # The evaluator defaults to FlashAttention-2; fall back to SDPA where flash_attn is not
    # installed (the training arms do not need it either without packing).
    return (
        "flash_attention_2"
        if importlib.util.find_spec("flash_attn") is not None
        else "sdpa"
    )


def proxy_cache_path(
    model_name: str, dataset_name: str = TRAIN_FIXED, proxy_nick: str = PROXY_NICK
) -> str:
    return (
        REPO_ROOT
        / "data"
        / "out"
        / "proxy_cache"
        / "mmlu"
        / proxy_nick
        / model_name
        / dataset_name
    ).as_posix()


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path(model_name), trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    setup_thinking_tokens(tokenizer)
    return tokenizer


def reasoning_adapter(tokenizer, dataset_name: str) -> CausalDatasetAdapter:
    return CausalDatasetAdapter(
        dataset=MMLUReasoningResponseDataset(
            config=QADatasetConfig(
                path=split_path(dataset_name), dataset_id=f"mmlu_{dataset_name}"
            ),
            tokenizer=tokenizer,
        )
    )


def run(
    model_name: str,
    relative_out_path: str,
    token_weighting: TokenWeightingConfig | None,
    train_dataset: str = TRAIN_FIXED,
    eval_dataset: str = DEV_DATASET,
    eval_split: str = DEV_SPLIT,
    save_schedule: list[int] | None = None,
    max_thinking_tokens: int = MAX_THINKING_TOKENS,
    packing: bool = False,
    seed: int = 42,
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
            training_args=LoRATrainingArgs(
                num_train_epochs=save_schedule[-1],
                per_device_train_batch_size=1,
                seed=seed,
                data_seed=seed,
            ),
            lora_training_args=lora_training_args,
            packing=PackingConfig(budget=packing_budget(model_name))
            if packing
            else None,
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
                max_new_tokens=max_thinking_tokens + 10,
                max_thinking_tokens=max_thinking_tokens,
                max_batch_size=256,
                attn_implementation=eval_attn_implementation(),
            ),
            summary_filename=f"summary_reasoning_evals_{eval_split}_cap{max_thinking_tokens}.json",
            skip_missing_checkpoints=skip_missing_checkpoints,
        ),
        tokenizer=tokenizer,
    )
    cot_evaluator.evaluate_all()


# --- Paper-3 regime: a fresh 1024-trace draw from the whole pool every epoch -----------------
#
# The fixed-subset pilot above is data-limited: after ~5 epochs on the same 1024 traces every arm
# sits on the same plateau and later epochs only measure how gracefully it overfits. The runs of
# distillation_by_metrics instead re-select 1024 traces (+256 single-letter rows) from the full
# train pool at the start of every epoch, and accuracy keeps rising with the number of distinct
# traces seen. `run_resampling` is that protocol (ResamplingTrainer, RandomSampler, the same
# merged adapter, batch 2 x 32, test split) with `token_weighting` plugged in, so an arm here is
# directly comparable to artifacts/distillation_by_metrics/mmlu/random/qwen_3b_* when it keeps
# the 200-epoch cosine schedule those runs used; `stop_after_epoch` ends training early
# without changing that schedule. The proxy cache must cover the whole pool
# (qwen_3b_proxy_pass_full.py); the single-letter rows have no cache entry and train at
# weight 1 (strict_cache=False), which shows up as a tiny tw/missing_frac.

RESAMPLING_TOTAL_EPOCHS = 200
RESAMPLING_SAVE_SCHEDULE = [10, 20, 50]


class StopAfterEpochCallback(TrainerCallback):
    """Stop the run after `epoch` completed epochs, leaving the LR schedule (built from
    num_train_epochs) untouched; the checkpoint of that epoch is saved first."""

    def __init__(self, epoch: int):
        self.epoch = epoch

    @override
    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        assert state.epoch is not None
        if round(state.epoch) >= self.epoch:
            control.should_training_stop = True
        return control


class _ResamplingTrainerWithStop(ResamplingTrainer):
    def __init__(self, *args, stop_after_epoch: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._stop_after_epoch = stop_after_epoch

    @override
    def _build_trainer(self, train_ds):
        trainer = super()._build_trainer(train_ds)
        if self._stop_after_epoch is not None:
            trainer.add_callback(StopAfterEpochCallback(self._stop_after_epoch))
        return trainer

    @override
    def _directory_is_empty(self, directory: str, expected_epochs: int) -> bool:
        # A run stopped at stop_after_epoch is complete at that epoch, not at num_train_epochs:
        # a rerun (e.g. for evaluation) must not resume it for one more epoch.
        if self._stop_after_epoch is not None:
            expected_epochs = self._stop_after_epoch
        return super()._directory_is_empty(directory, expected_epochs)


def teacher_entropy_dataset_path(
    out_path: Path, train_dataset: str = TRAIN_DATASET
) -> Path:
    """The train pool merged with the 70B/72B single-token entropies (what the complexity runner
    scores and the samplers draw from). Rebuilt per run directory, as in distillation_by_metrics."""
    save_path = out_path / "teacher_entropy.parquet"
    merge_mmlu_on_question_id(
        main_path=split_path(train_dataset),
        extra_paths=[
            TEACHER_ENTROPY_DIR / "mmlu_llama_70b.parquet",
            TEACHER_ENTROPY_DIR / "mmlu_qwen_72b.parquet",
        ],
        extra_columns=[
            {"entropy_value": "llama_70b_entropy_value"},
            {"entropy_value": "qwen_72b_entropy_value"},
        ],
        aggregation_function=lambda df: add_average_column(
            df, "llama_70b_entropy_value", "qwen_72b_entropy_value", "teacher_entropy"
        ),
        save_path=save_path,
    )
    return save_path


def resampling_adapter(
    sampler_cls: type[BaseDatasetSampler],
    reasoning_top_k: int = 1024,
    single_token_top_k: int = 256,
) -> MergedDatasetAdapter:
    # Same two children as distillation_by_metrics.mmlu.shared.get_merged_adapter_with_data_mix:
    # the traces, plus a small single-letter-answer set drawn by the same policy so the model keeps
    # answering with one letter and the per-epoch entropy estimation stays measurable.
    return MergedDatasetAdapter(
        [
            CausalDatasetAdapter(
                dataset=MMLUReasoningResponseDataset(
                    config=QADatasetConfig(
                        path="should be overridden by SetResamplingPathCallback",
                        dataset_id=TRAIN_DATASET,
                    ),
                    tokenizer=None,  # type: ignore[arg-type]  # set by override_tokenizer below
                ),
                dataset_sampler=sampler_cls(
                    BaseDatasetSamplerConfig(top_k=reasoning_top_k)
                ),
            ),
            CausalDatasetAdapter(
                dataset=MMLUSingleTokenResponseDataset(
                    config=QADatasetConfig(
                        path="should be overridden by SetResamplingPathCallback",
                        dataset_id="mmlu_single_token_train",
                    ),
                    tokenizer=None,  # type: ignore[arg-type]
                ),
                dataset_sampler=sampler_cls(
                    BaseDatasetSamplerConfig(top_k=single_token_top_k)
                ),
            ),
        ]
    )


def run_resampling(
    model_name: str,
    relative_out_path: str,
    token_weighting: TokenWeightingConfig | None,
    sampler_cls: type[BaseDatasetSampler] = RandomSampler,
    save_schedule: list[int] | None = None,
    total_epochs: int = RESAMPLING_TOTAL_EPOCHS,
    stop_after_epoch: int | None = None,
    train_dataset: str = TRAIN_DATASET,
    max_thinking_tokens: int = MAX_THINKING_TOKENS,
    seed: int = 42,
    shuffle: bool = False,
    skip_missing_checkpoints: bool = True,
    run_evaluation: bool = True,
    lora_training_args: LoRASpecificTrainingArgs | None = None,
    use_teacher_entropy: bool = False,
):
    """`use_teacher_entropy` merges the Llama-70B/Qwen-72B single-token entropies into the pool
    (data/out/single_token_entropy_normalized), which only the entropy-gain samplers read; the
    random re-draw scores rows by `random_value` alone and draws straight from the train parquet."""
    save_schedule = save_schedule or RESAMPLING_SAVE_SCHEDULE
    if stop_after_epoch is None:
        stop_after_epoch = save_schedule[-1]
    assert stop_after_epoch <= total_epochs and save_schedule[-1] <= stop_after_epoch
    if token_weighting is not None and token_weighting.needs_proxy():
        assert not token_weighting.strict_cache, (
            "the single-letter rows have no proxy entry; use strict_cache=False"
        )

    tokenizer = load_tokenizer(model_name)
    out_path = ARTIFACTS_DIR / relative_out_path
    out_path.mkdir(parents=True, exist_ok=True)

    lora_training_args = lora_training_args or LoRASpecificTrainingArgs()
    lora_training_args.train_thinking_token_embeddings = True
    if model_name == "phi4_mini":
        lora_training_args.target_modules = phi4_mini_lora_target_modules

    pool_path = (
        teacher_entropy_dataset_path(out_path, train_dataset)
        if use_teacher_entropy
        else Path(split_path(train_dataset))
    )
    train_adapter = resampling_adapter(sampler_cls)
    train_adapter.override_tokenizer(tokenizer)

    trainer = _ResamplingTrainerWithStop(
        config=ResamplingTrainerConfig(
            training_args=LoRATrainingArgs(
                num_train_epochs=total_epochs,
                per_device_train_batch_size=2,
                seed=seed,
                data_seed=seed,
            ),
            lora_training_args=lora_training_args,
            save_schedule=save_schedule,
            shuffle=shuffle,
            out_path=out_path.as_posix(),
            model_id=base_model_path(model_name),
            train_dataset=train_adapter,
            complexity_evaluation_dataset=QADatasetAdapter(
                dataset=MMLUSingleTokenResponseDataset(
                    config=QADatasetConfig(
                        path=pool_path.as_posix(), dataset_id="mmlu_teacher_entropy"
                    ),
                    tokenizer=tokenizer,
                )
            ),
            complexity_estimator=SingleTokenEntropyWithRandomEstimator(),
            complexity_estimation_runner_generation_config=ModelGenerateConfig(
                max_new_tokens=1
            ),
            token_weighting=token_weighting,
        ),
        tokenizer=tokenizer,
        stop_after_epoch=stop_after_epoch,
    )
    trainer.train()
    trainer.unload()

    if not run_evaluation:
        return

    eval_dataset_id = f"mmlu_random_test_cap{max_thinking_tokens}"
    cot_evaluator = MultiCheckpointEvaluator(
        config=MultiCheckpointEvaluatorConfig(
            checkpoints_dir=out_path.as_posix(),
            eval_dataset=QADatasetAdapter(
                dataset=MMLUReasoningResponseDataset(
                    config=QADatasetConfig(
                        path=split_path(TEST_DATASET), dataset_id=eval_dataset_id
                    ),
                    tokenizer=tokenizer,
                ),
                add_thinking_start_token=True,
            ),
            generation=GenerationConfig(
                max_new_tokens=max_thinking_tokens + 10,
                max_thinking_tokens=max_thinking_tokens,
                max_batch_size=256,
                attn_implementation=eval_attn_implementation(),
            ),
            summary_filename=f"summary_reasoning_evals_cap{max_thinking_tokens}.json",
            skip_missing_checkpoints=skip_missing_checkpoints,
        ),
        tokenizer=tokenizer,
    )
    cot_evaluator.evaluate_all()


def load_student(
    model_name: str, checkpoint_dir: str | Path | None = None, torch_dtype=None
):
    """The student for diagnostics: the base model, or a training checkpoint (LoRA adapter plus
    the trained <think>/</think> embedding rows, loaded the way the evaluator loads them)."""
    import torch
    from transformers import AutoModelForCausalLM

    from core.utils.device import DEVICE

    torch_dtype = torch_dtype or torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path(model_name), trust_remote_code=True, torch_dtype=torch_dtype
    ).to(DEVICE)
    if checkpoint_dir is None:
        return model

    from peft import PeftModel

    from core.training.callbacks.save_thinking_token_rows import ROWS_FILENAME

    checkpoint_dir = Path(checkpoint_dir)
    rows_path = checkpoint_dir / ROWS_FILENAME
    if rows_path.exists():
        payload = torch.load(rows_path, map_location="cpu", weights_only=True)
        ids = torch.tensor(payload["new_ids"], dtype=torch.long)
        with torch.no_grad():
            in_w = model.get_input_embeddings().weight
            in_w.data[ids.to(in_w.device)] = payload["input_rows"].to(
                dtype=in_w.dtype, device=in_w.device
            )
            if "output_rows" in payload:
                out_layer = model.get_output_embeddings()
                assert out_layer is not None
                out_w = out_layer.weight
                out_w.data[ids.to(out_w.device)] = payload["output_rows"].to(
                    dtype=out_w.dtype, device=out_w.device
                )
    return PeftModel.from_pretrained(model, checkpoint_dir.as_posix()).eval()
