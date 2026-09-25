"""Budget-20 experiment: does entropy-gain top-k acquire *better* teacher traces, or just fewer?

Entropy-gain top-k re-selects the 1024 highest-gain questions every epoch. Because it keeps
re-selecting the same core, it needs far fewer distinct teacher traces than random sampling for
the same student compute. Three arms separate the two explanations:

- random:          RandomSampler, a fresh uniform draw every epoch.
- entropy_gain:    EntropyGainSampler top-k.
- matched_random:  MatchedAcquisitionRandomSampler replaying entropy_gain's per-epoch acquisition
                   curve (same number of new and reused traces per epoch, same seed), but with
                   uniformly random questions.

entropy_gain vs matched_random isolates *which* traces are acquired; matched_random vs random
isolates *how many*. All arms share the 20-epoch schedule (so identical LR warmup/cosine and
checkpoints), the 1024 trace + 256 single-token mix, per-epoch shuffling and the estimator.

matched_random reads the entropy_gain run of the same model and seed from disk, so run it after
that run has finished all 20 epochs.
"""

import json

import pandas as pd

from core.complexity_estimation.entropy.single_token_entropy_with_random_estimator import (
    SingleTokenEntropyWithRandomEstimator,
)
from core.dataset_samplers.base_sampler import BaseDatasetSampler, BaseDatasetSamplerConfig
from core.dataset_samplers.entropy_gain_sampler import EntropyGainSampler
from core.dataset_samplers.matched_acquisition import AcquisitionStep, acquisition_schedule
from core.dataset_samplers.matched_acquisition_random_sampler import (
    MatchedAcquisitionRandomSampler,
    MatchedAcquisitionRandomSamplerConfig,
)
from core.dataset_samplers.random_sampler import RandomSampler
from experiments.distillation_by_metrics.mmlu.shared import (
    COMPLEXITY_EVALUATION_DATASET_ID,
    get_merged_adapter_with_data_mix_from_factory,
    out_path_for,
    run,
)

ARMS = ("random", "entropy_gain", "matched_random")
SAVE_SCHEDULE = [5, 10, 15, 20]
EPOCHS = SAVE_SCHEDULE[-1]
# Must match the trace adapter's top_k in get_merged_adapter_with_data_mix_from_factory.
TRACE_TOP_K = 1024
TRAIN_DATASET = "train_corrected_answer_deepseek_v4_pro_and_others_head_truncated8192"
SCHEDULE_FILENAME = "acquisition_schedule.json"


def relative_out_path(arm: str, model_name: str, seed: int) -> str:
    suffix = "" if seed == 42 else f"_seed{seed}"
    return f"./budget20/{arm}/{model_name}_head_truncated8192{suffix}"


def epoch_dump(arm: str, model_name: str, seed: int, epoch: int) -> pd.DataFrame:
    path = (
        out_path_for(relative_out_path(arm, model_name, seed))
        / "resampling_trainer_data"
        / str(epoch)
        / "complexity_estimation"
        / f"{COMPLEXITY_EVALUATION_DATASET_ID}.parquet"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing epoch-{epoch} dump of the {arm} arm: {path}")
    return pd.read_parquet(path)


def load_schedule(model_name: str, seed: int) -> list[AcquisitionStep]:
    path = out_path_for(relative_out_path("matched_random", model_name, seed)) / SCHEDULE_FILENAME
    return [AcquisitionStep(*step) for step in json.loads(path.read_text())]


def sampler_for(arm: str, top_k: int, seed: int, schedule: list[AcquisitionStep] | None = None) -> BaseDatasetSampler:
    if arm == "random":
        return RandomSampler(BaseDatasetSamplerConfig(top_k=top_k))
    if arm == "entropy_gain":
        return EntropyGainSampler(BaseDatasetSamplerConfig(top_k=top_k))
    if arm == "matched_random":
        assert schedule is not None, "matched_random needs the entropy_gain acquisition schedule"
        return MatchedAcquisitionRandomSampler(
            MatchedAcquisitionRandomSamplerConfig(top_k=top_k, schedule=schedule, seed=seed)
        )
    raise ValueError(f"Unknown arm {arm!r}; expected one of {ARMS}")


def trace_selections(
    arm: str, model_name: str, seed: int, schedule: list[AcquisitionStep] | None = None
) -> list[list[str]]:
    """Question ids each epoch trained on with teacher traces, reconstructed from the per-epoch dumps.

    The trace adapter's sampler runs on the raw dump before tokenization (BaseDatasetAdapter.
    process_dataset), so replaying it on the saved parquet reproduces the training selection.
    """
    sampler = sampler_for(arm, TRACE_TOP_K, seed, schedule)
    selections = []
    for epoch in range(EPOCHS):
        sampler.set_epoch(epoch)
        selected = sampler._select(epoch_dump(arm, model_name, seed, epoch))
        selections.append(selected["question_id"].astype(str).tolist())
    return selections


def run_arm(arm: str, model_name: str, seed: int = 42) -> None:
    schedule = None
    if arm == "matched_random":
        # Fails before any model is loaded if the entropy_gain run is missing or unfinished.
        schedule = acquisition_schedule(trace_selections("entropy_gain", model_name, seed))
        out_dir = out_path_for(relative_out_path(arm, model_name, seed))
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / SCHEDULE_FILENAME).write_text(json.dumps([list(step) for step in schedule]))

    run(
        model_name=model_name,
        relative_out_path=relative_out_path(arm, model_name, seed),
        train_dataset=TRAIN_DATASET,
        train_dataset_adapter=get_merged_adapter_with_data_mix_from_factory(
            lambda top_k: sampler_for(arm, top_k, seed, schedule)
        ),
        save_schedule=SAVE_SCHEDULE,
        # random_value for the random arm; harmless extra column for the others, which keeps the
        # estimation step identical across arms.
        complexity_estimator_override=SingleTokenEntropyWithRandomEstimator(),
        shuffle=True,
        seed=seed,
    )
