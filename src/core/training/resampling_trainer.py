from pathlib import Path
from typing import override

import pandas as pd
from torch.utils.data import IterableDataset
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer

from core.complexity_estimation.complexity_estimation_runner import (
    BaseComplexityEstimator,
    ComplexityEstimationRunner,
    ComplexityEstimationRunnerConfig,
    ModelGenerateConfig,
    QADatasetAdapter,
)
from core.datasets.abstract_dataset_adapter import AbstractDatasetAdapter
from core.training.lora_trainer import LoRATrainer, LoRATrainerConfig
from core.utils.logger import logger


class ResamplingDataset(IterableDataset):
    def __init__(self, dataset: AbstractDatasetAdapter, tokenizer: PreTrainedTokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self._logged_samples = False

        self.dataset_path: str | None = None

    def __len__(self) -> int:
        # HF Trainer requires the train dataset to be sized (otherwise it raises and demands
        # max_steps). Each adapter reports its post-sampling size (the sampler's top_k, summed
        # across a MergedDatasetAdapter). This must stay O(1) / I/O-free: __len__ is evaluated in
        # Trainer.__init__ while dataset_path is still None, before the first on_epoch_begin sets it.
        size = self.dataset.sampled_size()
        if size is not None:
            return size
        return len(self.dataset.process_dataset(path_override=self.dataset_path))

    def __iter__(self):
        # Calling process_dataset to re-sample dataset after EstimateComplexityCallback runs
        dataset = self.dataset.process_dataset(path_override=self.dataset_path)

        if not self._logged_samples:
            first_sample = dataset[0]
            logger.info("Dataset samples")
            logger.info("Train")
            logger.info(f"Input: {self.tokenizer.decode(first_sample['input_ids'])}")
            labels = [tok for tok in first_sample["labels"] if tok != -100]
            logger.info(f"Labels: {self.tokenizer.decode(labels)}")
            self._logged_samples = True

        yield from dataset


class EstimateComplexityCallback(TrainerCallback):
    def __init__(
        self,
        complexity_evaluation_dataset: QADatasetAdapter,
        complexity_estimator: BaseComplexityEstimator,
        complexity_estimation_runner_generation_config: ModelGenerateConfig,
        out_path: Path,
        max_failed_estimation_fraction: float = 0.1,
    ) -> None:
        super().__init__()

        self._complexity_evaluation_dataset = complexity_evaluation_dataset
        self._complexity_estimator = complexity_estimator
        self._complexity_estimation_runner_generation_config = complexity_estimation_runner_generation_config
        self._out_path = out_path
        self._max_failed_estimation_fraction = max_failed_estimation_fraction

    @override
    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        assert state.epoch is not None

        logger.info(f"Estimating complexity for epoch {state.epoch}...")

        model: PreTrainedModel = kwargs["model"]

        ComplexityEstimationRunner(
            config=ComplexityEstimationRunnerConfig(
                out_path=self.out_path_for_epoch(int(state.epoch)).as_posix(),
                answer_field_name="estimation_phase_answer",
                answer_correctness_field_name="estimation_phase_answer_correctness",
                generate_config=self._complexity_estimation_runner_generation_config,
            ),
            complexity_estimator=self._complexity_estimator,
        ).estimate(dataset_adapter=self._complexity_evaluation_dataset, model=model)

        self._reconcile_estimation(int(state.epoch))

    def _reconcile_estimation(self, epoch: int) -> None:
        # After a LoRA epoch the model may emit a thinking token instead of a single answer letter,
        # so those rows fail to measure (entropy_value is NaN). Abort if too many fail this epoch;
        # otherwise reuse the previous epoch's last-known-good entropy for the failed rows. Any
        # residual NaN (epoch 0, or rows that failed every epoch) is harmlessly dropped by the
        # sampler (NaN scores sort to the bottom).
        path = self.out_path_for_epoch(epoch)
        df = pd.read_parquet(path)

        failed = df["entropy_value"].isna()
        failure_rate = float(failed.mean())
        if failure_rate > self._max_failed_estimation_fraction:
            raise RuntimeError(
                f"Complexity estimation failed for {failure_rate:.1%} of rows at epoch {epoch} "
                f"(> {self._max_failed_estimation_fraction:.0%} allowed); aborting training."
            )

        if epoch > 0 and failed.any():
            prev = pd.read_parquet(self.out_path_for_epoch(epoch - 1))
            prev_entropy = prev.dropna(subset=["entropy_value"]).set_index("question_id")["entropy_value"]
            df.loc[failed, "entropy_value"] = df.loc[failed, "question_id"].map(prev_entropy)
            df.to_parquet(path, index=False)
            backfilled = int(failed.sum() - df["entropy_value"].isna().sum())
            logger.info(f"Backfilled {backfilled} failed entropy rows from epoch {epoch - 1}.")

    def out_path_for_epoch(self, epoch: int) -> Path:
        return (
            self._out_path
            / str(epoch)
            / "complexity_estimation"
            / f"{self._complexity_evaluation_dataset.dataset.dataset_id}.parquet"
        )


class SetResamplingPathCallback(TrainerCallback):
    def __init__(
        self,
        estimation_complexity_callback: EstimateComplexityCallback,
        resampling_ds: ResamplingDataset,
    ) -> None:
        super().__init__()

        self.estimation_complexity_callback = estimation_complexity_callback
        self.resampling_ds = resampling_ds

    @override
    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        assert state.epoch is not None

        self.resampling_ds.dataset_path = self.estimation_complexity_callback.out_path_for_epoch(
            int(state.epoch)
        ).as_posix()


class ResamplingTrainerConfig(LoRATrainerConfig):
    complexity_evaluation_dataset: QADatasetAdapter
    complexity_estimator: BaseComplexityEstimator
    complexity_estimation_runner_generation_config: ModelGenerateConfig
    # Abort training if more than this fraction of rows fail single-token entropy measurement
    # in an epoch (the model drifting to chain-of-thought output).
    max_failed_estimation_fraction: float = 0.1


class ResamplingTrainer(LoRATrainer[ResamplingTrainerConfig]):
    @override
    def _prepare_data(self):
        train_ds = ResamplingDataset(self.config.train_dataset, self.tokenizer)
        return train_ds

    @override
    def _build_trainer(self, train_ds):
        trainer = super()._build_trainer(train_ds)

        self._estimate_complexity_callback = EstimateComplexityCallback(
            complexity_evaluation_dataset=self.config.complexity_evaluation_dataset,
            complexity_estimator=self.config.complexity_estimator,
            complexity_estimation_runner_generation_config=self.config.complexity_estimation_runner_generation_config,
            out_path=self._data_path,
            max_failed_estimation_fraction=self.config.max_failed_estimation_fraction,
        )
        trainer.add_callback(self._estimate_complexity_callback)
        trainer.add_callback(
            SetResamplingPathCallback(
                estimation_complexity_callback=self._estimate_complexity_callback, resampling_ds=train_ds
            )
        )

        return trainer

    def _path_for_epoch(self, epoch: int) -> Path:
        return self._data_path / str(epoch)

    @property
    def _data_path(self) -> Path:
        return Path(self.config.out_path) / "resampling_trainer_data"
