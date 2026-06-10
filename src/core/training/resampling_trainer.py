import json
import os
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

        if not self._logged_samples and len(dataset) > 0:
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
        save_schedule: list[int] | None = None,
        train_dataset: AbstractDatasetAdapter | None = None,
    ) -> None:
        super().__init__()

        self._complexity_evaluation_dataset = complexity_evaluation_dataset
        self._complexity_estimator = complexity_estimator
        self._complexity_estimation_runner_generation_config = complexity_estimation_runner_generation_config
        self._out_path = out_path
        self._max_failed_estimation_fraction = max_failed_estimation_fraction
        self._save_schedule = save_schedule
        # Used to record, per epoch, how many rows the train sampler(s) actually select from the
        # finalized complexity parquet (after the non-positive-score drop). Optional so the callback
        # can be constructed standalone (e.g. in tests) without the count.
        self._train_dataset = train_dataset

    @override
    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        assert state.epoch is not None

        # `epoch` == int(state.epoch) at on_epoch_begin == the number of COMPLETED epochs, i.e. the
        # model state this complexity is measured on — the *previous* training epoch's output. That
        # state is held by the checkpoint whose trainer_state.epoch == epoch, which exists iff
        # `epoch in save_schedule`. (e.g. with schedule [1,3]: dir "1" is model-after-1 == checkpoint
        # epoch=1; dir "2" is model-after-2, which is NOT checkpointed.)
        epoch = int(state.epoch)

        # Resume: only reuse a dumped complexity when the weights it was measured on were checkpointed,
        # so resume reloads exactly that model. A non-checkpointed epoch is re-trained from an earlier
        # checkpoint, so its earlier dump is stale for the new trajectory -> recompute it on the
        # current weights (estimate() still resumes from a partial .tmp within that recompute).
        if self._should_reuse(epoch):
            logger.info(f"Reusing dumped complexity for checkpointed epoch {epoch}; skipping re-estimation.")
            return

        if self._is_checkpoint_epoch(epoch) and self._estimation_complete(epoch):
            logger.info(f"Complexity estimate for checkpointed epoch {epoch} already on disk; running reconcile only.")
        else:
            logger.info(f"Estimating complexity for epoch {epoch}...")

            model: PreTrainedModel = kwargs["model"]

            ComplexityEstimationRunner(
                config=ComplexityEstimationRunnerConfig(
                    out_path=self.out_path_for_epoch(epoch).as_posix(),
                    answer_field_name="estimation_phase_answer",
                    answer_correctness_field_name="estimation_phase_answer_correctness",
                    generate_config=self._complexity_estimation_runner_generation_config,
                ),
                complexity_estimator=self._complexity_estimator,
            ).estimate(dataset_adapter=self._complexity_evaluation_dataset, model=model)

        self._reconcile_estimation(epoch)

    def _reconcile_estimation(self, epoch: int) -> None:
        # After a LoRA epoch the model may emit a thinking token instead of a single answer letter,
        # so those rows fail to measure (entropy_value is NaN). Abort if too many fail this epoch;
        # otherwise reuse the previous epoch's last-known-good entropy for the failed rows. Any
        # residual NaN (epoch 0, or rows that failed every epoch) is harmlessly dropped by the
        # sampler (NaN scores sort to the bottom).
        path = self.out_path_for_epoch(epoch)
        df = pd.read_parquet(path)

        failed = df["entropy_value"].isna()
        failed_count = int(failed.sum())
        total_rows = int(len(df))
        failure_rate = float(failed.mean())
        if failure_rate > self._max_failed_estimation_fraction:
            raise RuntimeError(
                f"Complexity estimation failed for {failure_rate:.1%} of rows at epoch {epoch} "
                f"(> {self._max_failed_estimation_fraction:.0%} allowed); aborting training."
            )

        backfilled = 0
        if epoch > 0 and failed.any():
            prev = pd.read_parquet(self.out_path_for_epoch(epoch - 1))
            prev_entropy = prev.dropna(subset=["entropy_value"]).set_index("question_id")["entropy_value"]
            df.loc[failed, "entropy_value"] = df.loc[failed, "question_id"].map(prev_entropy)
            df.to_parquet(path, index=False)
            backfilled = int(failed_count - df["entropy_value"].isna().sum())
            logger.info(f"Backfilled {backfilled} failed entropy rows from epoch {epoch - 1}.")

        # Count how many rows the train sampler(s) will actually select from this finalized parquet
        # (after the non-positive-score drop). Computed here, after backfill, so `df` matches exactly
        # what SetResamplingPathCallback feeds the samplers for this same epoch.
        by_dataset = self._train_dataset.selected_sample_counts(df) if self._train_dataset is not None else {}
        selected_total = sum(by_dataset.values()) if by_dataset else None

        # Persist how many rows could not be measured this epoch (failed_count, before backfill).
        # This file is also the per-epoch "finalized" marker used by on_epoch_begin to skip
        # re-estimation on resume, so it is written last, only after a successful reconcile.
        self._write_estimation_stats(
            epoch,
            total_rows=total_rows,
            failed_rows=failed_count,
            backfilled=backfilled,
            residual_failed=int(df["entropy_value"].isna().sum()),
            selected_samples=selected_total,
            selected_samples_by_dataset=by_dataset or None,
        )

    def out_path_for_epoch(self, epoch: int) -> Path:
        return (
            self._out_path
            / str(epoch)
            / "complexity_estimation"
            / f"{self._complexity_evaluation_dataset.dataset.dataset_id}.parquet"
        )

    def _stats_path_for_epoch(self, epoch: int) -> Path:
        return self.out_path_for_epoch(epoch).parent / "estimation_stats.json"

    def _is_checkpoint_epoch(self, epoch: int) -> bool:
        # `epoch` == int(state.epoch) at on_epoch_begin == number of completed epochs. The previous
        # training epoch saved its checkpoint under trainer_state.epoch == epoch (when that value is in
        # save_schedule), so this tests whether the weights this complexity was measured on are
        # checkpointed and thus reloaded exactly on resume. No schedule => HF checkpoints every epoch
        # (save_strategy="epoch"), so every epoch qualifies.
        return self._save_schedule is None or epoch in self._save_schedule

    def _should_reuse(self, epoch: int) -> bool:
        # Reuse a finalized dump only when the weights it was measured on were checkpointed (so resume
        # reloads them exactly). Non-checkpointed epochs are re-trained -> the dump is stale -> recompute.
        return self._is_checkpoint_epoch(epoch) and self._stats_path_for_epoch(epoch).exists()

    def _estimation_complete(self, epoch: int) -> bool:
        # estimate() writes out_path only at the very end and deletes its .tmp right after, so a
        # parquet with no sibling .tmp means estimation finished (reconcile may still be pending).
        out_path = self.out_path_for_epoch(epoch)
        return out_path.exists() and not out_path.with_suffix(".tmp").exists()

    def _write_estimation_stats(
        self,
        epoch: int,
        total_rows: int,
        failed_rows: int,
        backfilled: int,
        residual_failed: int,
        selected_samples: int | None = None,
        selected_samples_by_dataset: dict[str, int] | None = None,
    ) -> None:
        stats = {
            "epoch": epoch,
            "total_rows": total_rows,
            "failed_rows": failed_rows,
            "failure_rate": (failed_rows / total_rows) if total_rows else 0.0,
            "backfilled": backfilled,
            "residual_failed": residual_failed,
            # Number of rows the train sampler(s) actually select this epoch after dropping
            # non-positive scores (shrinks over time as the student catches the teacher).
            "selected_samples": selected_samples,
            "selected_samples_by_dataset": selected_samples_by_dataset,
        }
        path = self._stats_path_for_epoch(epoch)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(stats, indent=2))
        os.replace(tmp, path)


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
    def train(self):
        try:
            return super().train()
        finally:
            # Aggregate per-epoch complexity-estimation failure counts. In finally so it also runs
            # after a partial (crashed/killed) run; the body is defensive and never re-raises.
            self._aggregate_estimation_stats()

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
            save_schedule=self.config.save_schedule,
            train_dataset=self.config.train_dataset,
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

    def _aggregate_estimation_stats(self) -> None:
        # Roll the per-epoch estimation_stats.json files up into one summary keyed by epoch. Runs in
        # train()'s finally, so it must never raise (it would mask a training error) — log and move on.
        try:
            base = self._data_path
            if not base.exists():
                return

            dataset_id = self.config.complexity_evaluation_dataset.dataset.dataset_id

            details: list[dict] = []
            for epoch_dir in sorted(
                (d for d in base.iterdir() if d.is_dir() and d.name.isdigit()),
                key=lambda d: int(d.name),
            ):
                epoch = int(epoch_dir.name)
                stats_path = epoch_dir / "complexity_estimation" / "estimation_stats.json"
                if stats_path.exists():
                    entry = json.loads(stats_path.read_text())
                    entry["source"] = "recorded"
                    # Pre-feature stats files lack selected_samples; surface as None rather than KeyError.
                    entry.setdefault("selected_samples", None)
                    entry.setdefault("selected_samples_by_dataset", None)
                else:
                    # A dump produced before this feature: its pre-backfill count is gone (reconcile
                    # overwrote the parquet), so report the post-backfill residual as a best effort.
                    parquet = epoch_dir / "complexity_estimation" / f"{dataset_id}.parquet"
                    if not parquet.exists():
                        continue
                    col = pd.read_parquet(parquet, columns=["entropy_value"])["entropy_value"]
                    residual = int(col.isna().sum())
                    entry = {
                        "epoch": epoch,
                        "total_rows": int(len(col)),
                        "failed_rows": residual,
                        "failure_rate": (residual / len(col)) if len(col) else 0.0,
                        "backfilled": None,
                        "residual_failed": residual,
                        # Can't recompute reliably here: this aggregate only knows the complexity
                        # dataset_id, not the train adapters' top_k(s)/scoring. Leave unknown.
                        "selected_samples": None,
                        "selected_samples_by_dataset": None,
                        "source": "derived",
                    }
                details.append(entry)

            summary = {
                "total_failed_rows": sum(d["failed_rows"] for d in details),
                "failed_rows_by_epoch": {str(d["epoch"]): d["failed_rows"] for d in details},
                "selected_samples_by_epoch": {str(d["epoch"]): d.get("selected_samples") for d in details},
                "details": details,
            }
            out_path = base / "complexity_estimation_failures.json"
            tmp = out_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(summary, indent=2))
            os.replace(tmp, out_path)
            logger.info(f"Wrote complexity-estimation failure summary to {out_path}")
        except Exception as ex:
            logger.warning(f"Failed to aggregate complexity-estimation stats: {ex}")
