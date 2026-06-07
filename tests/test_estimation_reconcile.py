"""EstimateComplexityCallback._reconcile_estimation: abort when too many rows fail to
measure single-token entropy in an epoch, otherwise backfill the failures from the
previous epoch's last-known-good values (joined on question_id)."""

from math import nan

import pandas as pd
import pytest

from core.complexity_estimation.complexity_estimation_runner import ModelGenerateConfig
from core.complexity_estimation.entropy.single_token_entropy_estimator import SingleTokenEntropyEstimator
from core.datasets.mmlu.mmlu_single_token_response_dataset import MMLUSingleTokenResponseDataset
from core.datasets.qa_dataset import QADatasetConfig
from core.datasets.qa_dataset_adapter import QADatasetAdapter
from core.training.resampling_trainer import EstimateComplexityCallback


def _callback(thinking_tokenizer, out_path, max_failed=0.1):
    return EstimateComplexityCallback(
        complexity_evaluation_dataset=QADatasetAdapter(
            dataset=MMLUSingleTokenResponseDataset(
                config=QADatasetConfig(path="sentinel", dataset_id="mmlu_teacher_entropy"),
                tokenizer=thinking_tokenizer,
            )
        ),
        complexity_estimator=SingleTokenEntropyEstimator(),
        complexity_estimation_runner_generation_config=ModelGenerateConfig(max_new_tokens=1),
        out_path=out_path,
        max_failed_estimation_fraction=max_failed,
    )


def _write_epoch(callback, epoch, entropy_values):
    path = callback.out_path_for_epoch(epoch)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"question_id": list(range(len(entropy_values))), "entropy_value": entropy_values}
    ).to_parquet(path, index=False)
    return path


def test_reconcile_backfills_from_previous_epoch(thinking_tokenizer, tmp_path):
    cb = _callback(thinking_tokenizer, tmp_path)
    n = 20
    _write_epoch(cb, 0, [0.5] * n)
    epoch1 = [0.9] * n
    epoch1[3] = nan  # one row failed this epoch (5% <= 10%)
    path1 = _write_epoch(cb, 1, epoch1)

    cb._reconcile_estimation(1)

    out = pd.read_parquet(path1)
    assert out["entropy_value"].notna().all()
    assert out.loc[out["question_id"] == 3, "entropy_value"].iloc[0] == 0.5  # from epoch 0
    assert out.loc[out["question_id"] == 0, "entropy_value"].iloc[0] == 0.9  # untouched


def test_reconcile_aborts_above_threshold(thinking_tokenizer, tmp_path):
    cb = _callback(thinking_tokenizer, tmp_path, max_failed=0.1)
    entropy = [0.5] * 10
    entropy[0] = nan
    entropy[1] = nan  # 2/10 = 20% > 10%
    _write_epoch(cb, 0, entropy)

    with pytest.raises(RuntimeError):
        cb._reconcile_estimation(0)


def test_reconcile_epoch0_keeps_residual_nan(thinking_tokenizer, tmp_path):
    # Epoch 0 has no previous epoch to backfill from; residual NaN stays (sampler drops it).
    cb = _callback(thinking_tokenizer, tmp_path)
    entropy = [0.5] * 20
    entropy[5] = nan  # 5% <= 10%, no abort
    path0 = _write_epoch(cb, 0, entropy)

    cb._reconcile_estimation(0)

    out = pd.read_parquet(path0)
    assert out["entropy_value"].isna().sum() == 1
