"""
Hold out a small development set from an MMLU train split for cheap checkpoint evals.

Draws `DEV_SIZE` questions stratified by `base_cluster` (proportional allocation, largest
remainder for the rounding) with a fixed seed and writes two parquets next to the source:

    <name>_dev<DEV_SIZE>.parquet         the held-out rows (evaluated with max_thinking_tokens caps)
    <name>_minus_dev<DEV_SIZE>.parquet   the remaining rows (what every arm trains on)

The test split stays untouched for the final comparison.

Usage:
  python src/preprocessing/make_mmlu_dev_split.py [train_parquet_name]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS_DIR = Path(__file__).resolve().parents[2] / "data" / "out" / "splits" / "random" / "mmlu"
DEFAULT_TRAIN = "train_corrected_answer_deepseek_v4_pro_and_others_head_truncated8192"
DEV_SIZE = 500
SEED = 42
STRATIFY_COLUMN = "base_cluster"


def stratified_indices(groups: pd.Series, size: int, seed: int) -> np.ndarray:
    counts = groups.value_counts()
    quotas = counts * size / counts.sum()
    allocation = np.floor(quotas).astype(int)
    remainder = size - int(allocation.sum())
    # Largest remainder: hand the leftover slots to the groups with the biggest fractional part.
    order = (quotas - allocation).sort_values(ascending=False).index
    for name in order[:remainder]:
        allocation[name] += 1

    rng = np.random.default_rng(seed)
    chosen = []
    for name, n in allocation.items():
        members = np.flatnonzero(groups.to_numpy() == name)
        chosen.extend(rng.choice(members, size=n, replace=False).tolist())
    return np.sort(np.array(chosen, dtype=int))


def main(train_name: str = DEFAULT_TRAIN) -> None:
    src = SPLITS_DIR / f"{train_name}.parquet"
    df = pd.read_parquet(src)
    assert df["question_id"].is_unique, "stratified split assumes one row per question_id"

    dev_idx = stratified_indices(df[STRATIFY_COLUMN], DEV_SIZE, SEED)
    mask = np.zeros(len(df), dtype=bool)
    mask[dev_idx] = True
    dev, rest = df[mask], df[~mask]

    dev_path = SPLITS_DIR / f"{train_name}_dev{DEV_SIZE}.parquet"
    rest_path = SPLITS_DIR / f"{train_name}_minus_dev{DEV_SIZE}.parquet"
    dev.to_parquet(dev_path, index=False)
    rest.to_parquet(rest_path, index=False)

    print(f"{src.name}: {len(df)} rows")
    print(f"  dev  -> {dev_path.name}: {len(dev)} rows")
    print(f"  rest -> {rest_path.name}: {len(rest)} rows")
    print(pd.concat({"all": df[STRATIFY_COLUMN].value_counts(), "dev": dev[STRATIFY_COLUMN].value_counts()}, axis=1))


if __name__ == "__main__":
    main(*sys.argv[1:2])
