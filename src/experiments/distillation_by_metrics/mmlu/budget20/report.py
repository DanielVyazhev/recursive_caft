"""Budget-20 report: test accuracy vs unique teacher traces per arm, and paired arm differences.

Usage: uv run src/experiments/distillation_by_metrics/mmlu/budget20/report.py [seed]

Arms that have not finished yet are skipped. Differences are paired over the same test questions
with a question-level bootstrap; this covers test-set noise only, not seed-to-seed training noise.
"""

import json
import sys

import numpy as np
import pandas as pd

from experiments.distillation_by_metrics.mmlu.budget20.shared import (
    ARMS,
    SAVE_SCHEDULE,
    load_schedule,
    relative_out_path,
    trace_selections,
)
from experiments.distillation_by_metrics.mmlu.shared import out_path_for

MODELS = ("qwen_3b", "llama_3b", "phi4_mini")
CAPS = (2048, 4096)
COMPARISONS = (("entropy_gain", "matched_random"), ("entropy_gain", "random"), ("matched_random", "random"))
BOOTSTRAP_SAMPLES = 2000


def unique_traces(arm: str, model_name: str, seed: int) -> dict[int, int]:
    schedule = load_schedule(model_name, seed) if arm == "matched_random" else None
    selections = trace_selections(arm, model_name, seed, schedule)
    return {epoch: len({q for selection in selections[:epoch] for q in selection}) for epoch in SAVE_SCHEDULE}


def correctness(arm: str, model_name: str, seed: int, cap: int) -> dict[int, pd.Series]:
    """Per-question correctness indexed by row_id, per checkpoint epoch."""
    out_dir = out_path_for(relative_out_path(arm, model_name, seed))
    summary = json.loads((out_dir / f"summary_reasoning_evals_cap{cap}.json").read_text())
    by_epoch = {}
    for record in next(iter(summary.values())):
        responses = out_dir / record["checkpoint"] / "evals" / f"mmlu_random_test_cap{cap}" / "responses.parquet"
        df = pd.read_parquet(responses, columns=["row_id", "is_correct"])
        by_epoch[int(record["epoch"])] = df.set_index("row_id")["is_correct"].astype(float).sort_index()
    return by_epoch


def paired_difference(a: pd.Series, b: pd.Series, rng: np.random.Generator) -> tuple[float, float, float]:
    assert a.index.equals(b.index), "Arms were evaluated on different test questions"
    diff = (a - b).to_numpy()
    means = diff[rng.integers(0, len(diff), size=(BOOTSTRAP_SAMPLES, len(diff)))].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return 100 * diff.mean(), 100 * low, 100 * high


def main(seed: int) -> None:
    rng = np.random.default_rng(0)
    for model_name in MODELS:
        print(f"\n## {model_name} (seed {seed})\n")
        traces, results = {}, {}
        for arm in ARMS:
            try:
                traces[arm] = unique_traces(arm, model_name, seed)
                results[arm] = {cap: correctness(arm, model_name, seed, cap) for cap in CAPS}
            except FileNotFoundError as error:
                print(f"- skipping {arm}: {error}")

        for cap in CAPS:
            print(f"\ncap {cap}\n")
            print("| epoch | " + " | ".join(f"{arm} acc (traces)" for arm in results) + " |")
            print("|---" * (len(results) + 1) + "|")
            for epoch in SAVE_SCHEDULE:
                cells = [
                    f"{100 * results[arm][cap][epoch].mean():.1f} ({traces[arm][epoch]})"
                    if epoch in results[arm][cap]
                    else "—"
                    for arm in results
                ]
                print(f"| {epoch} | " + " | ".join(cells) + " |")

            for a, b in COMPARISONS:
                if a not in results or b not in results:
                    continue
                cells = []
                for epoch in SAVE_SCHEDULE:
                    if epoch in results[a][cap] and epoch in results[b][cap]:
                        mean, low, high = paired_difference(results[a][cap][epoch], results[b][cap][epoch], rng)
                        cells.append(f"ep{epoch} {mean:+.1f} [{low:+.1f}, {high:+.1f}]")
                print(f"\n{a} − {b}: " + "; ".join(cells))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 42)
