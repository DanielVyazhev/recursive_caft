"""Reproduce entropy-only residual coefficients from existing random runs.

Reads only saved entropy columns; does not train models or read answer labels.
Run from any directory: python src/analysis/residual_score_diagnostics.py
"""
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def main():
    rows = []
    for model in ["qwen_3b", "phi4_mini", "llama_3b"]:
        base = (
            ROOT / "artifacts/distillation_by_metrics/mmlu/random"
            / (model + "_head_truncated8192") / "resampling_trainer_data"
        )
        for epoch in [0, 10, 50, 99, 149, 199]:
            paths = sorted((base / str(epoch) / "complexity_estimation").glob("*.parquet"))
            if len(paths) != 1:
                raise ValueError(f"Expected one snapshot for {model}/{epoch}, found {len(paths)}")
            df = pd.read_parquet(paths[0], columns=["entropy_value", "teacher_entropy"])
            df = df.replace([np.inf, -np.inf], np.nan).dropna()
            ranks = df.rank(method="average", pct=True)
            s = ranks.entropy_value.to_numpy()
            p = ranks.teacher_entropy.to_numpy()
            if not len(s):
                raise ValueError(f"No valid measurements for {model}/{epoch}")
            centered_s, centered_p = s - s.mean(), p - p.mean()
            var_p = np.mean(centered_p ** 2)
            covariance = np.mean(centered_s * centered_p)
            beta = max(0.0, covariance / var_p) if var_p > 0 else 0.0
            residual = centered_s - beta * centered_p
            assert abs(residual.mean()) < 1e-12
            if covariance >= 0:
                assert abs(np.mean(residual * centered_p)) < 1e-12
            assert np.isclose(np.var(s), beta ** 2 * var_p + np.var(residual))
            rows.append(dict(model=model, snapshot=epoch, n_valid=len(s), beta=beta))
    result = pd.DataFrame(rows)
    out = ROOT / "artifacts/selection_report_audit/residual_coefficients.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    print(result.pivot(index="model", columns="snapshot", values="beta").round(6).to_string())


if __name__ == "__main__":
    main()
