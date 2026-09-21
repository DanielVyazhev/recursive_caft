"""Exploratory coefficient fits and transition diagnostics on saved random runs.

These are training-pool diagnostics, not validation/test estimates of the value
of a new policy. Run: python src/analysis/selection_report_diagnostics.py
"""
from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/recursive-caft-report-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/selection_report_audit"
MODELS = ["qwen_3b", "phi4_mini", "llama_3b"]
COLS = ["question_id", "entropy_value", "teacher_entropy", "random_value", "estimation_phase_answer_correctness"]


def plot_coefficients(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for model in MODELS:
        d = df[(df.model == model) & (df.target == "student_error_proxy_correct")]
        axes[0].plot(d.epoch, d.chosen_lambda, marker="o", label=model)
        axes[1].plot(d.epoch, d.auc_chosen-d.auc_student, marker="o", label=model)
    axes[0].set_ylabel("Fitted proxy penalty (percentile-rank features)")
    axes[1].set_ylabel("Held-out-row AUROC gain over student entropy")
    axes[1].axhline(0, color="black", linestyle=":", linewidth=1)
    for ax in axes:
        ax.set_xlabel("Random-training scoring snapshot")
        ax.grid(alpha=.2)
    axes[0].legend()
    fig.suptitle("Target: student wrong and Qwen-72B right; training-pool diagnostic")
    fig.savefig(OUT / "adaptive_proxy_diagnostic.png", dpi=180)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    proxy = pd.read_parquet(ROOT / "data/out/single_token_entropy_normalized/mmlu_qwen_72b.parquet", columns=["question_id", "model_answer_correct"])
    proxy = proxy.set_index("question_id").model_answer_correct
    pool_ids = pd.read_parquet(ROOT / "data/out/splits/random/mmlu/train_original.parquet", columns=["question_id"]).question_id.to_numpy()
    shuffled_ids = np.random.default_rng(20260919).permutation(np.sort(pool_ids))
    calibration_ids = set(shuffled_ids[:int(.7*len(shuffled_ids))])
    coefficient_rows, transition_rows, opportunity_rows = [], [], []
    for model in MODELS:
        base = ROOT / "artifacts/distillation_by_metrics/mmlu/random" / (model + "_head_truncated8192")
        paths = sorted(base.glob("resampling_trainer_data/*/complexity_estimation/*.parquet"), key=lambda p: int(p.parents[1].name))
        prev = None
        prev_epoch = None
        for path in paths:
            epoch = int(path.parents[1].name)
            df = pd.read_parquet(path, columns=COLS).set_index("question_id")
            if prev is not None and epoch == prev_epoch + 1:
                common = prev.index.intersection(df.index)
                d = prev.loc[common].copy()
                d["next_correct"] = df.loc[common, "estimation_phase_answer_correctness"]
                d = d.dropna(subset=["estimation_phase_answer_correctness", "next_correct", "entropy_value", "teacher_entropy", "random_value"])
                # Selection cutoffs are determined on the complete previous pool.
                order = prev.random_value.dropna().sort_values(ascending=False).index
                d["arm"] = "unselected"
                d.loc[d.index.isin(order[:1024]), "arm"] = "trace_only"
                d.loc[d.index.isin(order[:256]), "arm"] = "trace_and_direct"
                for score_name, score in [("student",d.entropy_value),("gain",(d.entropy_value-d.teacher_entropy).clip(lower=0))]:
                    # Ties, particularly clipped zeros, use average ranks; tied values
                    # occupy the same bin, avoiding artificial ordering by row id.
                    bins = np.minimum((score.rank(pct=True,method="average") * 5).astype(int),4)
                    for (bin_id,arm,correct),block in d.assign(bin=bins).groupby(["bin","arm","estimation_phase_answer_correctness"]):
                        transition_rows.append(dict(model=model,epoch=prev_epoch,score=score_name,bin=int(bin_id),arm=arm,initial_correct=bool(correct),n=len(block),next_correct=int(block.next_correct.sum())))
            if epoch in [0,10,50,99,149,199]:
                d=df.dropna(subset=COLS[1:]).copy()
                d["proxy_correct"]=d.index.map(proxy)
                d=d.dropna(subset=["proxy_correct"])
                u=d.entropy_value.rank(pct=True).to_numpy()
                v=d.teacher_entropy.rank(pct=True).to_numpy()
                # Fixed pseudo-random split of row IDs across all snapshots/model runs.
                fit_mask=d.index.isin(calibration_ids)
                train=np.flatnonzero(fit_mask); test=np.flatnonzero(~fit_mask)
                yerror=~d.estimation_phase_answer_correctness.astype(bool).to_numpy()
                repair=yerror & d.proxy_correct.astype(bool).to_numpy()
                for label,y in [("student_error",yerror),("student_error_proxy_correct",repair)]:
                    grid=np.arange(0,2.01,.1)
                    values=[roc_auc_score(y[train],(u-a*v)[train]) for a in grid]
                    a=float(grid[int(np.argmax(values))])
                    fullgrid=np.arange(-2,2.01,.1)
                    allvalues=[roc_auc_score(y[train],(u-b*v)[train]) for b in fullgrid]
                    b=float(fullgrid[int(np.argmax(allvalues))])
                    x1=u[:,None]; x2=np.column_stack([u,v,u*v])
                    m1=LogisticRegression(C=1,max_iter=1000).fit(x1[train],y[train])
                    m2=LogisticRegression(C=1,max_iter=1000).fit(x2[train],y[train])
                    p1=m1.predict_proba(x1[test])[:,1]; p2=m2.predict_proba(x2[test])[:,1]
                    coefficient_rows.append(dict(model=model,epoch=epoch,target=label,n=len(d),prevalence=y.mean(),
                        auc_student=roc_auc_score(y[test],u[test]),auc_rank_gap=roc_auc_score(y[test],(u-v)[test]),
                        chosen_lambda=a,auc_chosen=roc_auc_score(y[test],(u-a*v)[test]),
                        unrestricted_lambda=b,auc_unrestricted=roc_auc_score(y[test],(u-b*v)[test]),
                        auc_logistic_student=roc_auc_score(y[test],p1),auc_logistic_both=roc_auc_score(y[test],p2),
                        logloss_student=log_loss(y[test],p1),logloss_both=log_loss(y[test],p2)))
                for name,score in [("student",d.entropy_value),("gain",(d.entropy_value-d.teacher_entropy).clip(lower=0))]:
                    selected=score.sort_values(ascending=False).head(1024).index
                    low=d.entropy_value<=d.entropy_value.quantile(.25)
                    opportunity_rows.append(dict(model=model,epoch=epoch,score=name,pool_error=yerror.mean(),
                        top1024_error=float((~d.loc[selected,"estimation_phase_answer_correctness"].astype(bool)).mean()),
                        pool_proxy_correct_error=repair.mean(),
                        top1024_proxy_correct_error=float((~d.loc[selected,"estimation_phase_answer_correctness"].astype(bool)&d.loc[selected,"proxy_correct"].astype(bool)).mean()),
                        lowest_quartile_error=float((~d.loc[low,"estimation_phase_answer_correctness"].astype(bool)).mean()),
                        lowest_quartile_fraction_of_errors=float((yerror & low.to_numpy()).sum()/yerror.sum())))
            prev=df
            prev_epoch=epoch
        print(model, "diagnostics complete",flush=True)
    coefficients = pd.DataFrame(coefficient_rows)
    coefficients.to_csv(OUT/"coefficient_diagnostics.csv",index=False)
    plot_coefficients(coefficients)
    pd.DataFrame(transition_rows).to_csv(OUT/"random_transition_cells.csv",index=False)
    pd.DataFrame(opportunity_rows).to_csv(OUT/"opportunity_diagnostics.csv",index=False)


if __name__ == "__main__":
    main()
