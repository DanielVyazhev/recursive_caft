"""Read-only audit of saved selection runs; writes report tables and figures.

Run from the repository root: python src/analysis/selection_report_audit.py
No model loading, network access, or training. Historical script defaults are not
assumed to be an exact record of how every saved run was executed.
"""

from pathlib import Path
import ast
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/recursive-caft-report-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/selection_report_audit"
MODELS = ["qwen_3b", "phi4_mini", "llama_3b"]
STRATEGIES = ["random", "entropy_gain", "student_entropy", "entropy_gain_proportional", "student_entropy_proportional"]


def auc(y, score):
    ok = np.isfinite(score) & pd.notna(y)
    return roc_auc_score(np.asarray(y)[ok].astype(bool), np.asarray(score)[ok]) if len(np.unique(np.asarray(y)[ok])) == 2 else np.nan


def audit_split_and_evals(curves):
    train = pd.read_parquet(ROOT / "data/out/splits/random/mmlu/train_original.parquet")
    test = pd.read_parquet(ROOT / "data/out/splits/random/mmlu/test.parquet")
    for frame in [train, test]:
        frame["qnorm"] = frame.question.str.strip()
        frame["optnorm"] = frame.options.map(lambda s: repr(ast.literal_eval(s)) if isinstance(s, str) else repr(s))
    exact = pd.MultiIndex.from_frame(test[["qnorm", "optnorm"]]).isin(pd.MultiIndex.from_frame(train[["qnorm", "optnorm"]]))
    text = test.qnorm.isin(train.qnorm)
    (OUT / "split_audit.json").write_text(json.dumps(dict(train_n=len(train),test_n=len(test),
        id_overlap=len(set(train.question_id)&set(test.question_id)),
        test_question_overlap=int(text.sum()),test_question_options_overlap=int(exact.sum()),
        train_internal_question_options_duplicates=int(train.duplicated(["qnorm","optnorm"]).sum()),
        test_internal_question_options_duplicates=int(test.duplicated(["qnorm","optnorm"]).sum())),indent=2))
    test_ids = set(test.question_id.astype(str))
    exact_ids = set(test.loc[exact,"question_id"].astype(str))
    text_ids = set(test.loc[text,"question_id"].astype(str))
    records = []
    mismatches = []
    for row in curves.itertuples():
        if str(row.cap) != "4096" or "head_truncated8192" not in row.run:
            continue
        path = ROOT / Path(row.source).parent / row.checkpoint / "evals/mmlu_random_test_cap4096/responses.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path,columns=["row_id","is_correct"])
        df["row_id"] = df.row_id.astype(str)
        assert df.row_id.is_unique and set(df.row_id)==test_ids, str(path)
        if abs(df.is_correct.mean()-row.accuracy)>=1e-10:
            mismatches.append(dict(source=str(path.relative_to(ROOT)),summary_accuracy=row.accuracy,
                                   response_accuracy=df.is_correct.mean()))
        for name,excluded in [("all",set()),("exclude_question_options_overlap",exact_ids),("exclude_question_overlap",text_ids)]:
            subset=df[~df.row_id.isin(excluded)]
            records.append(dict(strategy=row.strategy,run=row.run,epoch=row.epoch,group=row.group,
                                split=name,n=len(subset),accuracy=subset.is_correct.mean()))
    pd.DataFrame(records).to_csv(OUT/"deduplication_sensitivity.csv",index=False)
    pd.DataFrame(mismatches,columns=["source","summary_accuracy","response_accuracy"]).to_csv(OUT/"summary_response_mismatches.csv",index=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for group in ["distillation_by_metrics", "distillation_on_synthetic_traces"]:
        base = ROOT / "artifacts" / group / "mmlu"
        for path in sorted(base.glob("*/*/summary_reasoning_evals*.json")):
            for key, records in json.loads(path.read_text()).items():
                for record in records:
                    rows.append(dict(group=group, strategy=path.parents[1].name, run=path.parent.name,
                                     cap=key.rsplit("cap", 1)[-1] if "cap" in key else "8192",
                                     source=str(path.relative_to(ROOT)), **record))
    curves = pd.DataFrame(rows)
    curves.to_csv(OUT / "evaluation_curves.csv", index=False)
    audit_split_and_evals(curves)
    keys = ["group", "strategy", "run", "cap"]
    peaks = curves.loc[curves.groupby(keys).accuracy.idxmax()]
    peaks.to_csv(OUT / "descriptive_peaks.csv", index=False)
    curves.sort_values("epoch").groupby(keys).tail(1).to_csv(OUT / "final_evaluations.csv", index=False)

    dynamics = []
    category_rows = []
    for strategy in STRATEGIES:
        for model in MODELS:
            run = model + "_head_truncated8192"
            if strategy == "student_entropy_proportional":
                run += "_shuffle"
            run_dir = ROOT / "artifacts/distillation_by_metrics/mmlu" / strategy / run
            paths = sorted(run_dir.glob("resampling_trainer_data/*/complexity_estimation/*.parquet"), key=lambda p: int(p.parents[1].name))
            counts = {}
            prev = set()
            for path in paths:
                epoch = int(path.parents[1].name)
                cols = [c for c in ["question_id", "category", "entropy_value", "teacher_entropy", "random_value", "estimation_phase_answer_correctness"] if c in pq.read_schema(path).names]
                df = pd.read_parquet(path, columns=cols)
                h = df.entropy_value
                hp = df.teacher_entropy
                signed = h - hp
                gain = signed.clip(lower=0)
                if strategy == "random":
                    score = df.random_value
                elif strategy == "student_entropy":
                    score = h
                elif strategy == "entropy_gain":
                    score = gain
                else:
                    u = df.random_value
                    # Only saved race keys can be reconstructed. Missing u is not
                    # redrawn here, so coverage is explicitly a reconstruction.
                    score = (h if strategy == "student_entropy_proportional" else gain) / -np.log(u.clip(lower=1e-12))
                    score = score.where((u > 0) & (u < 1))
                selected = df.assign(score=score).loc[score > 0].sort_values("score", ascending=False).head(1024)
                ids = set(selected.question_id)
                for q in ids:
                    counts[q] = counts.get(q, 0) + 1
                v = np.array(list(counts.values()))
                y = df.estimation_phase_answer_correctness
                sel_y = selected.estimation_phase_answer_correctness
                r = dict(strategy=strategy, model=model, run=run, epoch=epoch,
                         n=len(df), n_valid=h.notna().sum(), accuracy=y.mean(),
                         auc_student=auc(y, -h), auc_gain=auc(y, -gain),
                         auc_signed_gain=auc(y, -signed), auc_proxy=auc(y, -hp),
                         h_mean=h.mean(), hp_mean=hp.mean(), h_std=h.std(), hp_std=hp.std(),
                         nonpositive_fraction=(signed <= 0).mean(), eligible=(score > 0).sum(),
                         selected=len(ids), selected_accuracy=sel_y.mean(), unique=len(counts),
                         coverage=len(counts)/len(df),
                         retention=len(ids & prev)/len(ids) if prev and ids else np.nan,
                         jaccard=len(ids & prev)/len(ids | prev) if prev else np.nan,
                         exposure_ess=float(v.sum()**2/(v*v).sum()) if len(v) else np.nan,
                         fraction_exposures_top10pct=float(np.sort(v)[-max(1,int(np.ceil(.1*len(df)))):].sum()/v.sum()) if len(v) else np.nan,
                         reconstruction_missing_random=int(df.random_value.isna().sum()) if "random_value" in df else 0)
                dynamics.append(r)
                if epoch in [0, 9, 49, 99, 199]:
                    for category, block in df.groupby("category"):
                        selected_block = selected[selected.category == category]
                        category_rows.append(dict(strategy=strategy, model=model, epoch=epoch, category=category,
                                                  pool_n=len(block), selected_n=len(selected_block),
                                                  pool_accuracy=block.estimation_phase_answer_correctness.mean(),
                                                  selected_accuracy=selected_block.estimation_phase_answer_correctness.mean()))
                prev = ids
            print(f"{strategy}/{model}: {len(paths)} scoring snapshots", flush=True)
    dyn = pd.DataFrame(dynamics)
    dyn.to_csv(OUT / "selection_dynamics.csv", index=False)
    pd.DataFrame(category_rows).to_csv(OUT / "category_composition.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    colors = dict(zip(STRATEGIES, ["#333333", "#c6503f", "#316aab", "#d79b30", "#439378"]))
    for j, model in enumerate(MODELS):
        for strategy in STRATEGIES:
            d = dyn[(dyn.model == model) & (dyn.strategy == strategy)]
            axes[0,j].plot(d.epoch+1,d.coverage,label=strategy,color=colors[strategy])
            axes[1,j].plot(d.epoch,d.auc_student,color=colors[strategy])
        axes[0,j].set_title(model)
        axes[0,j].set_ylim(0,1.02)
        axes[1,j].set_ylim(.4,1)
        axes[1,j].set_xlabel("Subset round / scoring snapshot")
        axes[0,j].grid(alpha=.2)
        axes[1,j].grid(alpha=.2)
    axes[0,0].set_ylabel("Cumulative reconstructed trace coverage")
    axes[1,0].set_ylabel("Training-pool direct-answer AUROC")
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc="outside lower center",ncol=3,fontsize=9)
    fig.savefig(OUT / "coverage_and_auc.png",dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for j,model in enumerate(MODELS):
        for strategy in STRATEGIES:
            run=model+"_head_truncated8192"+("_shuffle" if strategy=="student_entropy_proportional" else "")
            d=curves[(curves.group=="distillation_by_metrics")&(curves.strategy==strategy)&(curves.run==run)&(curves.cap=="4096")].sort_values("epoch")
            axes[j].plot(d.epoch*1024/9626,d.accuracy,marker="o",ms=3,label=strategy,color=colors[strategy])
        d=curves[(curves.group=="distillation_on_synthetic_traces")&(curves.strategy=="corrected_answer")&(curves.run==model+"_head_truncated8192")&(curves.cap=="4096")].sort_values("epoch")
        axes[j].plot(d.epoch,d.accuracy,marker="s",ms=3,color="#888888",linestyle="--",label="full corrected traces")
        axes[j].set_title(model)
        axes[j].set_xlabel("Nominal trace presentations / 9,626")
        axes[j].grid(alpha=.2)
    axes[0].set_ylabel("Test accuracy (cap 4,096)")
    fig.legend(*axes[0].get_legend_handles_labels(),loc="outside lower center",ncol=3,fontsize=9)
    fig.savefig(OUT / "accuracy_vs_nominal_trace_exposure.png",dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
