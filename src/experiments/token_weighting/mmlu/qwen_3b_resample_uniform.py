from experiments.token_weighting.mmlu.shared import run_resampling

# Plain SFT in the same re-selection regime: the paired baseline for resample_rho, and a check
# that this code path reproduces distillation_by_metrics/mmlu/random/qwen_3b_head_truncated8192
# (0.465 / 0.521 / 0.554 at epochs 10 / 20 / 50, seed 42).
run_resampling(
    model_name="qwen_3b",
    relative_out_path="./resample_uniform/qwen_3b",
    token_weighting=None,
)
