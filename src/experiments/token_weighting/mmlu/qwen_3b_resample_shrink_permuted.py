from core.training.token_weighting import TokenWeightingConfig
from experiments.token_weighting.mmlu.shared import (
    TRAIN_DATASET,
    proxy_cache_path,
    run_resampling,
)

# Control for resample_shrink: the same a_t values, shuffled among the content tokens of every
# packed row at every step. Keeps the amount of KD, destroys its assignment to positions; a gain
# that survives this comes from mixing in the proxy as such, not from the calibration.
run_resampling(
    model_name="qwen_3b",
    relative_out_path="./resample_shrink_permuted/qwen_3b",
    token_weighting=TokenWeightingConfig(
        mode="shrink",
        proxy_cache_path=proxy_cache_path("qwen_3b", TRAIN_DATASET),
        strict_cache=False,
        normalize_per_pack=False,
        shrink_alpha_rule="calibrated",
        shrink_gate=True,
        shrink_b_floor=0.05,
        permute_within_row=True,
    ),
)
