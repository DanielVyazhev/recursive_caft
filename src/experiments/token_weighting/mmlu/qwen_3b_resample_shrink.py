from core.training.token_weighting import TokenWeightingConfig
from experiments.token_weighting.mmlu.shared import (
    TRAIN_DATASET,
    proxy_cache_path,
    run_resampling,
)

# Shrinkage targets in the per-epoch re-draw regime (compare with resample_uniform): on every
# content token the target is (1 - a_t) onehot(y_t) + a_t p~_P with a_t = H_P / (H_P + b_t).
# b_t = max(mean excess surprise of the proxy on this trace (floor 0.05), excess surprise on
# this token): where the proxy is certain the trace token is learned, where the position is
# open and the proxy expected the token the spread is learned, where the proxy did not expect
# the token it is learned from the trace.
run_resampling(
    model_name="qwen_3b",
    relative_out_path="./resample_shrink/qwen_3b",
    token_weighting=TokenWeightingConfig(
        mode="shrink",
        proxy_cache_path=proxy_cache_path("qwen_3b", TRAIN_DATASET),
        strict_cache=False,
        normalize_per_pack=False,
        shrink_alpha_rule="calibrated",
        shrink_gate=True,
        shrink_b_floor=0.05,
        shrink_tau=0.0,
    ),
)
