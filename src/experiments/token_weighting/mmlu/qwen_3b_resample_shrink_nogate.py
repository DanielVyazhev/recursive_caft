from core.training.token_weighting import TokenWeightingConfig
from experiments.token_weighting.mmlu.shared import (
    TRAIN_DATASET,
    proxy_cache_path,
    run_resampling,
)

# Control for resample_shrink: one global bias for every token, b_t = 0.131 (the proxy's mean
# excess surprise over the diagnostic set), no per-trace and no per-token gate. a_t then only
# depends on H_P, so the tokens the proxy did not see coming (7% of tokens, 35% of the SFT loss)
# are pulled towards the proxy's distribution as well -- the mechanism behind the proxy-KD drop.
run_resampling(
    model_name="qwen_3b",
    relative_out_path="./resample_shrink_nogate/qwen_3b",
    token_weighting=TokenWeightingConfig(
        mode="shrink",
        proxy_cache_path=proxy_cache_path("qwen_3b", TRAIN_DATASET),
        strict_cache=False,
        normalize_per_pack=False,
        shrink_alpha_rule="calibrated",
        shrink_gate=False,
        shrink_b=0.131,
    ),
)
