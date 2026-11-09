from core.training.token_weighting import TokenWeightingConfig
from experiments.token_weighting.mmlu.shared import (
    TRAIN_DATASET,
    proxy_cache_path,
    run_resampling,
)

# Control for resample_shrink: a_t = 0.6 on every content token (the mean a_t of the calibrated
# rule over the diagnostic set), i.e. a fixed 40/60 CE/KD mix. Separates "some KD from the proxy
# helps" from "KD where the proxy is trustworthy helps".
run_resampling(
    model_name="qwen_3b",
    relative_out_path="./resample_shrink_const/qwen_3b",
    token_weighting=TokenWeightingConfig(
        mode="shrink",
        proxy_cache_path=proxy_cache_path("qwen_3b", TRAIN_DATASET),
        strict_cache=False,
        normalize_per_pack=False,
        shrink_alpha_rule="const",
        shrink_alpha_const=0.6,
    ),
)
