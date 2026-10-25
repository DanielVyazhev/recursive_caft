from core.training.token_weighting import TokenWeightingConfig
from experiments.token_weighting.mmlu.shared import (
    TRAIN_DATASET,
    proxy_cache_path,
    run_resampling,
)

# Permuted-rho control in the re-selection regime (same weight spread, random assignment).
run_resampling(
    model_name="qwen_3b",
    relative_out_path="./resample_rho_permuted/qwen_3b",
    token_weighting=TokenWeightingConfig(
        mode="rho",
        proxy_cache_path=proxy_cache_path("qwen_3b", TRAIN_DATASET),
        strict_cache=False,
        permute_within_row=True,
    ),
)
