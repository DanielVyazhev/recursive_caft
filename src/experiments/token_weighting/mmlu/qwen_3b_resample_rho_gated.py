from core.training.token_weighting import TokenWeightingConfig
from experiments.token_weighting.mmlu.shared import (
    TRAIN_DATASET,
    proxy_cache_path,
    run_resampling,
)

# Gated rho (see qwen_3b_rho_gated.py) in the per-epoch re-selection regime.
run_resampling(
    model_name="qwen_3b",
    relative_out_path="./resample_rho_gated/qwen_3b",
    token_weighting=TokenWeightingConfig(
        mode="rho",
        proxy_cache_path=proxy_cache_path("qwen_3b", TRAIN_DATASET),
        strict_cache=False,
        eg_gate=True,
        eg_gate_h0=1.0,
        max_norm_scale=4.0,
    ),
)
