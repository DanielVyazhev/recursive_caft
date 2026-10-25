from core.training.token_weighting import TokenWeightingConfig
from experiments.token_weighting.mmlu.shared import (
    TRAIN_DATASET,
    proxy_cache_path,
    run_resampling,
)

# rho weights in the Paper-3 regime: a fresh random 1024-trace draw from the whole pool every
# epoch, 200-epoch cosine schedule stopped after epoch 50, test split. Compare with the three
# seeds of artifacts/distillation_by_metrics/mmlu/random/qwen_3b_head_truncated8192* at epochs
# 10/20/50 and with resample_uniform.
run_resampling(
    model_name="qwen_3b",
    relative_out_path="./resample_rho/qwen_3b",
    token_weighting=TokenWeightingConfig(
        mode="rho",
        proxy_cache_path=proxy_cache_path("qwen_3b", TRAIN_DATASET),
        strict_cache=False,
    ),
)
