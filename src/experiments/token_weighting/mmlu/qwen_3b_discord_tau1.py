from core.training.token_weighting import TokenWeightingConfig
from experiments.token_weighting.mmlu.shared import proxy_cache_path, run

# Implementation check: at the paper default tau = 1 the weights are nearly flat on our traces
# (p10-p90 of 0.29-0.61), so after per-document normalisation this must land on the uniform arm.
run(
    model_name="qwen_3b",
    relative_out_path="./discord_tau1/qwen_3b",
    token_weighting=TokenWeightingConfig(
        mode="discord", proxy_cache_path=proxy_cache_path("qwen_3b"), discord_tau=1.0
    ),
    save_schedule=[5, 10, 15, 20],
)
