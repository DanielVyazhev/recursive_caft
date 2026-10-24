from core.training.token_weighting import TokenWeightingConfig
from experiments.token_weighting.mmlu.shared import proxy_cache_path, run

# "Why not distil from the proxy": plain SFT on the traces plus a forward-KL term towards the
# proxy's distribution (renormalised top-32) on every content token, lambda = 1.
run(
    model_name="qwen_3b",
    relative_out_path="./proxykd/qwen_3b",
    token_weighting=TokenWeightingConfig(mode="uniform", proxy_cache_path=proxy_cache_path("qwen_3b"), kd_lambda=1.0),
    save_schedule=[5, 10, 15, 20],
)
