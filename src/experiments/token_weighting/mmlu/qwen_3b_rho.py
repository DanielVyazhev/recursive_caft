from core.training.token_weighting import TokenWeightingConfig
from experiments.token_weighting.mmlu.shared import proxy_cache_path, run

# Reducible-fraction weights rho_t = KL_t / (H_P(t) + KL_t).
run(
    model_name="qwen_3b",
    relative_out_path="./rho/qwen_3b",
    token_weighting=TokenWeightingConfig(mode="rho", proxy_cache_path=proxy_cache_path("qwen_3b")),
)
