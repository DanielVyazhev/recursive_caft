from core.training.token_weighting import TokenWeightingConfig
from experiments.token_weighting.mmlu.shared import proxy_cache_path, run

run(
    model_name="qwen_3b",
    relative_out_path="./rho/qwen_3b_seed43",
    token_weighting=TokenWeightingConfig(mode="rho", proxy_cache_path=proxy_cache_path("qwen_3b")),
    save_schedule=[5, 10, 15, 20],
    seed=43,
)
