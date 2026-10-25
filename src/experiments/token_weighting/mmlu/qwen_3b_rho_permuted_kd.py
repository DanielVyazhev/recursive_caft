from core.training.token_weighting import TokenWeightingConfig
from experiments.token_weighting.mmlu.shared import proxy_cache_path, run

# Discriminating control for rho+KD: the same weight spread, randomly reassigned every step, plus
# the same KD term. If this collapses like uniform+KD, the rho+KD gain comes from where the weights
# go; if it matches rho+KD, it comes from the spread alone.
run(
    model_name="qwen_3b",
    relative_out_path="./rho_permuted_kd/qwen_3b",
    token_weighting=TokenWeightingConfig(
        mode="rho", proxy_cache_path=proxy_cache_path("qwen_3b"), permute_within_row=True, kd_lambda=1.0
    ),
    save_schedule=[5, 10, 15, 20],
)
