from core.training.token_weighting import TokenWeightingConfig
from experiments.token_weighting.mmlu.shared import proxy_cache_path, run

run(
    model_name="qwen_3b",
    relative_out_path="./rho_permuted/qwen_3b_seed44",
    token_weighting=TokenWeightingConfig(
        mode="rho", proxy_cache_path=proxy_cache_path("qwen_3b"), permute_within_row=True
    ),
    save_schedule=[5, 10, 15, 20],
    seed=44,
)
