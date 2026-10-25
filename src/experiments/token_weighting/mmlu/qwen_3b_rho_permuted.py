from core.training.token_weighting import TokenWeightingConfig
from experiments.token_weighting.mmlu.shared import proxy_cache_path, run

# Control for rho: the same per-trace weight distribution, randomly reassigned to tokens every
# step. If this matches rho, the gain came from the spread of the weights, not from where they go.
run(
    model_name="qwen_3b",
    relative_out_path="./rho_permuted/qwen_3b",
    token_weighting=TokenWeightingConfig(
        mode="rho", proxy_cache_path=proxy_cache_path("qwen_3b"), permute_within_row=True
    ),
    save_schedule=[5, 10, 15, 20],
)
