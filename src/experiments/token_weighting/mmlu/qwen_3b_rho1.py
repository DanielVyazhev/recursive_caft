from core.training.token_weighting import TokenWeightingConfig
from experiments.token_weighting.mmlu.shared import proxy_cache_path, run

# Rho-1 / Token Cleaning baseline: keep the 60% of content tokens per trace with the highest
# excess loss l_S - l_P (proxy as the reference model), mean loss over the kept tokens.
run(
    model_name="qwen_3b",
    relative_out_path="./rho1_top60/qwen_3b",
    token_weighting=TokenWeightingConfig(
        mode="excess_loss", proxy_cache_path=proxy_cache_path("qwen_3b"), select_top_frac=0.6
    ),
    save_schedule=[5, 10, 15, 20],
)
