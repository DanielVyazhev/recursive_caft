from core.training.token_weighting import TokenWeightingConfig
from experiments.token_weighting.mmlu.shared import proxy_cache_path, run

# Ablation: w_t = KL_t (reducible part without the H_P normalisation).
run(
    model_name="qwen_3b",
    relative_out_path="./kl/qwen_3b",
    token_weighting=TokenWeightingConfig(mode="kl", proxy_cache_path=proxy_cache_path("qwen_3b")),
)
