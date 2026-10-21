from core.training.token_weighting import TokenWeightingConfig
from experiments.token_weighting.mmlu.shared import proxy_cache_path, run

# Ablation: token-level entropy gain w_t = max(H_S(t) - H_P(t), 0).
run(
    model_name="qwen_3b",
    relative_out_path="./eg/qwen_3b",
    token_weighting=TokenWeightingConfig(mode="eg", proxy_cache_path=proxy_cache_path("qwen_3b")),
)
