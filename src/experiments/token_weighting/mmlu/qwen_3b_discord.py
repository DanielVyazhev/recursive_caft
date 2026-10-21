from core.training.token_weighting import TokenWeightingConfig
from experiments.token_weighting.mmlu.shared import proxy_cache_path, run

# DISCORD baseline: w_t = sigmoid(log p_S(y_t) - log p_P(y_t)), tau = 1, same proxy as reference.
run(
    model_name="qwen_3b",
    relative_out_path="./discord/qwen_3b",
    token_weighting=TokenWeightingConfig(mode="discord", proxy_cache_path=proxy_cache_path("qwen_3b")),
)
