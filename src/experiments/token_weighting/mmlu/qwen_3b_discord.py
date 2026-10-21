from core.training.token_weighting import TokenWeightingConfig
from experiments.token_weighting.mmlu.shared import proxy_cache_path, run

# DISCORD baseline: w_t = sigmoid(tau * (log p_S(y_t) - log p_P(y_t))) with the same proxy as
# reference. tau = 4 is the value DISCORD tunes for Qwen2.5-14B -> Qwen2.5-3B on top of SFT
# (their Table 6); at the paper default tau = 1 the weights on our traces are nearly flat
# (p10-p90 of 0.29-0.61 in the diagnostics), i.e. indistinguishable from uniform.
run(
    model_name="qwen_3b",
    relative_out_path="./discord/qwen_3b",
    token_weighting=TokenWeightingConfig(
        mode="discord", proxy_cache_path=proxy_cache_path("qwen_3b"), discord_tau=4.0
    ),
)
