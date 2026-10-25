from core.training.token_weighting import TokenWeightingConfig
from experiments.token_weighting.mmlu.shared import proxy_cache_path, run

# Pure proxy-KD: no one-hot CE on the trace tokens (ce_scale=0), only the forward-KL term towards
# the proxy's renormalised top-32 on every content token; CE stays on the structural tokens so the
# think/answer format is still learned. Third point on the "how much one-hot" axis next to
# uniform+KD (1.0) and rho+KD (~rho).
run(
    model_name="qwen_3b",
    relative_out_path="./purekd/qwen_3b",
    token_weighting=TokenWeightingConfig(
        mode="uniform", proxy_cache_path=proxy_cache_path("qwen_3b"), kd_lambda=1.0, ce_scale=0.0
    ),
    save_schedule=[5, 10, 15, 20],
)
