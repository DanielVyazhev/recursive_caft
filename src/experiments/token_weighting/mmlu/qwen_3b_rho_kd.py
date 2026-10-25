from core.training.token_weighting import TokenWeightingConfig
from experiments.token_weighting.mmlu.shared import proxy_cache_path, run

# rho-weighted SFT plus the proxy-KD term: does the weighting add anything on top of distilling
# from the proxy (compare with proxykd and rho).
run(
    model_name="qwen_3b",
    relative_out_path="./rho_kd/qwen_3b",
    token_weighting=TokenWeightingConfig(mode="rho", proxy_cache_path=proxy_cache_path("qwen_3b"), kd_lambda=1.0),
    save_schedule=[5, 10, 15, 20],
)
