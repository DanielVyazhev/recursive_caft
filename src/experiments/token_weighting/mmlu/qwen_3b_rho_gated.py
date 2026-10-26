from core.training.token_weighting import TokenWeightingConfig
from experiments.token_weighting.mmlu.shared import proxy_cache_path, run

# rho with the excess-sharpness gate (EG's one-sided condition): tokens where the student is
# already sharper than the proxy keep their weight only to the extent the proxy is decisive
# (exp(-H_P / 1 nat)). The normalisation factor is capped at 4 so a shrinking support anneals the
# loss instead of blowing the remaining weights up (the EG failure). Watch tw/gate in the logs:
# the mean kept fraction, ~0.85 on the untrained student.
run(
    model_name="qwen_3b",
    relative_out_path="./rho_gated/qwen_3b",
    token_weighting=TokenWeightingConfig(
        mode="rho",
        proxy_cache_path=proxy_cache_path("qwen_3b"),
        eg_gate=True,
        eg_gate_h0=1.0,
        max_norm_scale=4.0,
    ),
    save_schedule=[5, 10, 15, 20],
)
