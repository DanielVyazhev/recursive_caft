from core.training.token_weighting import TokenWeightingConfig
from experiments.token_weighting.mmlu.shared import proxy_cache_path, run

# DISCORD as published: raw sigmoid weights (mean about 0.5, i.e. an implicit halving of the
# step) without the per-document rescaling to mean 1 used by the other arms.
run(
    model_name="qwen_3b",
    relative_out_path="./discord_raw/qwen_3b",
    token_weighting=TokenWeightingConfig(
        mode="discord",
        proxy_cache_path=proxy_cache_path("qwen_3b"),
        discord_tau=4.0,
        normalize_per_pack=False,
    ),
    save_schedule=[5, 10, 15, 20],
)
