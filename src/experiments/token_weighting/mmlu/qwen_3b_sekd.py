from core.training.token_weighting import TokenWeightingConfig
from experiments.token_weighting.mmlu.shared import proxy_cache_path, run

# SE-KD / "80/20 rule" baseline: keep the 20% of content tokens per trace with the highest
# student entropy, mean loss over the kept tokens. Needs no proxy statistics (the cache is
# only read for the structural mask bookkeeping and ignored otherwise).
run(
    model_name="qwen_3b",
    relative_out_path="./sekd_top20/qwen_3b",
    token_weighting=TokenWeightingConfig(
        mode="student_entropy", proxy_cache_path=proxy_cache_path("qwen_3b"), select_top_frac=0.2
    ),
    save_schedule=[5, 10, 15, 20],
)
