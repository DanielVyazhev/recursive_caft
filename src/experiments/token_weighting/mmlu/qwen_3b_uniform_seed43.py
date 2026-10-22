from experiments.token_weighting.mmlu.shared import run

run(
    model_name="qwen_3b",
    relative_out_path="./uniform/qwen_3b_seed43",
    token_weighting=None,
    save_schedule=[5, 10, 15, 20],
    seed=43,
)
