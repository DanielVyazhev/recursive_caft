from experiments.token_weighting.mmlu.shared import run

# Full SFT under the pilot protocol (train minus dev500, 20 epochs, dev500 evals at cap 2048).
# The stock loss: token_weighting=None leaves the trainer exactly as in
# distillation_on_synthetic_traces, only the data split and the eval set differ.
run(
    model_name="qwen_3b",
    relative_out_path="./uniform/qwen_3b",
    token_weighting=None,
)
