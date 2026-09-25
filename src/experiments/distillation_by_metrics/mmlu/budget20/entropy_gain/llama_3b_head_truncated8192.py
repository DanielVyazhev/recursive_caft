from experiments.distillation_by_metrics.mmlu.budget20.shared import run_arm

run_arm("entropy_gain", model_name="llama_3b", seed=42)
