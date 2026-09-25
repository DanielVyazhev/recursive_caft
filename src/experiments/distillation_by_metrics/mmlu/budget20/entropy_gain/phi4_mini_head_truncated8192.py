from experiments.distillation_by_metrics.mmlu.budget20.shared import run_arm

run_arm("entropy_gain", model_name="phi4_mini", seed=42)
