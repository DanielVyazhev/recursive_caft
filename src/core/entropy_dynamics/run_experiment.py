"""
Here we have two modes:
  # Mode A (forced answer) on MMLU 
  python -m core.entropy_dynamics.run_experiment \
      --teacher_reasoning_path data/out/distillation/mmlu_synth_gptoss_b_t0_8.parquet \
      --out_dir artifacts/entropy_dynamics/mmlu_forced \
      --mode forced \
      --dataset_type mmlu \
      --window_size 32

  # Mode B (continuation) on GSM8K
  python -m core.entropy_dynamics.run_experiment \
      --teacher_reasoning_path data/out/reasoning/gsm8k_qwen72b.parquet \
      --out_dir artifacts/entropy_dynamics/gsm8k_continuation \
      --mode continuation \
      --dataset_type gsm8k \
      --window_size 64
"""

import argparse
import sys

from core.entropy_dynamics.analyzer import run_full_analysis
from core.entropy_dynamics.config import EntropyDynamicsConfig, InferenceMode, StudentModelConfig
from core.entropy_dynamics.runner import EntropyDynamicsRunner

DEFAULT_STUDENTS = [
    StudentModelConfig(model_id="Qwen/Qwen2.5-3B", label="qwen_3b"),
    StudentModelConfig(model_id="microsoft/Phi-4-mini-instruct", label="phi4_mini"),
    StudentModelConfig(model_id="meta-llama/Llama-3.2-3B-Instruct", label="llama_3b"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Entropy Dynamics Experiment")
    parser.add_argument("--teacher_reasoning_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="artifacts/entropy_dynamics")
    parser.add_argument("--mode", type=str, default="forced", choices=["forced", "continuation"])
    parser.add_argument("--dataset_type", type=str, default="mmlu", choices=["mmlu", "gpqa", "gsm8k"])
    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--students", type=str, nargs="*", default=None,
                        help="Filter students by label, e.g. --students qwen_3b llama_3b")
    parser.add_argument("--analysis_only", action="store_true",
                        help="Skip inference, just run analysis on existing results.")
    return parser.parse_args()


def main():
    args = parse_args()

    students = DEFAULT_STUDENTS
    if args.students:
        students = [s for s in students if s.label in args.students]
        if not students:
            print(f"No matching students for {args.students}. Available: "
                  f"{[s.label for s in DEFAULT_STUDENTS]}")
            sys.exit(1)

    config = EntropyDynamicsConfig(
        teacher_reasoning_path=args.teacher_reasoning_path,
        out_dir=args.out_dir,
        students=students,
        mode=InferenceMode(args.mode),
        dataset_type=args.dataset_type,
        window_size=args.window_size,
    )

    if not args.analysis_only:
        print(f"Running entropy dynamics experiment:")
        print(f"  Mode: {config.mode.value}")
        print(f"  Dataset: {config.dataset_type}")
        print(f"  Window: {config.window_size} tokens")
        print(f"  Students: {[s.label for s in config.students]}")
        print(f"  Teacher reasoning: {config.teacher_reasoning_path}")
        print()

        runner = EntropyDynamicsRunner(config)
        df = runner.run()
        print(f"\nInference complete. {len(df)} measurements collected.")

    results_path = config.out_path / "entropy_dynamics_results.parquet"
    if results_path.exists():
        print("\nRunning analysis...")
        run_full_analysis(results_path, config.out_path / "analysis")
    else:
        print(f"Results file not found at {results_path}. Run inference first.")


if __name__ == "__main__":
    main()
