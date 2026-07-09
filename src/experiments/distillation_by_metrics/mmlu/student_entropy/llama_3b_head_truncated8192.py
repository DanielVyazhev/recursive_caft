from core.dataset_samplers.student_entropy_sampler import StudentEntropySampler
from experiments.distillation_by_metrics.mmlu.shared import get_merged_adapter_with_data_mix, run

run(
    model_name="llama_3b",
    relative_out_path="./student_entropy/llama_3b_head_truncated8192",
    train_dataset="train_corrected_answer_deepseek_v4_pro_and_others_head_truncated8192",
    train_dataset_adapter=get_merged_adapter_with_data_mix(StudentEntropySampler),
    save_schedule=[5, 10, 20, 40, 60, 80, 100],
)
