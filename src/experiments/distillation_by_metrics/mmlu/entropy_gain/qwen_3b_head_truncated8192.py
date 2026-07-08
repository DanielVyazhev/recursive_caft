from core.dataset_samplers.entropy_gain_sampler import EntropyGainSampler
from core.dataset_samplers.entropy_ratio_sampler import BaseDatasetSamplerConfig
from core.datasets.causal_dataset_adapter import CausalDatasetAdapter
from core.datasets.merged_dataset_adapter import MergedDatasetAdapter
from core.datasets.mmlu.mmlu_reasoning_response_dataset import MMLUReasoningResponseDataset
from core.datasets.mmlu.mmlu_single_token_response_dataset import MMLUSingleTokenResponseDataset
from core.datasets.qa_dataset import QADatasetConfig
from experiments.distillation_by_metrics.mmlu.shared import run

run(
    model_name="qwen_3b",
    relative_out_path="./entropy_gain/qwen_3b_head_truncated8192",
    train_dataset="train_corrected_answer_deepseek_v4_pro_and_others_head_truncated8192",
    train_dataset_adapter=MergedDatasetAdapter(
        [
            CausalDatasetAdapter(
                dataset=MMLUReasoningResponseDataset(
                    config=QADatasetConfig(
                        path="should be overridden by SetResamplingPathCallback",
                        dataset_id="train_corrected_answer_deepseek_v4_pro_and_others_head_truncated8192",
                    ),
                    # Will be overridden
                    tokenizer=None,  # type: ignore
                ),
                dataset_sampler=EntropyGainSampler(BaseDatasetSamplerConfig(top_k=1024)),
            ),
            # Mix in a smaller single-token-answer set (hardest by gain) so the model keeps
            # answering with a single letter, keeping the per-epoch entropy estimation stable.
            CausalDatasetAdapter(
                dataset=MMLUSingleTokenResponseDataset(
                    config=QADatasetConfig(
                        path="should be overridden by SetResamplingPathCallback",
                        dataset_id="mmlu_single_token_train",
                    ),
                    # Will be overridden
                    tokenizer=None,  # type: ignore
                ),
                dataset_sampler=EntropyGainSampler(BaseDatasetSamplerConfig(top_k=256)),
            ),
        ]
    ),
    save_schedule=[5, 10, 20, 40, 60, 80, 100],
)
