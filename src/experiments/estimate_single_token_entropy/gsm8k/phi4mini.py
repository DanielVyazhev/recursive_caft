from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

from core.complexity_estimation.complexity_estimation_runner import (
    ComplexityEstimationRunner,
    ComplexityEstimationRunnerConfig,
    ModelGenerateConfig,
)
from core.complexity_estimation.entropy.single_token_entropy_estimator import SingleTokenEntropyEstimator
from core.datasets.gsm8k.gsm8k_single_token_response_dataset import GSM8KSingleTokenResponseDataset, QADatasetConfig
from core.datasets.qa_dataset_adapter import QADatasetAdapter
from core.utils.device import DEVICE

MODEL_NAME = "microsoft/Phi-4-mini-instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True).to(DEVICE)

ComplexityEstimationRunner(
    config=ComplexityEstimationRunnerConfig(
        out_path=str(
            Path(__file__).parent.joinpath("../../../../data/out/single_token_entropy/gsm8k_phi4mini.parquet")
        ),
        answer_field_name="model_answer",
        answer_correctness_field_name="model_answer_correct",
        generate_config=ModelGenerateConfig(max_new_tokens=1),
        save_every=100,
    ),
    complexity_estimator=SingleTokenEntropyEstimator(),
).estimate(
    QADatasetAdapter(
        GSM8KSingleTokenResponseDataset(
            tokenizer,
            QADatasetConfig(
                path=str(Path(__file__).parent.joinpath("../../../../data/source/gsm8k/gsm8k_train.parquet"))
            ),
        )
    ),
    model,
)
