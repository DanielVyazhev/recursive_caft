import os
from pathlib import Path

import torch
from pydantic import BaseModel
from pydraconf import PydraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from transformers.generation.utils import GenerateDecoderOnlyOutput

from core.complexity_estimation.complexity_estimator import BaseComplexityEstimator
from core.datasets.abstract_dataset_adapter import TokenizedRow
from core.datasets.qa_dataset_adapter import QADatasetAdapter


class ModelGenerateConfig(BaseModel):
    max_new_tokens: int
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    do_sample: bool = False
    num_beams: int = 1


class ComplexityEstimationRunnerConfig(PydraConfig):
    out_path: str
    model_id: str
    answer_field_name: str
    answer_correctness_field_name: str
    generate_config: ModelGenerateConfig
    save_every: int = 100


class ComplexityEstimationRunner:
    def __init__(self, config: ComplexityEstimationRunnerConfig, complexity_estimator: BaseComplexityEstimator):
        self.config = config
        self.complexity_estimator = complexity_estimator

    def estimate(self, dataset_adapter: QADatasetAdapter, device: torch.device):
        invalid_answers = 0
        processed_rows = 0

        if os.path.exists(self.config.out_path) and os.path.exists(self.tmp_path()):
            print(f"Output path {self.config.out_path} already exists. Resuming from temporary file.")
            ds = dataset_adapter.process_dataset(path_override=str(self.tmp_path()))
        else:
            print(f"No temporary file found. Resuming from output file {self.config.out_path}.")
            ds = dataset_adapter.process_dataset()

            ds = ds.add_column(self.config.answer_field_name, [None] * len(ds))
            ds = ds.add_column(self.config.answer_correctness_field_name, [None] * len(ds))

        self.complexity_estimator.prepare_dataset(ds)

        model = AutoModelForCausalLM.from_pretrained(self.config.model_id).to(device)

        for index, row_dict in tqdm(enumerate(ds), total=len(ds)):
            processed_rows += 1

            if row_dict[self.config.answer_field_name] is not None:
                continue

            row = TokenizedRow(**row_dict)

            input_ids = torch.tensor(row.input_ids).unsqueeze(0).to(device)
            attention_mask = torch.tensor(row.attention_mask).unsqueeze(0).to(device)

            outputs: GenerateDecoderOnlyOutput = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict_in_generate=True,
                output_scores=True,
                **self.config.generate_config.model_dump(),
                pad_token_id=dataset_adapter.dataset.tokenizer.pad_token_id,
            )

            input_length = len(row.input_ids)
            response_raw = outputs.sequences[0, input_length:]

            response = dataset_adapter.dataset.tokenizer.decode(response_raw, skip_special_tokens=True).strip()

            try:
                parsed_answer, answer_correctness = dataset_adapter.dataset.verify_assistant_response(
                    ds[index], response
                )

                ds[index][self.config.answer_field_name] = parsed_answer
                ds[index][self.config.answer_correctness_field_name] = answer_correctness

                self.complexity_estimator.estimate_row(ds[index], row, outputs, parsed_answer, answer_correctness)
            except Exception:
                invalid_answers += 1

            if processed_rows % self.config.save_every == 0:
                ds.save_to_disk(self.tmp_path())

                print(
                    f"Processing dataset {self.config.out_path}... Processed: {processed_rows}/{len(ds)}. Invalid answers: {invalid_answers}"
                )

        ds = ds.remove_columns(["input_ids", "attention_mask", "labels"])
        ds.to_parquet(self.config.out_path)

        print(f"Processed dataset {self.config.out_path}. Total entries: {len(ds)}. Invalid answers: {invalid_answers}")

    def tmp_path(self) -> Path:
        return Path(self.config.out_path).with_suffix(".tmp")
