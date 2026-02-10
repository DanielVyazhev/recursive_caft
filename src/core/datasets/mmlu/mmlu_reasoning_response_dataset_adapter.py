import pandas as pd

from core.datasets.mmlu.mmlu_single_token_response_dataset_adapter import MMLUSingleTokenResponseDatasetAdapter


class MMLUSingleTokenResponseDatasetAdapter(MMLUSingleTokenResponseDatasetAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def assistant_response(self, row: pd.Series) -> str:
        reasoning_chain = row["thinking"].strip()
        answer = str(row["answer"]).strip().lower()
        return f"{self.tokenizer.thinking_start_token}{reasoning_chain}{self.tokenizer.thinking_end_token}{answer}"
