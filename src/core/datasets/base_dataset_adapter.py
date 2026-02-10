from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

import pandas as pd
from datasets import Dataset
from transformers import PreTrainedTokenizer


@dataclass
class TokenizedRow:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    row_id: str


class BaseDatasetAdapter(ABC):
    def __init__(self, df_path: str, tokenizer: PreTrainedTokenizer):
        self.df_path: str = df_path
        self.tokenizer: PreTrainedTokenizer = tokenizer

    @abstractmethod
    def process_row(self, row: pd.Series) -> TokenizedRow: ...

    def _load_df(self) -> pd.DataFrame:
        df = pd.read_parquet(
            self.df_path,
        )
        return df

    def process_dataset(self):
        df = self._load_df()

        dataset = Dataset.from_pandas(df)

        processed_ds = dataset.map(
            lambda row: asdict(self.process_row(row)),
            num_proc=4,
            remove_columns=dataset.column_names,
        )

        return processed_ds
