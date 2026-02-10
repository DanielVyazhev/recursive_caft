import ast
import string

import pandas as pd

from core.datasets.causal_dataset_adapter import CausalDatasetAdapter


class MMLUSingleTokenResponseDatasetAdapter(CausalDatasetAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.option_ids = list(string.ascii_lowercase)

    def system_prompt(self, row: pd.Series) -> str:
        subject = row["subject"]
        return f"The following are multiple choice questions about {subject}. Choose a correct option letter. Answer with a single symbol. Do not print anything else."

    def user_prompt(self, row: pd.Series) -> str:
        question = row["question"]
        options = ast.literal_eval(row["options"])

        options_str = "\n".join(
            [f"{option_id}. {answer}".strip() for option_id, answer in zip(self.option_ids, options)]
        )
        user_prompt = f"Question: {question.strip()}\nOptions:\n{options_str}\n"
        return user_prompt

    def assistant_response(self, row: pd.Series) -> str:
        return str(row["answer"]).strip().lower()

    def row_id(self, row: pd.Series) -> str:
        return row["question_id"]
