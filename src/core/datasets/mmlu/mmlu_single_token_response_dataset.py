import ast
import string
from typing import override

from core.datasets.qa_dataset import QADataset


class MMLUSingleTokenResponseDataset(QADataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.option_ids = list(string.ascii_lowercase)

    @override
    def system_prompt(self, row: dict) -> str:
        subject = row["subject"]
        return f"The following are multiple choice questions about {subject}. Choose a correct option letter. Answer with a single symbol. Do not print anything else."

    @override
    def user_prompt(self, row: dict) -> str:
        question = row["question"]
        options = ast.literal_eval(row["options"])

        options_str = "\n".join(
            [f"{option_id}. {answer}".strip() for option_id, answer in zip(self.option_ids, options)]
        )
        user_prompt = f"Question: {question.strip()}\nOptions:\n{options_str}\n"
        return user_prompt

    @override
    def assistant_response(self, row: dict) -> str:
        return str(row["answer"]).strip().lower()

    @override
    def row_id(self, row: dict) -> str:
        return row["question_id"]

    @override
    def verify_assistant_response(self, row: dict, assistant_response: str) -> tuple[str, bool]:
        parsed_answer = assistant_response.strip().lower()

        try:
            return parsed_answer, self.assistant_response(row) == parsed_answer
        except:
            return parsed_answer, False
