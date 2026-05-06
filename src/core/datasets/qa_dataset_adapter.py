from core.datasets.base_dataset_adapter import BaseDatasetAdapter, BaseDatasetSampler, TokenizedRow
from core.datasets.qa_dataset import QADataset


class QADatasetAdapter(BaseDatasetAdapter[QADataset]):
    def __init__(
        self,
        dataset: QADataset,
        dataset_sampler: BaseDatasetSampler | None = None,
        add_thinking_start_token: bool = False,
    ):
        super().__init__(dataset, dataset_sampler)
        self.add_thinking_start_token = add_thinking_start_token

    def process_row(self, row: dict) -> TokenizedRow:
        tokenized = self.dataset.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": self.dataset.system_prompt(row)},
                {"role": "user", "content": self.dataset.user_prompt(row)},
            ],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        if self.add_thinking_start_token:
            assert isinstance(self.dataset.tokenizer.thinking_start_token, str)

            thinking_token_ids = self.dataset.tokenizer.encode(
                self.dataset.tokenizer.thinking_start_token,
                add_special_tokens=False,
            )
            input_ids = input_ids + thinking_token_ids
            attention_mask = attention_mask + [1] * len(thinking_token_ids)

        row_id = self.dataset.row_id(row)

        return TokenizedRow(
            **row,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=[],
            row_id=row_id,
        )
