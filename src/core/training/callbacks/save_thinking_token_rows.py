"""Callback that dumps the <think>/</think> rows of embed_tokens (and lm_head if
untied) into each checkpoint dir alongside the adapter. Needed because PEFT's
save_pretrained writes only adapter weights, while these rows live in the base
model and would otherwise be lost.

Loaded back at eval time by MultiCheckpointEvaluator after attaching the adapter.
"""

from pathlib import Path
from typing import override

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from transformers.modeling_utils import PreTrainedModel

ROWS_FILENAME = "thinking_token_rows.pt"


class SaveThinkingTokenRowsCallback(TrainerCallback):
    def __init__(self, new_ids: list[int]):
        assert new_ids, "new_ids must be non-empty"
        self.new_ids = new_ids

    @override
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        model: PreTrainedModel = kwargs["model"]
        # Trainer just wrote checkpoint-<global_step>. Match its naming exactly.
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not checkpoint_dir.exists():
            return

        ids_t = torch.tensor(self.new_ids, dtype=torch.long)
        in_w: torch.Tensor = model.get_input_embeddings().weight.detach()  # type: ignore[assignment]
        in_rows = in_w[ids_t].cpu().clone()

        payload: dict = {
            "new_ids": list(self.new_ids),
            "input_rows": in_rows,
        }

        tied = bool(getattr(model.config, "tie_word_embeddings", False))
        if not tied:
            out_layer = model.get_output_embeddings()
            if out_layer is not None:
                out_w: torch.Tensor = out_layer.weight.detach()  # type: ignore[assignment]
                payload["output_rows"] = out_w[ids_t].cpu().clone()

        torch.save(payload, checkpoint_dir / ROWS_FILENAME)
