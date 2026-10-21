"""Collators that attach per-token proxy statistics to a batch.

Both emit, next to the usual model inputs, `tw_*` tensors aligned to the LABEL positions of
the batch: index `s` holds the proxy's prediction for the token at `s` (produced one position
earlier). The trainer shifts them together with the labels.

    tw_entropy    float32 [B, T]     H_P(s)
    tw_logp       float32 [B, T]     log p_P(input_ids[s])
    tw_topk_ids   int64   [B, T, K]  proxy top-K ids at s
    tw_topk_logp  float32 [B, T, K]  their proxy log-probs
    tw_valid      bool    [B, T]     proxy statistics present (labeled position of a cached row)
    tw_structural bool    [B, T]     <think>, </think> and every labeled token after </think>

`ProxyAwarePaddedCollator` is the non-packed case (one document per row, padded on the
tokenizer's side); `ProxyAwarePackedCollator` reads the block layout `pack_dataset(...,
with_members=True)` recorded. Rows without a cache entry get no valid positions, which the
loss treats as weight 1 (or raise, with `strict`).
"""

from dataclasses import dataclass

import numpy as np
import torch
from transformers import DataCollatorForTokenClassification, PreTrainedTokenizer

from core.datasets.sequence_packing import IGNORE_LABEL, row_key
from core.training.packed_data_collator import PackedSequenceCollator
from core.training.token_weighting.proxy_cache import ProxyCache


def structural_mask(input_ids: torch.Tensor, labels: torch.Tensor, thinking_token_ids: tuple[int, int]) -> torch.Tensor:
    """Labeled positions of one document that carry trace structure rather than content:
    the <think> token, the </think> token and everything after it (answer, end-of-turn)."""
    think_start, think_end = thinking_token_ids
    labeled = labels != IGNORE_LABEL
    is_end = (input_ids == think_end) & labeled
    at_or_after_end = torch.cumsum(is_end.to(torch.int64), dim=0) > 0
    return ((input_ids == think_start) | at_or_after_end) & labeled


@dataclass
class _ProxyTensors:
    entropy: torch.Tensor
    logp: torch.Tensor
    topk_ids: torch.Tensor
    topk_logp: torch.Tensor
    valid: torch.Tensor
    structural: torch.Tensor

    @classmethod
    def empty(cls, bsz: int, seq: int, top_k: int) -> "_ProxyTensors":
        return cls(
            entropy=torch.zeros(bsz, seq, dtype=torch.float32),
            logp=torch.zeros(bsz, seq, dtype=torch.float32),
            topk_ids=torch.zeros(bsz, seq, top_k, dtype=torch.int64),
            topk_logp=torch.zeros(bsz, seq, top_k, dtype=torch.float32),
            valid=torch.zeros(bsz, seq, dtype=torch.bool),
            structural=torch.zeros(bsz, seq, dtype=torch.bool),
        )

    def as_batch(self) -> dict[str, torch.Tensor]:
        return {
            "tw_entropy": self.entropy,
            "tw_logp": self.logp,
            "tw_topk_ids": self.topk_ids,
            "tw_topk_logp": self.topk_logp,
            "tw_valid": self.valid,
            "tw_structural": self.structural,
        }


class _ProxyAttacher:
    """Shared document-level fill used by both collators."""

    def __init__(self, cache: ProxyCache | None, thinking_token_ids: tuple[int, int], strict: bool):
        self.cache = cache
        self.thinking_token_ids = thinking_token_ids
        self.strict = strict
        self.top_k = cache.top_k if cache is not None else 1
        self.missing_members = 0
        self.seen_members = 0

    def fill(self, out: _ProxyTensors, b: int, start: int, doc_ids: torch.Tensor, doc_labels: torch.Tensor, key: int) -> None:
        length = doc_ids.numel()
        out.structural[b, start : start + length] = structural_mask(doc_ids, doc_labels, self.thinking_token_ids)
        self.seen_members += 1
        entry = self.cache.lookup(key) if self.cache is not None else None
        if entry is None:
            self.missing_members += 1
            if self.strict and self.cache is not None:
                raise KeyError(f"document with key {key} has no proxy cache entry")
            return
        labeled = (doc_labels != IGNORE_LABEL).nonzero(as_tuple=True)[0]
        _, n = entry
        assert n == labeled.numel(), f"proxy cache entry has {n} tokens but the document has {labeled.numel()} labeled tokens"
        sl = self.cache.slice(key)
        pos = labeled + start
        out.entropy[b, pos] = torch.as_tensor(np.array(sl["entropy"], dtype=np.float32))
        out.logp[b, pos] = torch.as_tensor(np.array(sl["logp"], dtype=np.float32))
        out.topk_ids[b, pos] = torch.as_tensor(np.array(sl["topk_ids"], dtype=np.int64))
        out.topk_logp[b, pos] = torch.as_tensor(np.array(sl["topk_logp"], dtype=np.float32))
        out.valid[b, pos] = True


class ProxyAwarePaddedCollator:
    """One document per batch row: DataCollatorForTokenClassification padding plus `tw_*`.
    Keys are hashed from the unpadded input_ids, so the dataset needs no extra columns."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        cache: ProxyCache | None,
        thinking_token_ids: tuple[int, int],
        strict: bool = True,
        pad_to_multiple_of: int | None = 8,
    ):
        self.tokenizer = tokenizer
        self.base = DataCollatorForTokenClassification(
            tokenizer=tokenizer, padding=True, pad_to_multiple_of=pad_to_multiple_of, return_tensors="pt"
        )
        self.attacher = _ProxyAttacher(cache, thinking_token_ids, strict)

    @property
    def missing_members(self) -> int:
        return self.attacher.missing_members

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        model_features = [{k: f[k] for k in ("input_ids", "attention_mask", "labels") if k in f} for f in features]
        batch = self.base(model_features)
        bsz, seq = batch["labels"].shape
        out = _ProxyTensors.empty(bsz, seq, self.attacher.top_k)
        left = self.tokenizer.padding_side == "left"
        for b, feature in enumerate(features):
            doc_ids = torch.as_tensor(feature["input_ids"], dtype=torch.long)
            doc_labels = torch.as_tensor(feature["labels"], dtype=torch.long)
            start = seq - doc_ids.numel() if left else 0
            self.attacher.fill(out, b, start, doc_ids, doc_labels, row_key(doc_ids.tolist()))
        batch.update(out.as_batch())
        return batch


class ProxyAwarePackedCollator(PackedSequenceCollator):
    """Packed blocks (no attention_mask, FA2 varlen via position_ids) plus `tw_*`, using the
    `member_keys`/`member_lengths` columns of `pack_dataset(..., with_members=True)`."""

    def __init__(
        self,
        cache: ProxyCache | None,
        thinking_token_ids: tuple[int, int],
        strict: bool = True,
        label_pad_token_id: int = IGNORE_LABEL,
    ):
        super().__init__(label_pad_token_id=label_pad_token_id)
        self.attacher = _ProxyAttacher(cache, thinking_token_ids, strict)

    @property
    def missing_members(self) -> int:
        return self.attacher.missing_members

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        batch = super().__call__(features)
        input_ids, labels = batch["input_ids"], batch["labels"]
        bsz, seq = labels.shape
        out = _ProxyTensors.empty(bsz, seq, self.attacher.top_k)
        for b, feature in enumerate(features):
            keys = feature.get("member_keys")
            lengths = feature.get("member_lengths")
            if keys is None or lengths is None:
                continue
            start = 0
            for key, length in zip(_as_list(keys), _as_list(lengths)):
                end = start + length
                self.attacher.fill(out, b, start, input_ids[b, start:end], labels[b, start:end], key)
                start = end
        batch.update(out.as_batch())
        return batch


def _as_list(value) -> list[int]:
    if isinstance(value, torch.Tensor):
        return value.tolist()
    return [int(v) for v in value]
