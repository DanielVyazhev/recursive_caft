"""Packed collator that attaches per-token proxy statistics to each block.

Extends `PackedSequenceCollator` (same `input_ids`/`labels`/`position_ids`, still no
`attention_mask`) with `tw_*` tensors aligned to the LABEL positions of the block: index
`s` holds the proxy's prediction for the token at `s` (produced one position earlier).
The trainer shifts them together with the labels.

    tw_entropy    float32 [B, T]     H_P(s)
    tw_logp       float32 [B, T]     log p_P(input_ids[s])
    tw_topk_ids   int64   [B, T, K]  proxy top-K ids at s
    tw_topk_logp  float32 [B, T, K]  their proxy log-probs
    tw_valid      bool    [B, T]     proxy statistics present (labeled position of a cached member)
    tw_structural bool    [B, T]     <think>, </think> and every labeled token after </think>

Blocks without `member_keys` (plain packs, null packs) get no valid positions, which the
loss treats as weight 1.
"""

import numpy as np
import torch

from core.datasets.sequence_packing import IGNORE_LABEL
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


class ProxyAwarePackedCollator(PackedSequenceCollator):
    def __init__(
        self,
        cache: ProxyCache | None,
        thinking_token_ids: tuple[int, int],
        strict: bool = True,
        label_pad_token_id: int = IGNORE_LABEL,
    ):
        super().__init__(label_pad_token_id=label_pad_token_id)
        self.cache = cache
        self.thinking_token_ids = thinking_token_ids
        self.strict = strict
        self.top_k = cache.top_k if cache is not None else 1
        self.missing_members = 0
        self.seen_members = 0

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        batch = super().__call__(features)
        input_ids, labels = batch["input_ids"], batch["labels"]
        bsz, seq = labels.shape
        k = self.top_k

        entropy = torch.zeros(bsz, seq, dtype=torch.float32)
        logp = torch.zeros(bsz, seq, dtype=torch.float32)
        topk_ids = torch.zeros(bsz, seq, k, dtype=torch.int64)
        topk_logp = torch.zeros(bsz, seq, k, dtype=torch.float32)
        valid = torch.zeros(bsz, seq, dtype=torch.bool)
        structural = torch.zeros(bsz, seq, dtype=torch.bool)

        for b, feature in enumerate(features):
            keys = feature.get("member_keys")
            lengths = feature.get("member_lengths")
            if keys is None or lengths is None:
                continue
            start = 0
            for key, length in zip(_as_list(keys), _as_list(lengths)):
                end = start + length
                doc_ids, doc_labels = input_ids[b, start:end], labels[b, start:end]
                labeled = (doc_labels != IGNORE_LABEL).nonzero(as_tuple=True)[0]
                structural[b, start:end] = structural_mask(doc_ids, doc_labels, self.thinking_token_ids)
                self.seen_members += 1
                entry = self.cache.lookup(key) if self.cache is not None else None
                if entry is None:
                    self.missing_members += 1
                    if self.strict and self.cache is not None:
                        raise KeyError(f"packed member with key {key} has no proxy cache entry")
                    start = end
                    continue
                _, n = entry
                assert n == labeled.numel(), (
                    f"proxy cache entry has {n} tokens but the member has {labeled.numel()} labeled tokens"
                )
                sl = self.cache.slice(key)
                pos = labeled + start
                entropy[b, pos] = torch.as_tensor(np.array(sl["entropy"], dtype=np.float32))
                logp[b, pos] = torch.as_tensor(np.array(sl["logp"], dtype=np.float32))
                topk_ids[b, pos] = torch.as_tensor(np.array(sl["topk_ids"], dtype=np.int64))
                topk_logp[b, pos] = torch.as_tensor(np.array(sl["topk_logp"], dtype=np.float32))
                valid[b, pos] = True
                start = end

        batch.update(
            tw_entropy=entropy,
            tw_logp=logp,
            tw_topk_ids=topk_ids,
            tw_topk_logp=topk_logp,
            tw_valid=valid,
            tw_structural=structural,
        )
        return batch


def _as_list(value) -> list[int]:
    if isinstance(value, torch.Tensor):
        return value.tolist()
    return [int(v) for v in value]
