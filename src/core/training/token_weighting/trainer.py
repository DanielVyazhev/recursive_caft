"""Seq2SeqTrainer with a per-token weighted cross-entropy driven by proxy statistics.

The model is called without `labels`, so it returns bf16 logits and no loss. The loss is
then assembled in fp32 over chunks of positions:

1. a no-grad pass computes the student side (log p_S(y_t), H_S(t), log p_S at the proxy's
   top-K ids), the per-token weights w_t (token_stats.raw_weights / finalize_weights) and
   the weighted sum of cross-entropies;
2. `_WeightedCrossEntropy` returns that sum and, in backward, writes the exact gradient
   w_t * (softmax(z_t) - onehot(y_t)) chunk by chunk into a single bf16 buffer, so no
   fp32 [T, V] tensor is ever kept alive between forward and backward.

The result is divided by HF's `num_items_in_batch` (labeled tokens in the accumulation
chunk), matching the token-normalized loss the stock trainer computes for `uniform`
weights. Weights are detached: they steer the loss but carry no gradient.
"""

from typing import override

import torch
from transformers.trainer_seq2seq import Seq2SeqTrainer

from core.datasets.sequence_packing import IGNORE_LABEL
from core.training.token_weighting.config import TokenWeightingConfig
from core.training.token_weighting.token_stats import (
    coarse_kl,
    discord_weight,
    entropy_from_logp,
    entropy_gain,
    finalize_weights,
    permute_within_rows,
    raw_weights,
    reducible_fraction,
    select_top_fraction,
    topk_renormalized_kl,
)
from core.utils.logger import logger

TW_PREFIX = "tw_"


class _WeightedCrossEntropy(torch.autograd.Function):
    """loss = sum_i w_i * CE(z_i, y_i) + sum_i kd_i * KL(p~_P,i || softmax_K(z_i[ids_i])), with the
    forward value precomputed (no-grad pass) and the gradient w.r.t. the logits recomputed
    chunk-wise in backward: w_i (softmax(z_i) - onehot(y_i)) plus, at the K proxy ids,
    kd_i (softmax_K(z_i[ids_i]) - p~_P,i)."""

    @staticmethod
    def forward(
        ctx,
        logits: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor,
        loss_value: torch.Tensor,
        chunk_size: int,
        topk_ids: torch.Tensor | None = None,
        topk_prob_p: torch.Tensor | None = None,
        kd_weights: torch.Tensor | None = None,
    ):
        ctx.has_kd = topk_ids is not None
        tensors = [logits, targets, weights]
        if ctx.has_kd:
            tensors += [topk_ids, topk_prob_p, kd_weights]
        ctx.save_for_backward(*tensors)
        ctx.chunk_size = chunk_size
        return loss_value.clone()

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.has_kd:
            logits, targets, weights, topk_ids, topk_prob_p, kd_weights = ctx.saved_tensors
        else:
            logits, targets, weights = ctx.saved_tensors
            topk_ids = topk_prob_p = kd_weights = None
        grad = torch.empty_like(logits)
        n = logits.shape[0]
        scale = grad_output.to(torch.float32)
        for a in range(0, n, ctx.chunk_size):
            b = min(a + ctx.chunk_size, n)
            w = weights[a:b]
            kd = kd_weights[a:b] if kd_weights is not None else None
            has_ce = bool(w.any())
            has_kd = kd is not None and bool(kd.any())
            if not has_ce and not has_kd:
                grad[a:b].zero_()
                continue
            if has_ce:
                g = torch.softmax(logits[a:b].float(), dim=-1)
                g[torch.arange(b - a, device=g.device), targets[a:b]] -= 1.0
                g.mul_((w * scale)[:, None])
            else:
                g = torch.zeros(b - a, logits.shape[1], dtype=torch.float32, device=logits.device)
            if has_kd:
                ids = topk_ids[a:b]
                q = torch.softmax(logits[a:b].gather(1, ids).float(), dim=-1)
                delta = (q - topk_prob_p[a:b]) * (kd * scale)[:, None]
                g.scatter_add_(1, ids, delta)
            grad[a:b] = g.to(grad.dtype)
        return grad, None, None, None, None, None, None, None


def student_statistics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    topk_ids: torch.Tensor,
    selected: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """No-grad student quantities for [N] positions: log p_S(y), H_S and log p_S at `topk_ids`
    ([N, K]). Positions outside `selected` are left at zero."""
    n, _ = logits.shape
    logp_s = torch.zeros(n, dtype=torch.float32, device=logits.device)
    h_s = torch.zeros(n, dtype=torch.float32, device=logits.device)
    topk_logp_s = torch.zeros(topk_ids.shape, dtype=torch.float32, device=logits.device)
    with torch.no_grad():
        for a in range(0, n, chunk_size):
            b = min(a + chunk_size, n)
            if not bool(selected[a:b].any()):
                continue
            lp = torch.log_softmax(logits[a:b].float(), dim=-1)
            logp_s[a:b] = lp.gather(1, targets[a:b, None]).squeeze(1)
            h_s[a:b] = entropy_from_logp(lp)
            topk_logp_s[a:b] = lp.gather(1, topk_ids[a:b])
    return logp_s, h_s, topk_logp_s


def _shift_left(x: torch.Tensor) -> torch.Tensor:
    """Align label-position tensors [B, T, ...] with the logits that predict them: out[:, t] = x[:, t + 1]."""
    out = torch.zeros_like(x)
    out[:, :-1] = x[:, 1:]
    return out


class WeightedTokenLossTrainer(Seq2SeqTrainer):
    def __init__(self, *args, token_weighting: TokenWeightingConfig, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_weighting = token_weighting
        self._tw_sums: dict[str, torch.Tensor] = {}
        self._warned_no_num_items = False

    @override
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        cfg = self.token_weighting
        tw = {k: inputs.pop(k) for k in [k for k in inputs if k.startswith(TW_PREFIX)]}
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
        bsz, seq, vocab = logits.shape
        n = bsz * seq
        k = tw["tw_topk_ids"].shape[-1]

        selected = _shift_left(labels != IGNORE_LABEL)
        targets = _shift_left(labels).clamp_min(0)
        h_p = _shift_left(tw["tw_entropy"])
        logp_p = _shift_left(tw["tw_logp"])
        topk_ids = _shift_left(tw["tw_topk_ids"])
        topk_logp_p = _shift_left(tw["tw_topk_logp"])
        valid = _shift_left(tw["tw_valid"]) & selected
        structural = _shift_left(tw["tw_structural"]) & selected

        flat_logits = logits.reshape(n, vocab)
        flat_targets = targets.reshape(n)
        logp_s, h_s, topk_logp_s = student_statistics(
            flat_logits, flat_targets, topk_ids.reshape(n, k), selected.reshape(n), cfg.chunk_size
        )
        with torch.no_grad():
            kl = coarse_kl(topk_logp_p.reshape(n, k), topk_logp_s)
            raw = raw_weights(
                cfg.mode,
                kl=kl,
                h_p=h_p.reshape(n),
                h_s=h_s,
                logp_p=logp_p.reshape(n),
                logp_s=logp_s,
                rho_eps=cfg.rho_eps,
                discord_tau=cfg.discord_tau,
            )
            raw = raw.reshape(bsz, seq)
            if cfg.permute_within_row:
                step = int(self.state.global_step) if self.state is not None else 0
                raw = permute_within_rows(raw, selected & valid & ~structural, seed=self.args.seed * 1_000_003 + step)
            if cfg.select_top_frac is not None:
                raw = select_top_fraction(raw, selected & valid & ~structural, cfg.select_top_frac)
            weights = finalize_weights(
                raw,
                selected=selected,
                structural=structural,
                valid=valid,
                structural_weight=cfg.structural_weight,
                normalize_per_pack=cfg.normalize_per_pack,
            )
            if cfg.ce_scale != 1.0:
                weights = torch.where(selected & valid & ~structural, weights * cfg.ce_scale, weights)
            weights = weights.reshape(n)
            ce = -logp_s
            loss_sum = (weights * ce).sum()
            kd_args: tuple = ()
            kd_value = None
            if cfg.kd_lambda > 0.0:
                kd_mask = (selected & valid & ~structural).reshape(n)
                kd_weights = kd_mask.to(torch.float32) * cfg.kd_lambda
                topk_logp_p_flat = topk_logp_p.reshape(n, k)
                kd_value = topk_renormalized_kl(topk_logp_p_flat, topk_logp_s)
                loss_sum = loss_sum + (kd_weights * kd_value).sum()
                kd_args = (topk_ids.reshape(n, k), torch.softmax(topk_logp_p_flat.float(), dim=-1), kd_weights)
            self._accumulate_stats(
                kl=kl,
                h_p=h_p.reshape(n),
                h_s=h_s,
                logp_p=logp_p.reshape(n),
                logp_s=logp_s,
                weights=weights,
                selected=selected.reshape(n),
                valid=valid.reshape(n),
                structural=structural.reshape(n),
                kd=kd_value,
            )

        loss = _WeightedCrossEntropy.apply(flat_logits, flat_targets, weights, loss_sum, cfg.chunk_size, *kd_args)
        if num_items_in_batch is not None:
            loss = loss / num_items_in_batch
            if self.args.average_tokens_across_devices:
                loss = loss * self.accelerator.num_processes
        else:
            if not self._warned_no_num_items:
                logger.warning("num_items_in_batch is None; falling back to a per-batch token mean")
                self._warned_no_num_items = True
            loss = loss / selected.sum().clamp_min(1)

        return (loss, outputs) if return_outputs else loss

    def _accumulate_stats(self, *, kl, h_p, h_s, logp_p, logp_s, weights, selected, valid, structural, kd=None) -> None:
        # Everything stays on the device: a .item()/float() here would sync the GPU on every
        # micro-step (64 per update) and serialise the collator with the model.
        cfg = self.token_weighting
        content = selected & valid & ~structural
        structural_sel = selected & structural
        rho = reducible_fraction(kl, h_p, cfg.rho_eps)
        eg = entropy_gain(h_s, h_p)
        disc = discord_weight(logp_s, logp_p, cfg.discord_tau)
        values = {
            "kl": kl,
            "h_p": h_p,
            "h_s": h_s,
            "rho": rho,
            "eg": eg,
            "discord": disc,
            "ce_p": -logp_p,
            "ce_s": -logp_s,
            "weight": weights,
        }
        if kd is not None:
            values["kd"] = kd
        content_f = content.to(kl.dtype)
        sums = {name: (value * content_f).sum() for name, value in values.items()}
        sums["weight_structural"] = (weights * structural_sel.to(kl.dtype)).sum()
        sums["missing"] = (selected & ~valid).sum().to(kl.dtype)
        sums["selected"] = selected.sum().to(kl.dtype)
        sums["count"] = content.sum().to(kl.dtype)
        sums["structural_count"] = structural_sel.sum().to(kl.dtype)
        for name, value in sums.items():
            self._tw_sums[name] = self._tw_sums[name] + value if name in self._tw_sums else value

    def _reset_stats(self) -> None:
        self._tw_sums = {}

    @override
    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        if "loss" in logs and self._tw_sums:
            sums = {name: float(value) for name, value in self._tw_sums.items()}
            count = sums.get("count", 0.0)
            if count > 0:
                for name in ("kl", "h_p", "h_s", "rho", "eg", "discord", "ce_p", "ce_s", "weight", "kd"):
                    if name in sums:
                        logs[f"tw/{name}"] = round(sums[name] / count, 4)
            if sums.get("structural_count", 0.0) > 0:
                logs["tw/weight_structural"] = round(sums["weight_structural"] / sums["structural_count"], 4)
            if sums.get("selected", 0.0) > 0:
                logs["tw/missing_frac"] = round(sums["missing"] / sums["selected"], 4)
            self._reset_stats()
        super().log(logs, start_time)
