"""Per-token statistics and weights shared by the trainer and the diagnostics script.

All functions are shape-preserving over a leading batch of positions and work on any
device. Proxy quantities come from the cache (built once, teacher-forced); student
quantities come from the current training forward pass.

Notation (per target position t, with y_t the trace token):
    H_P      full-vocab entropy of the proxy,   nats
    H_S      full-vocab entropy of the student, nats
    KL_t     KL(p_P || p_S) estimated on the proxy's top-K support (see coarse_kl)
    logp_P   log p_P(y_t), logp_S = log p_S(y_t)
"""

import math

import torch

from core.training.token_weighting.config import WeightingMode


def coarse_kl(topk_logp_p: torch.Tensor, topk_logp_s: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """KL(P || S) over the partition {each of the proxy's top-K tokens, the rest}.

    `topk_logp_p`/`topk_logp_s` are [..., K] log-probabilities of the same K token ids (the
    proxy's top-K) under the proxy and the student. The complement of the top-K is one
    bucket with the leftover mass of each model. By the data-processing inequality the
    coarsened KL is a lower bound of the full-vocab KL, and it is exact when the top-K
    holds all the proxy's mass. fp16-stored proxy log-probs can overshoot a total of 1 by
    rounding, hence the clamps.
    """
    p = topk_logp_p.float().exp()
    head = (p * (topk_logp_p.float() - topk_logp_s.float())).sum(-1)
    p_tail = (1.0 - p.sum(-1)).clamp_min(0.0)
    s_tail = (1.0 - topk_logp_s.float().exp().sum(-1)).clamp_min(eps)
    tail = torch.where(p_tail > eps, p_tail * (p_tail.clamp_min(eps).log() - s_tail.log()), torch.zeros_like(p_tail))
    return (head + tail).clamp_min(0.0)


def entropy_from_logp(logp: torch.Tensor) -> torch.Tensor:
    """Entropy (nats) of a full distribution given as [..., V] log-probabilities.
    One [..., V] temporary (in-place product), which matters for 150k-vocab chunks."""
    p = logp.exp()
    p.mul_(logp)
    return -p.sum(-1)


def reducible_fraction(kl: torch.Tensor, h_p: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """rho_t = KL_t / (H_P(t) + KL_t). 0 when the student already matches a proxy that is itself
    uncertain, 1 when the proxy is certain and the student is not."""
    return kl / (h_p + kl + eps)


def entropy_gain(h_s: torch.Tensor, h_p: torch.Tensor) -> torch.Tensor:
    """Token-level counterpart of the question-level EG(x) = max{H_S - H_P, 0}."""
    return (h_s - h_p).clamp_min(0.0)


def discord_weight(logp_s: torch.Tensor, logp_p: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """DISCORD token weight: sigmoid(tau * log(p_S(y_t) / p_P(y_t))). Down-weights tokens the
    reference finds far more plausible than the student does."""
    return torch.sigmoid(tau * (logp_s - logp_p))


def raw_weights(
    mode: WeightingMode,
    *,
    kl: torch.Tensor,
    h_p: torch.Tensor,
    h_s: torch.Tensor,
    logp_p: torch.Tensor,
    logp_s: torch.Tensor,
    rho_eps: float,
    discord_tau: float,
) -> torch.Tensor:
    if mode == "uniform":
        return torch.ones_like(kl)
    if mode == "rho":
        return reducible_fraction(kl, h_p, rho_eps)
    if mode == "kl":
        return kl
    if mode == "eg":
        return entropy_gain(h_s, h_p)
    if mode == "discord":
        return discord_weight(logp_s, logp_p, discord_tau)
    if mode == "excess_loss":
        return logp_p - logp_s  # l_S - l_P
    if mode == "student_entropy":
        return h_s
    raise ValueError(f"Unknown token weighting mode {mode!r}")


def select_top_fraction(score: torch.Tensor, content: torch.Tensor, frac: float) -> torch.Tensor:
    """Binary weights [B, T]: per row, 1 on the `frac` share of content positions with the highest
    score (at least one), 0 elsewhere. Non-content positions are never selected."""
    out = torch.zeros_like(score)
    neg_inf = torch.finfo(score.dtype).min
    masked = torch.where(content, score, torch.full_like(score, neg_inf))
    for b in range(score.shape[0]):
        n = int(content[b].sum())
        if n == 0:
            continue
        k = max(1, min(n, int(math.ceil(frac * n))))
        idx = torch.topk(masked[b], k).indices
        out[b, idx] = 1.0
    return out


def finalize_weights(
    raw: torch.Tensor,
    *,
    selected: torch.Tensor,
    structural: torch.Tensor,
    valid: torch.Tensor,
    structural_weight: float,
    normalize_per_pack: bool,
) -> torch.Tensor:
    """Turn raw per-position weights [B, T] into the weights applied to the loss.

    - positions outside `selected` (no label) get 0;
    - `structural` positions get `structural_weight` and positions without proxy statistics
      (`valid` False) get 1, regardless of the mode; neither takes part in the normalization;
    - with `normalize_per_pack`, the remaining (content) positions of every batch row are
      rescaled so their weights sum to their count: the content tokens keep a mean weight of
      1, exactly as in the uniform run, and only the distribution among them changes. A row
      whose content weights sum to zero falls back to uniform.
    """
    content = selected & valid & ~structural
    w = torch.where(content, raw, torch.ones_like(raw))
    w = torch.where(structural, torch.full_like(w, structural_weight), w)
    w = torch.where(selected, w, torch.zeros_like(w))
    if normalize_per_pack:
        n_content = content.sum(-1, keepdim=True).to(w.dtype)
        total = (w * content).sum(-1, keepdim=True)
        scale = torch.where(total > 0, n_content / total.clamp_min(torch.finfo(w.dtype).tiny), torch.ones_like(total))
        normalized = torch.where(total > 0, w * scale, torch.ones_like(w))
        w = torch.where(content, normalized, w)
    return w
