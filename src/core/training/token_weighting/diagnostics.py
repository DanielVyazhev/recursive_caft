"""Pre-training diagnostic: where the proxy and the untrained student disagree.

Runs the student teacher-forced over cached rows, computes the same per-token quantities the
trainer uses (H_P, H_S, KL, rho, EG, DISCORD) and summarises their joint distribution. The
point of the summary is the (H_P, KL) plane: rho keeps the tokens where the proxy is
confident and the student is off (low H_P, high KL) and drops the ones where the proxy is
itself uncertain, while DISCORD drops exactly the former whenever the student is far
behind the proxy on the trace token.
"""

from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from tqdm import tqdm

from core.datasets.sequence_packing import IGNORE_LABEL, row_key
from core.training.token_weighting.collator import structural_mask
from core.training.token_weighting.proxy_cache import ProxyCache
from core.training.token_weighting.token_stats import (
    coarse_kl,
    discord_weight,
    entropy_from_logp,
    entropy_gain,
    reducible_fraction,
)

H_P_BINS = [0.0, 0.1, 0.5, 1.0, 2.0, float("inf")]
KL_BINS = [0.0, 0.1, 0.5, 1.0, 2.0, float("inf")]
MODES = ("rho", "kl", "eg", "discord")
# DISCORD tunes its sigmoid temperature per teacher-student pair; the frame keeps the paper
# default (tau = 1) in `discord`, and the summary adds the weights at these temperatures.
DISCORD_EXTRA_TAUS = (4.0,)


@torch.inference_mode()
def token_diagnostics(
    model,
    dataset: Dataset,
    cache: ProxyCache,
    thinking_token_ids: tuple[int, int],
    max_rows: int | None = None,
    rho_eps: float = 1e-3,
    discord_tau: float = 1.0,
    chunk_size: int = 2048,
) -> pd.DataFrame:
    """Per-token frame over the first `max_rows` cached rows of `dataset` (tokenized by the
    student tokenizer, as the proxy pass saw it)."""
    model.eval()
    device = next(model.parameters()).device
    frames: list[pd.DataFrame] = []
    n_rows = len(dataset) if max_rows is None else min(max_rows, len(dataset))
    for i in tqdm(range(n_rows), desc="student pass"):
        row = dataset[i]
        input_ids = torch.tensor(row["input_ids"], dtype=torch.long)
        labels = torch.tensor(row["labels"], dtype=torch.long)
        key = row_key(row["input_ids"])
        if key not in cache:
            continue
        sl = cache.slice(key)
        target_pos = (labels != IGNORE_LABEL).nonzero(as_tuple=True)[0]
        assert target_pos.numel() == len(sl["entropy"])
        if target_pos.numel() == 0:
            continue
        targets = input_ids[target_pos]
        structural = structural_mask(input_ids, labels, thinking_token_ids)[target_pos]

        out = model(input_ids=input_ids[None].to(device), logits_to_keep=target_pos - 1, use_cache=False)
        logits = out.logits[0]
        topk_ids = torch.as_tensor(np.array(sl["topk_ids"], dtype=np.int64), device=logits.device)
        logp_s_parts, h_s_parts, topk_s_parts = [], [], []
        for a in range(0, logits.shape[0], chunk_size):
            lp = torch.log_softmax(logits[a : a + chunk_size].float(), dim=-1)
            y = targets[a : a + chunk_size].to(lp.device)
            logp_s_parts.append(lp.gather(1, y[:, None]).squeeze(1).cpu())
            h_s_parts.append(entropy_from_logp(lp).cpu())
            topk_s_parts.append(lp.gather(1, topk_ids[a : a + chunk_size]).cpu())
        logp_s = torch.cat(logp_s_parts)
        h_s = torch.cat(h_s_parts)
        topk_logp_s = torch.cat(topk_s_parts)

        h_p = torch.as_tensor(np.array(sl["entropy"], dtype=np.float32))
        logp_p = torch.as_tensor(np.array(sl["logp"], dtype=np.float32))
        kl = coarse_kl(torch.as_tensor(np.array(sl["topk_logp"], dtype=np.float32)), topk_logp_s)
        frames.append(
            pd.DataFrame(
                {
                    "row_id": str(row["row_id"]),
                    "position": np.arange(targets.numel(), dtype=np.int32),
                    "token_id": targets.numpy().astype(np.int32),
                    "structural": structural.numpy(),
                    "h_p": h_p.numpy(),
                    "h_s": h_s.numpy(),
                    "kl": kl.numpy(),
                    "rho": reducible_fraction(kl, h_p, rho_eps).numpy(),
                    "eg": entropy_gain(h_s, h_p).numpy(),
                    "discord": discord_weight(logp_s, logp_p, discord_tau).numpy(),
                    "logp_p": logp_p.numpy(),
                    "logp_s": logp_s.numpy(),
                }
            )
        )
    assert frames, "no cached rows found in the dataset"
    return pd.concat(frames, ignore_index=True)


def _top_decile_share(weights: np.ndarray) -> float:
    if weights.size == 0 or weights.sum() <= 0:
        return float("nan")
    w = np.sort(weights)[::-1]
    k = max(1, int(np.ceil(0.1 * w.size)))
    return float(w[:k].sum() / w.sum())


def summarize(df: pd.DataFrame, corr_sample: int = 1_000_000, seed: int = 0) -> dict[str, Any]:
    content = df[~df["structural"]].copy()
    extra_discord = []
    for tau in DISCORD_EXTRA_TAUS:
        name = f"discord_tau{tau:g}"
        content[name] = discord_weight(
            torch.tensor(content["logp_s"].to_numpy()), torch.tensor(content["logp_p"].to_numpy()), tau
        ).numpy()
        extra_discord.append(name)
    quant = {}
    for col in ("h_p", "h_s", "kl", "rho", "eg", "discord", *extra_discord):
        values = content[col].to_numpy(dtype=np.float64)
        quant[col] = {
            "mean": float(values.mean()),
            "p10": float(np.percentile(values, 10)),
            "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)),
        }

    h_bins = pd.cut(content["h_p"], H_P_BINS, right=False, include_lowest=True)
    kl_bins = pd.cut(content["kl"], KL_BINS, right=False, include_lowest=True)
    joint = pd.crosstab(h_bins, kl_bins, normalize=True)
    joint_table = {str(h): {str(k): float(v) for k, v in row.items()} for h, row in joint.iterrows()}

    rho_high = content["rho"] > 0.8
    conflicts: dict[str, float] = {
        "share_rho_gt_0.8": float(rho_high.mean()),
        "share_h_p_lt_0.5_and_kl_gt_1": float(((content["h_p"] < 0.5) & (content["kl"] > 1.0)).mean()),
        "share_h_p_gt_2": float((content["h_p"] > 2.0).mean()),
    }
    for name in ("discord", *extra_discord):
        low = content[name] < 0.2
        conflicts[f"share_{name}_lt_0.2"] = float(low.mean())
        conflicts[f"share_rho_gt_0.8_and_{name}_lt_0.2"] = float((rho_high & low).mean())
        conflicts[f"share_rho_lt_0.2_and_{name}_gt_0.8"] = float(((content["rho"] < 0.2) & (content[name] > 0.8)).mean())

    sample = content.sample(n=min(corr_sample, len(content)), random_state=seed)
    spearman = sample[["rho", "kl", "eg", "discord", "h_p", "h_s"]].corr(method="spearman")
    correlations = {
        "rho_vs_eg": float(spearman.loc["rho", "eg"]),
        "rho_vs_kl": float(spearman.loc["rho", "kl"]),
        "rho_vs_discord": float(spearman.loc["rho", "discord"]),
        "kl_vs_discord": float(spearman.loc["kl", "discord"]),
        "h_p_vs_h_s": float(spearman.loc["h_p", "h_s"]),
    }

    # Weight concentration after per-row mean-1 normalisation (what the trainer does per pack).
    concentration = {}
    for mode in (*MODES, *extra_discord):
        per_row_mean = content.groupby("row_id")[mode].transform("mean").replace(0, np.nan)
        normalized = (content[mode] / per_row_mean).fillna(1.0).to_numpy(dtype=np.float64)
        concentration[mode] = _top_decile_share(normalized)

    return {
        "rows": int(df["row_id"].nunique()),
        "tokens": len(df),
        "content_tokens": len(content),
        "structural_share": float(df["structural"].mean()),
        "quantiles": quant,
        "h_p_x_kl_share": joint_table,
        "conflicts": conflicts,
        "spearman": correlations,
        "top_decile_weight_share": concentration,
    }
