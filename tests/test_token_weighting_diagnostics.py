"""CPU test for the pre-training diagnostic on the tiny model used by the trainer tests."""

import json

import numpy as np
import torch

from core.datasets.sequence_packing import row_key
from core.training.token_weighting.diagnostics import summarize, token_diagnostics
from core.training.token_weighting.proxy_cache import build_proxy_cache
from core.training.token_weighting.token_stats import coarse_kl
from tests.test_token_weighting_trainer import (
    THINK_END,
    THINK_START,
    TOP_K,
    VOCAB,
    _rows,
    _tiny_model,
)


def test_diagnostics_frame_and_summary(tmp_path):
    proxy, student = _tiny_model(11), _tiny_model(12)
    rows = _rows(13, [(3, 8), (2, 5), (4, 7)])
    cache = build_proxy_cache(
        proxy,
        rows,
        str(tmp_path / "cache"),
        top_k=TOP_K,
        vocab_size=VOCAB,
        flush_every=1,
    )

    df = token_diagnostics(
        student, rows, cache, (THINK_START, THINK_END), max_rows=2, chunk_size=3
    )
    assert df["row_id"].nunique() == 2
    labeled = sum(sum(1 for t in labs if t != -100) for labs in rows["labels"][:2])
    assert len(df) == labeled
    assert set(df.columns) >= {
        "h_p",
        "h_s",
        "kl",
        "rho",
        "eg",
        "discord",
        "logp_p",
        "logp_s",
        "structural",
    }
    # <think>, </think>, answer, eos per row
    assert int(df["structural"].sum()) == 2 * 4
    assert ((df["rho"] >= 0) & (df["rho"] <= 1)).all()
    assert ((df["rho2"] >= 0) & (df["rho2"] <= 1)).all()
    assert ((df["pp"] > 0) & (df["pp"] <= 1) & (df["ps"] >= 0) & (df["ps"] <= 1)).all()
    assert (df["kl"] >= 0).all()

    # Spot-check one token against a direct computation.
    row = rows[0]
    with torch.inference_mode():
        lp_s = torch.log_softmax(
            student(input_ids=torch.tensor([row["input_ids"]])).logits[0].float(), -1
        )
    first_label = next(s for s, lab in enumerate(row["labels"]) if lab != -100)
    sl = cache.slice(row_key(row["input_ids"]))
    lp_pred = lp_s[first_label - 1]
    kl0 = coarse_kl(
        torch.as_tensor(np.array(sl["topk_logp"][0], dtype=np.float32))[None],
        lp_pred[torch.as_tensor(np.array(sl["topk_ids"][0], dtype=np.int64))][None],
    )[0]
    got = df[(df["row_id"] == row["row_id"]) & (df["position"] == 0)].iloc[0]
    assert abs(got["kl"] - float(kl0)) < 1e-4
    assert abs(got["logp_s"] - float(lp_pred[row["input_ids"][first_label]])) < 1e-4

    summary = summarize(df)
    json.dumps(summary)  # serialisable
    assert summary["rows"] == 2 and summary["tokens"] == labeled
    assert (
        abs(sum(sum(r.values()) for r in summary["h_p_x_kl_share"].values()) - 1.0)
        < 1e-6
    )
    assert set(summary["top_decile_weight_share"]) == {
        "rho",
        "kl",
        "eg",
        "discord",
        "rho2",
        "discord_tau4",
    }
    assert "share_rho_gt_0.8_and_discord_tau4_lt_0.2" in summary["conflicts"]
