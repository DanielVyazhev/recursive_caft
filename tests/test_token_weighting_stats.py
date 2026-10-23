"""CPU tests for the token-weighting math and the packing metadata it rides on."""

import pytest
import torch
from datasets import Dataset

from core.datasets.sequence_packing import IGNORE_LABEL, pack_dataset, row_key
from core.training.token_weighting.token_stats import (
    coarse_kl,
    discord_weight,
    entropy_from_logp,
    entropy_gain,
    finalize_weights,
    raw_weights,
    reducible_fraction,
)


def _random_logp(n: int, v: int, seed: int, temperature: float = 1.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.log_softmax(torch.randn(n, v, generator=g) / temperature, dim=-1)


def test_coarse_kl_matches_full_kl_on_full_support():
    v = 16
    lp_p, lp_s = _random_logp(8, v, 0), _random_logp(8, v, 1)
    full = (lp_p.exp() * (lp_p - lp_s)).sum(-1)
    top_lp, top_ids = lp_p.topk(v, dim=-1)
    got = coarse_kl(top_lp, lp_s.gather(1, top_ids))
    assert torch.allclose(got, full, atol=1e-5)


def test_coarse_kl_is_a_lower_bound_and_monotone_in_k():
    v = 40
    lp_p, lp_s = _random_logp(32, v, 2, temperature=0.5), _random_logp(32, v, 3)
    full = (lp_p.exp() * (lp_p - lp_s)).sum(-1)
    prev = torch.zeros(32)
    for k in (1, 4, 8, 16, 40):
        top_lp, top_ids = lp_p.topk(k, dim=-1)
        got = coarse_kl(top_lp, lp_s.gather(1, top_ids))
        assert (got <= full + 1e-5).all()
        assert (got >= prev - 1e-5).all()
        prev = got


def test_coarse_kl_zero_for_identical_distributions_and_fp16_storage():
    lp = _random_logp(5, 20, 4)
    top_lp, top_ids = lp.topk(8, dim=-1)
    got = coarse_kl(top_lp.to(torch.float16), lp.gather(1, top_ids))
    assert (got >= 0).all()
    assert got.max() < 1e-3


def test_entropy_from_logp():
    lp = torch.log(torch.tensor([[0.5, 0.5], [1.0, 0.0]]).clamp_min(1e-30))
    h = entropy_from_logp(lp)
    assert torch.isclose(h[0], torch.log(torch.tensor(2.0)))
    assert torch.isclose(h[1], torch.tensor(0.0), atol=1e-6)


def test_reducible_fraction_limits():
    kl = torch.tensor([0.0, 1.0, 1.0, 5.0])
    h_p = torch.tensor([1.0, 0.0, 1.0, 0.0])
    rho = reducible_fraction(kl, h_p, eps=1e-3)
    assert rho[0] == 0
    assert rho[1] > 0.99
    assert abs(rho[2] - 0.5) < 1e-3
    assert (rho >= 0).all() and (rho <= 1).all()


def test_entropy_gain_and_discord():
    assert torch.equal(entropy_gain(torch.tensor([2.0, 0.5]), torch.tensor([1.0, 1.0])), torch.tensor([1.0, 0.0]))
    w = discord_weight(torch.log(torch.tensor([0.2, 0.9, 0.01])), torch.log(torch.tensor([0.2, 0.1, 0.9])))
    assert abs(w[0] - 0.5) < 1e-6
    assert w[1] > 0.5 > w[2]


def test_raw_weights_dispatch():
    n = 6
    kw = dict(
        kl=torch.rand(n),
        h_p=torch.rand(n),
        h_s=torch.rand(n) + 1,
        logp_p=-torch.rand(n),
        logp_s=-torch.rand(n),
        rho_eps=1e-3,
        discord_tau=1.0,
    )
    assert torch.equal(raw_weights("uniform", **kw), torch.ones(n))
    assert torch.equal(raw_weights("kl", **kw), kw["kl"])
    assert torch.allclose(raw_weights("rho", **kw), reducible_fraction(kw["kl"], kw["h_p"], 1e-3))
    assert torch.allclose(raw_weights("eg", **kw), entropy_gain(kw["h_s"], kw["h_p"]))
    assert torch.allclose(raw_weights("discord", **kw), discord_weight(kw["logp_s"], kw["logp_p"], 1.0))
    assert torch.allclose(raw_weights("excess_loss", **kw), kw["logp_p"] - kw["logp_s"])
    assert torch.equal(raw_weights("student_entropy", **kw), kw["h_s"])
    with pytest.raises(ValueError):
        raw_weights("nope", **kw)  # type: ignore[arg-type]


def test_finalize_weights_masks_and_normalizes_per_row():
    raw = torch.tensor([[0.2, 0.4, 0.8, 0.0, 9.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
    selected = torch.tensor([[True, True, True, True, False], [True, True, False, False, False]])
    structural = torch.tensor([[False, False, False, True, False], [False, False, False, False, False]])
    valid = torch.tensor([[True, True, False, True, True], [True, True, True, True, True]])
    w = finalize_weights(
        raw, selected=selected, structural=structural, valid=valid, structural_weight=1.0, normalize_per_pack=True
    )
    # unselected -> 0; structural and invalid stay at exactly 1; the content positions (0, 1)
    # are rescaled to a mean of 1 while keeping their ratio
    assert w[0, 4] == 0
    assert w[0, 2] == 1.0 and w[0, 3] == 1.0
    assert torch.isclose(w[0, 0] + w[0, 1], torch.tensor(2.0))
    assert torch.isclose(w[0, 1] / w[0, 0], torch.tensor(2.0))
    assert torch.isclose(w[0].sum(), torch.tensor(4.0))
    # a row whose raw content weights are all zero falls back to uniform over selected positions
    assert torch.equal(w[1], torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0]))

    w_raw = finalize_weights(
        raw, selected=selected, structural=structural, valid=valid, structural_weight=1.0, normalize_per_pack=False
    )
    assert torch.equal(w_raw[0], torch.tensor([0.2, 0.4, 1.0, 1.0, 0.0]))


def test_row_key_is_deterministic_and_content_bound():
    assert row_key([1, 2, 3]) == row_key((1, 2, 3))
    assert row_key([1, 2, 3]) != row_key([1, 2, 4])
    assert row_key([1, 2, 3]) != row_key([1, 2, 3, 0])
    key = row_key(list(range(1000)))
    assert -(2**63) <= key < 2**63
    assert isinstance(key, int)


def _tokenized_rows(lengths: list[int], prefix: int = 2) -> Dataset:
    rows = []
    for i, n in enumerate(lengths):
        ids = [100 + i] * n
        rows.append({"input_ids": ids, "labels": [IGNORE_LABEL] * prefix + ids[prefix:], "row_id": str(i)})
    return Dataset.from_list(rows)


def test_pack_dataset_members_are_opt_in_and_aligned():
    ds = _tokenized_rows([5, 3, 4, 2])
    plain = pack_dataset(ds, budget=8, pad_token_id=0)
    assert set(plain.column_names) == {"input_ids", "labels", "position_ids"}

    packed = pack_dataset(ds, budget=8, pad_token_id=0, with_members=True)
    assert set(packed.column_names) == {"input_ids", "labels", "position_ids", "member_keys", "member_lengths"}
    keys_by_row = {row_key(ids): ids for ids in ds["input_ids"]}
    seen = 0
    for block in packed:
        start = 0
        for key, length in zip(block["member_keys"], block["member_lengths"]):
            segment = block["input_ids"][start : start + length]
            assert segment == keys_by_row[key]
            assert block["position_ids"][start : start + length] == list(range(length))
            start += length
            seen += 1
        assert all(tok == 0 for tok in block["input_ids"][start:])
    assert seen == len(ds)


def test_select_top_fraction_per_row():
    from core.training.token_weighting.token_stats import select_top_fraction

    score = torch.tensor([[0.1, 0.9, 0.5, 0.7, 0.3, 9.0], [0.2, 0.1, 0.0, 0.0, 0.0, 0.0]])
    content = torch.tensor([[True, True, True, True, True, False], [True, True, False, False, False, False]])
    w = select_top_fraction(score, content, 0.4)
    # row 0: 5 content tokens -> ceil(2.0) = 2 kept: scores 0.9 and 0.7; the non-content 9.0 is never kept
    assert w[0].tolist() == [0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
    # row 1: 2 content tokens -> ceil(0.8) = 1 kept
    assert w[1].tolist() == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert select_top_fraction(score, content, 1.0)[0].sum() == 5
    # negative scores (excess loss) rank fine
    neg = -score
    assert select_top_fraction(neg, content, 0.4)[0].tolist() == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]


def test_excess_loss_mode_requires_selection():
    from core.training.token_weighting.config import TokenWeightingConfig

    with pytest.raises(Exception):
        TokenWeightingConfig(mode="excess_loss", proxy_cache_path="x")
    cfg = TokenWeightingConfig(mode="excess_loss", proxy_cache_path="x", select_top_frac=0.6)
    assert cfg.select_top_frac == 0.6
    with pytest.raises(Exception):
        TokenWeightingConfig(mode="student_entropy", proxy_cache_path="x")
    with pytest.raises(Exception):
        TokenWeightingConfig(mode="rho", proxy_cache_path="x", select_top_frac=1.5)
