"""CPU tests for the proxy cache, the proxy-aware collator and the weighted loss, on a tiny
randomly initialised Qwen2 (no hub access, no GPU, no flash-attn).

The loss checks lock in the contract with the stock trainer: with `uniform` weights the
value and the logits gradient coincide with HF's token-normalized ForCausalLMLoss, and with
proxy-driven weights the loss equals the brute-force sum_i w_i * CE_i / num_items_in_batch.
"""

import numpy as np
import pytest
import torch
from datasets import Dataset
from transformers import Qwen2Config, Qwen2ForCausalLM, Seq2SeqTrainingArguments

from core.datasets.sequence_packing import IGNORE_LABEL, pack_dataset, row_key
from core.training.token_weighting.collator import ProxyAwarePackedCollator, ProxyAwarePaddedCollator, structural_mask
from core.training.token_weighting.config import TokenWeightingConfig
from core.training.token_weighting.proxy_cache import ProxyCache, ProxyCacheWriter, build_proxy_cache
from core.training.token_weighting.token_stats import coarse_kl, finalize_weights, raw_weights
from core.training.token_weighting.trainer import WeightedTokenLossTrainer, student_statistics

VOCAB = 64
THINK_START, THINK_END = VOCAB - 2, VOCAB - 1
PAD = 0
TOP_K = 8


def _tiny_model(seed: int) -> Qwen2ForCausalLM:
    torch.manual_seed(seed)
    config = Qwen2Config(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        tie_word_embeddings=False,
    )
    return Qwen2ForCausalLM(config).eval()


def _rows(seed: int, lengths: list[tuple[int, int]]) -> Dataset:
    """Rows shaped like CausalDatasetAdapter output: prompt prefix (labels -100), then
    <think> reasoning </think> answer <eos>, all labeled."""
    g = torch.Generator().manual_seed(seed)
    rows = []
    for i, (prefix_len, reasoning_len) in enumerate(lengths):
        prefix = torch.randint(1, THINK_START, (prefix_len,), generator=g).tolist()
        reasoning = torch.randint(1, THINK_START, (reasoning_len,), generator=g).tolist()
        answer = [int(torch.randint(1, 10, (1,), generator=g)), 2]
        response = [THINK_START, *reasoning, THINK_END, *answer]
        ids = prefix + response
        rows.append({"input_ids": ids, "labels": [IGNORE_LABEL] * prefix_len + response, "row_id": f"q{i}"})
    return Dataset.from_list(rows)


@pytest.fixture(scope="module")
def proxy():
    return _tiny_model(1)


@pytest.fixture(scope="module")
def student():
    return _tiny_model(2)


@pytest.fixture(scope="module")
def rows():
    return _rows(3, [(3, 6), (4, 2), (2, 9), (5, 4)])


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory, proxy, rows):
    path = tmp_path_factory.mktemp("proxy_cache")
    build_proxy_cache(proxy, rows, str(path), top_k=TOP_K, vocab_size=VOCAB, chunk_size=4, flush_every=1)
    return path


def _reference_stats(model, ids: list[int], labels: list[int]):
    with torch.inference_mode():
        logits = model(input_ids=torch.tensor([ids])).logits[0, :, :VOCAB].float()
    lp = torch.log_softmax(logits, dim=-1)
    pos = [s for s, lab in enumerate(labels) if lab != IGNORE_LABEL]
    lp_pred = lp[[s - 1 for s in pos]]
    targets = torch.tensor([ids[s] for s in pos])
    entropy = -(lp_pred.exp() * lp_pred).sum(-1)
    logp = lp_pred.gather(1, targets[:, None]).squeeze(1)
    return lp_pred, entropy, logp


def test_cache_matches_direct_computation(cache_dir, proxy, rows):
    cache = ProxyCache(str(cache_dir))
    assert len(cache) == len(rows)
    assert cache.top_k == TOP_K and cache.vocab_size == VOCAB
    for row in rows:
        lp_pred, entropy, logp = _reference_stats(proxy, row["input_ids"], row["labels"])
        sl = cache.slice(row_key(row["input_ids"]))
        assert np.allclose(sl["entropy"], entropy.numpy(), atol=1e-5)
        assert np.allclose(sl["logp"], logp.numpy(), atol=1e-5)
        top_lp, top_ids = lp_pred.topk(TOP_K, dim=-1)
        assert np.array_equal(sl["topk_ids"], top_ids.numpy().astype(np.int32))
        assert np.allclose(sl["topk_logp"].astype(np.float32), top_lp.numpy(), atol=2e-3)


def test_cache_resumes_from_progress(tmp_path, proxy, rows):
    path = tmp_path / "partial"
    keys = [row_key(ids) for ids in rows["input_ids"]]
    lengths = [sum(1 for t in labs if t != IGNORE_LABEL) for labs in rows["labels"]]
    writer = ProxyCacheWriter(str(path), keys, lengths, rows["row_id"], top_k=TOP_K, vocab_size=VOCAB)
    n0 = lengths[0]
    writer.write_row(0, np.full(n0, 7.0, np.float32), np.zeros(n0, np.float32), np.zeros((n0, TOP_K), np.int32), np.zeros((n0, TOP_K), np.float16))
    writer.flush()
    writer.mark_done(1)
    writer.close()
    with pytest.raises(AssertionError, match="incomplete"):
        ProxyCache(str(path))

    cache = build_proxy_cache(proxy, rows, str(path), top_k=TOP_K, vocab_size=VOCAB, chunk_size=4, flush_every=1)
    # row 0 was marked done before the resume, so its sentinel values survive; the rest are real
    assert np.all(cache.slice(keys[0])["entropy"] == 7.0)
    _, entropy, _ = _reference_stats(proxy, rows[1]["input_ids"], rows[1]["labels"])
    assert np.allclose(cache.slice(keys[1])["entropy"], entropy.numpy(), atol=1e-5)

    with pytest.raises(AssertionError, match="different dataset"):
        ProxyCacheWriter(str(path), keys[::-1], lengths[::-1], rows["row_id"][::-1], top_k=TOP_K, vocab_size=VOCAB)


def test_structural_mask_marks_think_tokens_and_answer():
    ids = torch.tensor([5, 6, THINK_START, 7, 8, THINK_END, 3, 2])
    labels = torch.tensor([IGNORE_LABEL, IGNORE_LABEL, THINK_START, 7, 8, THINK_END, 3, 2])
    mask = structural_mask(ids, labels, (THINK_START, THINK_END))
    assert mask.tolist() == [False, False, True, False, False, True, True, True]


def _packed(rows: Dataset, budget: int = 32) -> Dataset:
    return pack_dataset(rows, budget=budget, pad_token_id=PAD, with_members=True).with_format("torch")


def test_collator_aligns_cache_entries_with_label_positions(cache_dir, rows):
    cache = ProxyCache(str(cache_dir))
    packed = _packed(rows)
    collator = ProxyAwarePackedCollator(cache, thinking_token_ids=(THINK_START, THINK_END))
    batch = collator([packed[i] for i in range(len(packed))])
    assert "attention_mask" not in batch
    assert batch["tw_topk_ids"].shape == (len(packed), 32, TOP_K)
    assert batch["tw_valid"].sum() == sum(int((torch.tensor(labs) != IGNORE_LABEL).sum()) for labs in rows["labels"])
    assert torch.equal(batch["tw_valid"] & (batch["labels"] == IGNORE_LABEL), torch.zeros_like(batch["tw_valid"]))
    assert torch.equal(batch["tw_structural"] & ~batch["tw_valid"], torch.zeros_like(batch["tw_valid"]))

    for b, block in enumerate(packed):
        start = 0
        for key, length in zip(block["member_keys"].tolist(), block["member_lengths"].tolist()):
            sl = cache.slice(key)
            labeled = (block["labels"][start : start + length] != IGNORE_LABEL).nonzero(as_tuple=True)[0] + start
            assert torch.allclose(batch["tw_entropy"][b, labeled], torch.as_tensor(np.array(sl["entropy"])))
            assert torch.allclose(batch["tw_logp"][b, labeled], torch.as_tensor(np.array(sl["logp"])))
            assert torch.equal(batch["tw_topk_ids"][b, labeled], torch.as_tensor(np.array(sl["topk_ids"], dtype=np.int64)))
            # the tokens after </think> (answer + eos) and <think>/</think> themselves are structural
            ids = block["input_ids"][start : start + length]
            end_pos = int((ids == THINK_END).nonzero()[0]) + start
            assert batch["tw_structural"][b, end_pos:start + length].all()
            assert batch["tw_structural"][b, start + int((ids == THINK_START).nonzero()[0])]
            assert not batch["tw_structural"][b, start + int((ids == THINK_START).nonzero()[0]) + 1 : end_pos].any()
            start += length
    assert collator.missing_members == 0


class _FakeTokenizer:
    """Just enough of PreTrainedTokenizer for DataCollatorForTokenClassification."""

    def __init__(self, padding_side: str):
        self.padding_side = padding_side
        self.pad_token_id = PAD

    def pad(self, encoded_inputs, padding=True, max_length=None, pad_to_multiple_of=None, return_tensors=None):
        length = max(len(f["input_ids"]) for f in encoded_inputs)
        if pad_to_multiple_of:
            length = ((length + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
        out = {"input_ids": [], "attention_mask": []}
        for f in encoded_inputs:
            ids = list(f["input_ids"])
            pad = [PAD] * (length - len(ids))
            mask = [1] * len(ids)
            if self.padding_side == "right":
                out["input_ids"].append(ids + pad)
                out["attention_mask"].append(mask + [0] * len(pad))
            else:
                out["input_ids"].append(pad + ids)
                out["attention_mask"].append([0] * len(pad) + mask)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}


def _padded_features(rows: Dataset) -> list[dict]:
    return [
        {"input_ids": r["input_ids"], "attention_mask": [1] * len(r["input_ids"]), "labels": r["labels"]}
        for r in rows
    ]


@pytest.mark.parametrize("padding_side", ["right", "left"])
def test_padded_collator_aligns_cache_entries(cache_dir, rows, padding_side):
    cache = ProxyCache(str(cache_dir))
    collator = ProxyAwarePaddedCollator(_FakeTokenizer(padding_side), cache, (THINK_START, THINK_END))
    batch = collator(_padded_features(rows))
    bsz, seq = batch["labels"].shape
    assert bsz == len(rows) and seq % 8 == 0
    assert batch["attention_mask"].sum() == sum(len(r["input_ids"]) for r in rows)
    for b, row in enumerate(rows):
        sl = cache.slice(row_key(row["input_ids"]))
        length = len(row["input_ids"])
        start = seq - length if padding_side == "left" else 0
        assert torch.equal(batch["input_ids"][b, start : start + length], torch.tensor(row["input_ids"]))
        labeled = (batch["labels"][b] != IGNORE_LABEL).nonzero(as_tuple=True)[0]
        assert labeled.numel() == len(sl["entropy"])
        assert torch.allclose(batch["tw_entropy"][b, labeled], torch.as_tensor(np.array(sl["entropy"])))
        assert torch.equal(batch["tw_topk_ids"][b, labeled], torch.as_tensor(np.array(sl["topk_ids"], dtype=np.int64)))
        assert batch["tw_valid"][b, labeled].all()
        assert batch["tw_structural"][b].sum() == 4  # <think>, </think>, answer, eos
    assert not (batch["tw_valid"] & (batch["labels"] == IGNORE_LABEL)).any()
    assert collator.missing_members == 0


def test_padded_weighted_loss_matches_packed_formula(cache_dir, student, rows, tmp_path):
    """Same documents, padded instead of packed: loss = sum_i w_i CE_i / N with per-document
    normalisation, attention_mask passed through to the model."""
    cache = ProxyCache(str(cache_dir))
    collator = ProxyAwarePaddedCollator(_FakeTokenizer("right"), cache, (THINK_START, THINK_END))
    batch = collator(_padded_features(rows))
    num_items = torch.tensor(int((batch["labels"] != IGNORE_LABEL).sum()))
    trainer = _trainer(student, tmp_path, TokenWeightingConfig(mode="rho", chunk_size=7), collator)
    loss = trainer.compute_loss(student, dict(batch), num_items_in_batch=num_items)

    with torch.no_grad():
        logits = student(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits[:, :-1].float()
    lp = torch.log_softmax(logits, dim=-1)
    labels = batch["labels"][:, 1:]
    selected = labels != IGNORE_LABEL
    ce = -lp.gather(2, labels.clamp_min(0)[..., None]).squeeze(-1)
    kl = coarse_kl(batch["tw_topk_logp"][:, 1:], lp.gather(2, batch["tw_topk_ids"][:, 1:]))
    raw = raw_weights(
        "rho",
        kl=kl,
        h_p=batch["tw_entropy"][:, 1:],
        h_s=-(lp.exp() * lp).sum(-1),
        logp_p=batch["tw_logp"][:, 1:],
        logp_s=-ce,
        rho_eps=1e-3,
        discord_tau=1.0,
    )
    w = finalize_weights(
        raw,
        selected=selected,
        structural=batch["tw_structural"][:, 1:] & selected,
        valid=batch["tw_valid"][:, 1:] & selected,
        structural_weight=1.0,
        normalize_per_pack=True,
    )
    assert torch.isclose(loss, (w * ce).sum() / num_items, atol=1e-5)
    assert torch.allclose(w.sum(-1), selected.sum(-1).float())


def test_collator_without_cache_or_members_yields_no_valid_positions(rows):
    packed = pack_dataset(rows, budget=32, pad_token_id=PAD).with_format("torch")
    collator = ProxyAwarePackedCollator(None, thinking_token_ids=(THINK_START, THINK_END))
    batch = collator([packed[0]])
    assert not batch["tw_valid"].any()
    assert not batch["tw_structural"].any()
    assert batch["tw_topk_ids"].shape[-1] == 1


def test_collator_strict_rejects_unknown_members(cache_dir, rows):
    cache = ProxyCache(str(cache_dir))
    other = _packed(_rows(99, [(2, 3)]))
    strict = ProxyAwarePackedCollator(cache, thinking_token_ids=(THINK_START, THINK_END), strict=True)
    with pytest.raises(KeyError):
        strict([other[0]])
    lenient = ProxyAwarePackedCollator(cache, thinking_token_ids=(THINK_START, THINK_END), strict=False)
    batch = lenient([other[0]])
    assert not batch["tw_valid"].any()
    assert lenient.missing_members == 1


def _trainer(model, tmp_path, config: TokenWeightingConfig, collator) -> WeightedTokenLossTrainer:
    args = Seq2SeqTrainingArguments(
        output_dir=str(tmp_path / "out"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        report_to="none",
        use_cpu=True,
        remove_unused_columns=False,
    )
    return WeightedTokenLossTrainer(model=model, args=args, data_collator=collator, token_weighting=config)


def _hf_reference(model, batch, num_items):
    out = model(
        input_ids=batch["input_ids"],
        position_ids=batch["position_ids"],
        labels=batch["labels"],
        num_items_in_batch=num_items,
    )
    return out.loss


@pytest.mark.parametrize("chunk_size", [3, 1000])
def test_uniform_loss_and_gradient_match_hf(cache_dir, student, rows, tmp_path, chunk_size):
    cache = ProxyCache(str(cache_dir))
    packed = _packed(rows)
    collator = ProxyAwarePackedCollator(cache, thinking_token_ids=(THINK_START, THINK_END))
    batch = collator([packed[0], packed[1]])
    num_items = torch.tensor(int((batch["labels"] != IGNORE_LABEL).sum()) + 5)  # chunk-level count > batch count

    config = TokenWeightingConfig(mode="uniform", chunk_size=chunk_size)
    trainer = _trainer(student, tmp_path, config, collator)
    assert trainer.model_accepts_loss_kwargs

    student.zero_grad()
    reference = _hf_reference(student, dict(batch), num_items)
    reference.backward()
    ref_grad = student.lm_head.weight.grad.clone()

    student.zero_grad()
    loss = trainer.compute_loss(student, dict(batch), num_items_in_batch=num_items)
    loss.backward()
    got_grad = student.lm_head.weight.grad.clone()

    assert torch.isclose(loss, reference, atol=1e-6)
    assert torch.allclose(got_grad, ref_grad, atol=1e-6)


@pytest.mark.parametrize("mode", ["rho", "kl", "eg", "discord"])
def test_weighted_loss_equals_brute_force(cache_dir, student, rows, tmp_path, mode):
    cache = ProxyCache(str(cache_dir))
    packed = _packed(rows)
    collator = ProxyAwarePackedCollator(cache, thinking_token_ids=(THINK_START, THINK_END))
    batch = collator([packed[i] for i in range(len(packed))])
    num_items = torch.tensor(int((batch["labels"] != IGNORE_LABEL).sum()))
    config = TokenWeightingConfig(mode=mode, chunk_size=5, rho_eps=1e-3, discord_tau=1.0)
    trainer = _trainer(student, tmp_path, config, collator)

    student.zero_grad()
    loss = trainer.compute_loss(student, dict(batch), num_items_in_batch=num_items)
    loss.backward()
    got_grad = student.lm_head.weight.grad.clone()

    # Brute force on the same forward: shift everything by one, weight, normalise per pack.
    with torch.no_grad():
        logits = student(input_ids=batch["input_ids"], position_ids=batch["position_ids"]).logits[:, :-1].float()
    lp = torch.log_softmax(logits, dim=-1)
    labels = batch["labels"][:, 1:]
    selected = labels != IGNORE_LABEL
    targets = labels.clamp_min(0)
    ce = -lp.gather(2, targets[..., None]).squeeze(-1)
    h_s = -(lp.exp() * lp).sum(-1)
    logp_s = -ce
    topk_ids = batch["tw_topk_ids"][:, 1:]
    kl = coarse_kl(batch["tw_topk_logp"][:, 1:], lp.gather(2, topk_ids))
    raw = raw_weights(
        mode,
        kl=kl,
        h_p=batch["tw_entropy"][:, 1:],
        h_s=h_s,
        logp_p=batch["tw_logp"][:, 1:],
        logp_s=logp_s,
        rho_eps=1e-3,
        discord_tau=1.0,
    )
    w = finalize_weights(
        raw,
        selected=selected,
        structural=batch["tw_structural"][:, 1:] & selected,
        valid=batch["tw_valid"][:, 1:] & selected,
        structural_weight=1.0,
        normalize_per_pack=True,
    )
    expected = (w * ce).sum() / num_items
    assert torch.isclose(loss, expected, atol=1e-5)
    assert torch.allclose(w.sum(-1), selected.sum(-1).float())
    assert not torch.allclose(w[selected], torch.ones_like(w[selected]))  # the mode actually reweights

    # Gradient: autograd through the brute-force weighted loss (weights constant) must agree.
    student.zero_grad()
    logits_grad = student(input_ids=batch["input_ids"], position_ids=batch["position_ids"]).logits[:, :-1].float()
    ce_grad = -torch.log_softmax(logits_grad, dim=-1).gather(2, targets[..., None]).squeeze(-1)
    ((w.detach() * ce_grad).sum() / num_items).backward()
    assert torch.allclose(student.lm_head.weight.grad, got_grad, atol=1e-6)


def test_student_statistics_skips_unselected_chunks(student):
    logits = torch.randn(10, VOCAB)
    targets = torch.randint(0, VOCAB, (10,))
    topk_ids = torch.randint(0, VOCAB, (10, 3))
    selected = torch.zeros(10, dtype=torch.bool)
    selected[6:] = True
    logp_s, h_s, topk_logp_s = student_statistics(logits, targets, topk_ids, selected, chunk_size=4)
    assert (logp_s[:4] == 0).all() and (h_s[:4] == 0).all() and (topk_logp_s[:4] == 0).all()
    lp = torch.log_softmax(logits[4:], dim=-1)
    assert torch.allclose(logp_s[4:], lp.gather(1, targets[4:, None]).squeeze(1))
    assert torch.allclose(topk_logp_s[4:], lp.gather(1, topk_ids[4:]))


def test_log_reports_running_token_statistics(cache_dir, student, rows, tmp_path):
    cache = ProxyCache(str(cache_dir))
    packed = _packed(rows)
    collator = ProxyAwarePackedCollator(cache, thinking_token_ids=(THINK_START, THINK_END))
    batch = collator([packed[0]])
    trainer = _trainer(student, tmp_path, TokenWeightingConfig(mode="rho", chunk_size=4), collator)
    trainer.compute_loss(student, dict(batch), num_items_in_batch=torch.tensor(10))
    logs: dict[str, float] = {"loss": 1.0}
    trainer.log(logs)
    assert {"tw/kl", "tw/h_p", "tw/h_s", "tw/rho", "tw/eg", "tw/discord", "tw/weight", "tw/weight_structural"} <= set(logs)
    assert 0 <= logs["tw/rho"] <= 1
    assert logs["tw/missing_frac"] == 0
    assert trainer._tw_sums == {}  # reset after logging


def test_rho1_selection_is_mean_loss_over_selected_tokens(cache_dir, student, rows, tmp_path):
    """mode=excess_loss + select_top_frac: the loss equals the mean CE over the kept content tokens
    (plus the structural tokens at weight 1), i.e. Rho-1's selective objective."""
    cache = ProxyCache(str(cache_dir))
    packed = _packed(rows)
    collator = ProxyAwarePackedCollator(cache, thinking_token_ids=(THINK_START, THINK_END))
    batch = collator([packed[0]])
    num_items = torch.tensor(int((batch["labels"] != IGNORE_LABEL).sum()))
    config = TokenWeightingConfig(mode="excess_loss", chunk_size=5, select_top_frac=0.6)
    trainer = _trainer(student, tmp_path, config, collator)
    loss = trainer.compute_loss(student, dict(batch), num_items_in_batch=num_items)

    with torch.no_grad():
        logits = student(input_ids=batch["input_ids"], position_ids=batch["position_ids"]).logits[:, :-1].float()
    lp = torch.log_softmax(logits, dim=-1)
    labels = batch["labels"][:, 1:]
    selected = labels != IGNORE_LABEL
    ce = -lp.gather(2, labels.clamp_min(0)[..., None]).squeeze(-1)
    structural = batch["tw_structural"][:, 1:] & selected
    content = selected & batch["tw_valid"][:, 1:] & ~structural
    excess = ce - (-batch["tw_logp"][:, 1:])
    n_content = int(content.sum())
    k = max(1, int(np.ceil(0.6 * n_content)))
    order = torch.argsort(torch.where(content, excess, torch.full_like(excess, -1e9))[0], descending=True)
    kept = torch.zeros_like(content)
    kept[0, order[:k]] = True
    assert int(kept.sum()) == k
    # kept content tokens share the content mass equally (n_content / k each), structural stay at 1
    expected = (ce[kept] * (n_content / k)).sum() + ce[structural].sum()
    assert torch.isclose(loss, expected / num_items, atol=1e-5)
