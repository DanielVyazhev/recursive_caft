"""On-disk cache of teacher-forced proxy statistics, one entry per labeled token.

Layout of a cache directory:

    meta.json      proxy id, top_k, vocab_size, num_rows, num_tokens
    index.json     keys / offsets / lengths / row_ids, in dataset order (written up front)
    progress.json  rows_done -- rows are processed in dataset order, so a restart resumes here
    entropy.npy    float32 [num_tokens]           H_P(t), full-vocab entropy of the proxy (nats)
    logp.npy       float32 [num_tokens]           log p_P(y_t) of the trace token
    topk_ids.npy   int32   [num_tokens, top_k]    proxy's top-K token ids
    topk_logp.npy  float16 [num_tokens, top_k]    their log-probabilities

Entry `j` of a row holds the proxy's prediction for the row's j-th labeled position
(labels != -100), i.e. the distribution produced by the logits one position earlier. Rows
are keyed by `row_key(input_ids)` (core.datasets.sequence_packing), so the cache is tied to
the tokenized content and serves any subset of the dataset it was built on.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from tqdm import tqdm

from core.datasets.sequence_packing import IGNORE_LABEL, row_key
from core.training.token_weighting.token_stats import entropy_from_logp
from core.utils.logger import logger

META_FILE = "meta.json"
INDEX_FILE = "index.json"
PROGRESS_FILE = "progress.json"
ARRAY_SPECS: dict[str, tuple[str, bool]] = {
    # name: (dtype, has top_k dimension)
    "entropy": ("float32", False),
    "logp": ("float32", False),
    "topk_ids": ("int32", True),
    "topk_logp": ("float16", True),
}


def labeled_count(labels: list[int]) -> int:
    return sum(1 for tok in labels if tok != IGNORE_LABEL)


class ProxyCacheWriter:
    def __init__(
        self,
        path: str,
        keys: list[int],
        lengths: list[int],
        row_ids: list[str],
        top_k: int,
        vocab_size: int,
        meta: dict[str, Any] | None = None,
    ):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.top_k = top_k
        self.vocab_size = vocab_size
        offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64) if lengths else np.zeros(0, np.int64)
        self.offsets = offsets.tolist()
        self.lengths = list(lengths)
        self.num_tokens = int(sum(lengths))

        index = {"keys": list(keys), "offsets": self.offsets, "lengths": self.lengths, "row_ids": list(row_ids)}
        full_meta = {
            "top_k": top_k,
            "vocab_size": vocab_size,
            "num_rows": len(keys),
            "num_tokens": self.num_tokens,
            **(meta or {}),
        }
        index_path = self.path / INDEX_FILE
        meta_path = self.path / META_FILE
        if index_path.exists():
            existing = json.loads(index_path.read_text())
            assert existing == index, f"Existing cache at {self.path} was built for a different dataset"
            existing_meta = json.loads(meta_path.read_text())
            for k in ("top_k", "vocab_size", "num_rows", "num_tokens"):
                assert existing_meta[k] == full_meta[k], f"Existing cache meta mismatch on {k}"
        else:
            index_path.write_text(json.dumps(index))
            meta_path.write_text(json.dumps(full_meta, indent=2))

        self.arrays: dict[str, np.memmap] = {}
        for name, (dtype, has_k) in ARRAY_SPECS.items():
            shape = (self.num_tokens, top_k) if has_k else (self.num_tokens,)
            file = self.path / f"{name}.npy"
            if file.exists():
                arr = np.load(file, mmap_mode="r+")
                assert arr.shape == shape and arr.dtype == np.dtype(dtype), f"{file} has shape {arr.shape}/{arr.dtype}"
            else:
                arr = np.lib.format.open_memmap(file, mode="w+", dtype=dtype, shape=shape)
            self.arrays[name] = arr

        progress_path = self.path / PROGRESS_FILE
        self.rows_done = json.loads(progress_path.read_text())["rows_done"] if progress_path.exists() else 0

    def write_row(
        self,
        row_index: int,
        entropy: np.ndarray,
        logp: np.ndarray,
        topk_ids: np.ndarray,
        topk_logp: np.ndarray,
    ) -> None:
        n = self.lengths[row_index]
        off = self.offsets[row_index]
        assert entropy.shape == (n,) and logp.shape == (n,), f"row {row_index}: expected {n} entries"
        assert topk_ids.shape == (n, self.top_k) and topk_logp.shape == (n, self.top_k)
        self.arrays["entropy"][off : off + n] = entropy
        self.arrays["logp"][off : off + n] = logp
        self.arrays["topk_ids"][off : off + n] = topk_ids
        self.arrays["topk_logp"][off : off + n] = topk_logp

    def mark_done(self, rows_done: int) -> None:
        self.rows_done = rows_done
        (self.path / PROGRESS_FILE).write_text(json.dumps({"rows_done": rows_done}))

    def flush(self) -> None:
        for arr in self.arrays.values():
            arr.flush()

    def close(self) -> None:
        self.flush()
        self.arrays = {}


class ProxyCache:
    """Read-only view of a finished cache. Lazily opens the memmaps so the object can be
    pickled into DataLoader workers."""

    def __init__(self, path: str):
        self.path = Path(path)
        meta = json.loads((self.path / META_FILE).read_text())
        index = json.loads((self.path / INDEX_FILE).read_text())
        progress_path = self.path / PROGRESS_FILE
        rows_done = json.loads(progress_path.read_text())["rows_done"] if progress_path.exists() else 0
        assert rows_done == meta["num_rows"], (
            f"Proxy cache at {self.path} is incomplete: {rows_done}/{meta['num_rows']} rows done"
        )
        self.meta = meta
        self.top_k: int = meta["top_k"]
        self.vocab_size: int = meta["vocab_size"]
        self._index: dict[int, tuple[int, int]] = {
            int(k): (int(off), int(n)) for k, off, n in zip(index["keys"], index["offsets"], index["lengths"])
        }
        self._arrays: dict[str, np.ndarray] | None = None

    def __len__(self) -> int:
        return len(self._index)

    def __contains__(self, key: int) -> bool:
        return int(key) in self._index

    def lookup(self, key: int) -> tuple[int, int] | None:
        return self._index.get(int(key))

    @property
    def arrays(self) -> dict[str, np.ndarray]:
        if self._arrays is None:
            self._arrays = {name: np.load(self.path / f"{name}.npy", mmap_mode="r") for name in ARRAY_SPECS}
        return self._arrays

    def slice(self, key: int) -> dict[str, np.ndarray]:
        """Views over the entries of one row (no copy)."""
        found = self.lookup(key)
        if found is None:
            raise KeyError(f"row key {key} not in proxy cache {self.path}")
        off, n = found
        return {name: arr[off : off + n] for name, arr in self.arrays.items()}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_arrays"] = None
        return state


@torch.inference_mode()
def build_proxy_cache(
    model,
    dataset: Dataset,
    path: str,
    top_k: int,
    vocab_size: int,
    chunk_size: int = 2048,
    flush_every: int = 20,
    meta: dict[str, Any] | None = None,
) -> ProxyCache:
    """Teacher-force `model` over every row of a tokenized `dataset` (columns input_ids / labels /
    row_id as produced by CausalDatasetAdapter) and store the per-token statistics.

    One row per forward; `logits_to_keep` restricts the lm_head to the positions that predict a
    labeled token, and the log-softmax runs in fp32 over `chunk_size` positions at a time. Logits
    are sliced to `vocab_size` (the tokenizer length, dropping the embedding padding rows) before
    normalization. Rows are processed in order and progress is persisted so the pass can resume.
    """
    model.eval()
    keys = [row_key(ids) for ids in dataset["input_ids"]]
    lengths = [labeled_count(labels) for labels in dataset["labels"]]
    row_ids = [str(r) for r in dataset["row_id"]]
    writer = ProxyCacheWriter(path, keys, lengths, row_ids, top_k=top_k, vocab_size=vocab_size, meta=meta)
    if writer.rows_done:
        logger.info(f"Proxy cache: resuming at row {writer.rows_done}/{len(dataset)}")

    device = next(model.parameters()).device
    for i in tqdm(range(writer.rows_done, len(dataset)), desc="proxy pass", initial=writer.rows_done, total=len(dataset)):
        row = dataset[i]
        input_ids = torch.tensor(row["input_ids"], dtype=torch.long)
        labels = torch.tensor(row["labels"], dtype=torch.long)
        target_pos = (labels != IGNORE_LABEL).nonzero(as_tuple=True)[0]
        assert target_pos.numel() == lengths[i]
        assert target_pos.numel() == 0 or int(target_pos.min()) >= 1, "a labeled token must have a predecessor"
        if target_pos.numel() == 0:
            continue
        targets = input_ids[target_pos]
        # Kept on CPU: works whether the model is on one device or sharded with device_map.
        logit_pos = target_pos - 1

        out = model(input_ids=input_ids[None].to(device), logits_to_keep=logit_pos, use_cache=False)
        logits = out.logits[0]  # [n, V'] on the lm_head device
        assert logits.shape[0] == targets.numel()
        assert logits.shape[1] >= vocab_size, f"model vocab {logits.shape[1]} < requested {vocab_size}"

        ent_parts, logp_parts, ids_parts, tlp_parts = [], [], [], []
        for a in range(0, logits.shape[0], chunk_size):
            lp = torch.log_softmax(logits[a : a + chunk_size, :vocab_size].float(), dim=-1)
            y = targets[a : a + chunk_size].to(lp.device)
            ent_parts.append(entropy_from_logp(lp).cpu())
            logp_parts.append(lp.gather(1, y[:, None]).squeeze(1).cpu())
            top_lp, top_ids = lp.topk(top_k, dim=-1)
            ids_parts.append(top_ids.to(torch.int32).cpu())
            tlp_parts.append(top_lp.to(torch.float16).cpu())
        del out, logits

        writer.write_row(
            i,
            torch.cat(ent_parts).numpy(),
            torch.cat(logp_parts).numpy(),
            torch.cat(ids_parts).numpy(),
            torch.cat(tlp_parts).numpy(),
        )
        if (i + 1) % flush_every == 0 or i + 1 == len(dataset):
            writer.flush()
            writer.mark_done(i + 1)

    writer.close()
    logger.info(f"Proxy cache complete: {len(dataset)} rows, {writer.num_tokens} tokens at {path}")
    return ProxyCache(path)
