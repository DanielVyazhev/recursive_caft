"""Proxy-vs-untrained-student token diagnostics on the pilot's fixed training subset (first
MAX_ROWS rows). Needs the proxy cache from qwen_3b_proxy_pass.py. Single process, one GPU."""

import json

import torch
from transformers import AutoModelForCausalLM

from core.training.thinking_tokens import new_token_ids
from core.training.token_weighting.diagnostics import summarize, token_diagnostics
from core.training.token_weighting.proxy_cache import ProxyCache
from core.utils.device import DEVICE
from core.utils.logger import logger
from experiments.token_weighting.mmlu.shared import (
    ARTIFACTS_DIR,
    TRAIN_FIXED,
    base_model_path,
    load_tokenizer,
    proxy_cache_path,
    reasoning_adapter,
)

MODEL_NAME = "qwen_3b"
MAX_ROWS = 1024
OUT_DIR = ARTIFACTS_DIR / "diagnostics" / MODEL_NAME


def main():
    tokenizer = load_tokenizer(MODEL_NAME)
    dataset = reasoning_adapter(tokenizer, TRAIN_FIXED).process_dataset()
    cache = ProxyCache(proxy_cache_path(MODEL_NAME, TRAIN_FIXED))
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path(MODEL_NAME), trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(DEVICE)

    think_start, think_end = new_token_ids(tokenizer)
    df = token_diagnostics(model, dataset, cache, (think_start, think_end), max_rows=MAX_ROWS)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_DIR / "tokens.parquet", index=False)
    summary = summarize(df)
    (OUT_DIR / "summary_common_token_diagnostics.json").write_text(json.dumps(summary, indent=2))
    logger.info(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
