"""Proxy-vs-student token diagnostics on the pilot's fixed training subset (first MAX_ROWS rows).
Needs the proxy cache from qwen_3b_proxy_pass.py. Single process, one GPU.

    python qwen_3b_diagnostics.py                      # untrained student -> diagnostics/qwen_3b/
    python qwen_3b_diagnostics.py <checkpoint dir>     # a training checkpoint (LoRA + think rows)
                                                       # -> diagnostics/qwen_3b/<arm>__<run>__<checkpoint>/
"""

import json
import sys
from pathlib import Path

from core.training.thinking_tokens import new_token_ids
from core.training.token_weighting.diagnostics import summarize, token_diagnostics
from core.training.token_weighting.proxy_cache import ProxyCache
from core.utils.logger import logger
from experiments.token_weighting.mmlu.shared import (
    ARTIFACTS_DIR,
    TRAIN_FIXED,
    load_student,
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
    checkpoint = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None
    model = load_student(MODEL_NAME, checkpoint)
    out_dir = (
        OUT_DIR
        if checkpoint is None
        else OUT_DIR
        / f"{checkpoint.parent.parent.name}__{checkpoint.parent.name}__{checkpoint.name}"
    )

    think_start, think_end = new_token_ids(tokenizer)
    df = token_diagnostics(
        model, dataset, cache, (think_start, think_end), max_rows=MAX_ROWS
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "tokens.parquet", index=False)
    summary = summarize(df)
    (out_dir / "summary_common_token_diagnostics.json").write_text(
        json.dumps(summary, indent=2)
    )
    logger.info(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
