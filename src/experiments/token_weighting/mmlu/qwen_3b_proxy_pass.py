"""One teacher-forced pass of the proxy over the Qwen-3B-tokenized traces of the pilot's
training subset. Writes the per-token cache every weighted arm reads.

Single process, one GPU. Qwen2.5-14B-Instruct in bf16 takes ~30 GB of the 40 GB card; the
lm_head only runs on the labeled positions and the log-softmax goes in fp32 chunks, so the
pass stays under the limit (set RECURSIVE_CAFT_PROXY_MODEL to a 7B to go lighter or to a
local path). Resumable: rerun to continue from the last flushed row. Switch PROXY_DATASET to
TRAIN_MINUS_DEV once a per-epoch re-selection arm needs the whole pool cached.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.training.thinking_tokens import (
    mean_init_new_rows,
    new_token_ids,
    setup_thinking_tokens,
)
from core.training.token_weighting.proxy_cache import build_proxy_cache
from core.utils.device import DEVICE_MAP
from core.utils.logger import logger
from experiments.token_weighting.mmlu.shared import (
    PROXY_MODEL_ID,
    PROXY_TOP_K,
    TRAIN_FIXED,
    load_tokenizer,
    proxy_cache_path,
    reasoning_adapter,
)

MODEL_NAME = "qwen_3b"
PROXY_DATASET = TRAIN_FIXED
CHUNK_SIZE = 1024


def main():
    tokenizer = load_tokenizer(MODEL_NAME)
    dataset = reasoning_adapter(tokenizer, PROXY_DATASET).process_dataset()
    logger.info(f"Proxy pass: {PROXY_MODEL_ID} over {len(dataset)} rows of {PROXY_DATASET}")

    proxy_tokenizer = AutoTokenizer.from_pretrained(PROXY_MODEL_ID, trust_remote_code=True)
    setup_thinking_tokens(proxy_tokenizer)
    # The cache stores token ids as produced by the student tokenizer; the proxy must read
    # them with the same vocabulary, thinking tokens included.
    assert proxy_tokenizer.get_vocab() == tokenizer.get_vocab(), "proxy and student tokenizers differ"
    assert new_token_ids(proxy_tokenizer) == new_token_ids(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        PROXY_MODEL_ID, trust_remote_code=True, device_map=DEVICE_MAP, torch_dtype=torch.bfloat16
    )
    new_ids = new_token_ids(tokenizer)
    rows = model.get_input_embeddings().weight.shape[0]
    if rows < max(new_ids) + 1:
        model.resize_token_embeddings(max(new_ids) + 1, pad_to_multiple_of=64)
    # Qwen2.5's embedding table is padded past the tokenizer, so the <think>/</think> ids
    # already have rows; give them the mean-row init the student starts from.
    mean_init_new_rows(model, new_ids)

    build_proxy_cache(
        model,
        dataset,
        proxy_cache_path(MODEL_NAME, PROXY_DATASET),
        top_k=PROXY_TOP_K,
        vocab_size=len(tokenizer),
        chunk_size=CHUNK_SIZE,
        meta={"proxy_model": PROXY_MODEL_ID, "student": MODEL_NAME, "dataset": PROXY_DATASET},
    )


if __name__ == "__main__":
    main()
