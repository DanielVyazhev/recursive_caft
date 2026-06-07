"""BUG B regression: the resampling train dataset must be sized.

`ResamplingDataset` is a torch IterableDataset. HF's Trainer (transformers 4.52.3)
raises in `Trainer.__init__` ("does not implement __len__, max_steps has to be
specified") when the train dataset has no length and max_steps is unset — see
`has_length` at trainer.py:708. `ResamplingDataset.__len__` returns the sampler's
top_k so that check passes and the normal per-epoch loop runs. These tests assert
the exact condition that gates that error, with no model/GPU.
"""

from transformers.trainer_utils import has_length

from core.dataset_samplers.base_sampler import BaseDatasetSamplerConfig
from core.dataset_samplers.entropy_gain_sampler import EntropyGainSampler
from core.datasets.causal_dataset_adapter import CausalDatasetAdapter
from core.datasets.mmlu.mmlu_reasoning_response_dataset import MMLUReasoningResponseDataset
from core.datasets.qa_dataset import QADatasetConfig
from core.training.resampling_trainer import ResamplingDataset

TOP_K = 7


def _adapter(tokenizer, sampler):
    return CausalDatasetAdapter(
        dataset=MMLUReasoningResponseDataset(
            config=QADatasetConfig(path="sentinel-overridden-at-runtime", dataset_id="t"),
            tokenizer=tokenizer,
        ),
        dataset_sampler=sampler,
    )


def test_len_returns_sampler_top_k(thinking_tokenizer):
    adapter = _adapter(thinking_tokenizer, EntropyGainSampler(BaseDatasetSamplerConfig(top_k=TOP_K)))
    ds = ResamplingDataset(adapter, thinking_tokenizer)
    assert len(ds) == TOP_K


def test_has_length_true(thinking_tokenizer):
    # This is precisely the guard HF evaluates at trainer.py:708.
    adapter = _adapter(thinking_tokenizer, EntropyGainSampler(BaseDatasetSamplerConfig(top_k=TOP_K)))
    ds = ResamplingDataset(adapter, thinking_tokenizer)
    assert has_length(ds) is True


def test_len_does_no_io(thinking_tokenizer):
    # __len__ is evaluated in Trainer.__init__ before any on_epoch_begin sets the
    # path, so it must not read the (possibly absent) resampling parquet.
    adapter = _adapter(thinking_tokenizer, EntropyGainSampler(BaseDatasetSamplerConfig(top_k=TOP_K)))
    ds = ResamplingDataset(adapter, thinking_tokenizer)
    ds.dataset_path = "/nonexistent/path/should/not/be/read.parquet"
    assert len(ds) == TOP_K


def test_len_fallback_without_sampler(thinking_tokenizer, tmp_path, make_resampling_df):
    # No sampler: length falls back to the underlying (path-overridden) dataset size.
    parquet = tmp_path / "ds.parquet"
    make_resampling_df(3).to_parquet(parquet)
    adapter = _adapter(thinking_tokenizer, None)
    ds = ResamplingDataset(adapter, thinking_tokenizer)
    ds.dataset_path = str(parquet)
    assert len(ds) == 3
