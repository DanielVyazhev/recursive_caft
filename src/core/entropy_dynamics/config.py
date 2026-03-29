"""Configuration for the Entropy Dynamics experiment.

Measures how student entropy on the answer token changes as we feed
incrementally more tokens of the teacher's reasoning chain.
"""

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field
from pydraconf import PydraConfig


class InferenceMode(str, Enum):
    FORCED = "forced"
    CONTINUATION = "continuation"

class StudentModelConfig(BaseModel):
    model_id: str
    label: str 


class EntropyDynamicsConfig(PydraConfig):
    teacher_reasoning_path: str
    out_dir: str = "artifacts/entropy_dynamics"
    students: list[StudentModelConfig]
    mode: InferenceMode = InferenceMode.FORCED
    window_size: int = 32
    max_new_tokens_continuation: int = 2048
    tail_entropy_window: int = 5
    dataset_type: str = "mmlu"
    batch_save_every: int = 50
    temperature: float = 0.7
    do_sample: bool = False

    @property
    def out_path(self) -> Path:
        return Path(self.out_dir)
