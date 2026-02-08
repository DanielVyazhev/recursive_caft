import math
from transformers import TrainerCallback, TrainerControl, TrainerState

class SaveEveryNEpochsCallback(TrainerCallback):
    def __init__(self, n_epochs: int = 4):
        if n_epochs <= 0:
            raise ValueError("n_epochs must be > 0")
        self.n = n_epochs

    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        epoch_num = 0
        if state.epoch is not None:
            epoch_num = int(math.floor(state.epoch))
        if epoch_num > 0 and (epoch_num % self.n == 0):
            control.should_save = True
        return control