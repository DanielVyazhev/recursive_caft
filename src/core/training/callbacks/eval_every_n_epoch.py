import math
from typing import List, Tuple, Optional
from transformers import TrainerCallback, TrainerControl, TrainerState

class EvalEveryNEpochsCallback(TrainerCallback):
    """
    schedule: Optional list of (start_epoch, end_epoch, step) tuples.
    If schedule is None, uses default: [(1,5,1), (6,10,2), (11,30,5)]
    """
    def __init__(
        self,
        schedule: Optional[List[Tuple[int, int, int]]] = None,
        eval_on_train_begin: bool = False,
        eval_on_train_end: bool = False,
    ):
        if schedule is None:
            schedule = [(1, 5, 1), (6, 10, 2), (11, 30, 5)]

        # validate schedule
        for t in schedule:
            if (
                not isinstance(t, tuple)
                or len(t) != 3
                or not all(isinstance(x, int) for x in t)
            ):
                raise ValueError("Each schedule item must be a tuple of 3 integers (start, end, step).")
            start, end, step = t
            if start <= 0 or end < start or step <= 0:
                raise ValueError("Invalid schedule tuple: start>0, end>=start, step>0 required.")

        self.schedule = schedule
        self.eval_on_train_begin = bool(eval_on_train_begin)
        self.eval_on_train_end = bool(eval_on_train_end)

    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        control.should_evaluate = bool(self.eval_on_train_begin)
        return control

    def _should_eval_epoch(self, epoch_num: int) -> bool:
        """Return True if an evaluation should run on this integer epoch number."""
        for start, end, step in self.schedule:
            if start <= epoch_num <= end:
                return ((epoch_num - start) % step) == 0
        return False

    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        epoch_num = int(math.floor(state.epoch)) if state.epoch is not None else 0

        control.should_evaluate = False

        if epoch_num > 0 and self._should_eval_epoch(epoch_num):
            control.should_evaluate = True

        return control

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        control.should_evaluate = bool(self.eval_on_train_end)
        return control
