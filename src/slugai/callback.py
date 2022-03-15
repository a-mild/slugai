from collections import defaultdict
from typing import List, Dict

import torch
from tqdm import tqdm

from slugai.phase import LoopPhase, Phase


class Callback:

    def training_started(self, **kwargs):
        ...

    def training_ended(self, **kwargs):
        ...

    def epoch_started(self, **kwargs):
        ...

    def epoch_ended(self, **kwargs):
        ...

    def phase_started(self, **kwargs):
        ...

    def phase_ended(self, **kwargs):
        ...

    def batch_started(self, **kwargs):
        ...

    def batch_ended(self, **kwargs):
        ...

    def after_backward_pass(self, **kwargs):
        ...


class CallbackAggregate(Callback):

    def __init__(self, cbs: List[Callback]) -> None:
        self.cbs = cbs

    def training_started(self, **kwargs):
        self("training_started", **kwargs)

    def training_ended(self, **kwargs):
        self("training_ended", **kwargs)

    def epoch_started(self, **kwargs):
        self("epoch_started", **kwargs)

    def epoch_ended(self, **kwargs):
        self("epoch_ended", **kwargs)

    def phase_started(self, **kwargs):
        self("phase_started", **kwargs)

    def phase_ended(self, **kwargs):
        self("phase_ended", **kwargs)

    def batch_started(self, **kwargs):
        self("batch_started", **kwargs)

    def batch_ended(self, **kwargs):
        self("batch_ended", **kwargs)

    def after_backward_pass(self, **kwargs):
        self("after_backward_pass", **kwargs)

    def __call__(self, method_name, **kwargs) -> None:
        for cb in self.cbs:
            method = getattr(cb, method_name, None)
            if method is None:
                continue
            method(**kwargs)


class ProgressBar(Callback):

    def __init__(self, print_every: int = 100):
        self.print_every = print_every
        self.bars: Dict[LoopPhase, tqdm] = {}

    def training_started(self, phases: List[Phase]):
        for phase in phases:
            self.bars[phase.type_] = tqdm(total=len(phase.dataloader), desc=str(phase.type_))

    def training_ended(self, **kwargs):
        print("Done")

    def epoch_started(self, epoch: int):
        print(f"Epoch {epoch} -------------------------")

    def epoch_ended(self, **kwargs):
        for bar in self.bars.values():
            bar.n = 0
            bar.refresh()

    def batch_ended(self, phase: Phase, pred, target, **kwargs):
        bar: tqdm = self.bars[phase.type_]
        bar.update(pred.size(0))


class Accuracy(Callback):

    def epoch_started(self, **kwargs):
        """Reset counts"""
        self.n_samples = defaultdict(int)
        self.correct = defaultdict(int)

    def batch_ended(self, phase: Phase, pred: torch.Tensor, target):
        self.n_samples[phase.type_] += pred.size(0)
        self.correct[phase.type_] += (pred.argmax(1) == target).type(torch.float).sum().item()

    def epoch_ended(self, phases: List[Phase]):
        for phase in phases:
            accuracy = self.correct[phase.type_] / self.n_samples[phase.type_]
            print(f"Validation error: \n Accuracy: {100*accuracy:>0.2f}% \n")
