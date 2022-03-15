from typing import List

import torch

from slugai.phase import LoopPhase


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


class Progress(Callback):

    def __init__(self, print_every: int = 100):
        self.print_every = print_every
        self.validation_loss = None
        self.correct = None
        self.train_size = None
        self.train_batch_size = None
        self.test_size = None
        self.test_batch_size = None

    def training_started(self, train_size: int, test_size: int,
                         train_batch_size: int, test_num_batches: int):
        self.train_size = train_size
        self.test_size = test_size
        self.train_batch_size = train_batch_size
        self.test_num_batches = test_num_batches

    def training_ended(self, **kwargs):
        print("Done")

    def epoch_started(self, epoch: int):
        print(f"Epoch {epoch} -------------------------")

    def phase_started(self, phase: LoopPhase):
        if phase == LoopPhase.VALIDATION:
            self.validation_loss, self.correct = 0, 0

    def phase_ended(self, phase: LoopPhase):
        if phase == LoopPhase.VALIDATION:
            validation_loss = self.validation_loss / self.test_num_batches
            correct = self.correct / self.test_size
            print(f"Validation error: \n Accuracy: {(100 * correct):>0.2f}%, Avg loss: {validation_loss:>8f} \n")

    def batch_ended(self, phase: LoopPhase, loss: float, pred: torch.Tensor, target):
        if phase == LoopPhase.VALIDATION:
            self.validation_loss += loss
            self.correct += (pred.argmax(1) == target).type(torch.float).sum().item()

    def after_backward_pass(self, loss: float, batch: int):
        if batch % self.print_every == 0:
            current = self.train_batch_size * batch
            print(f"loss: {loss:>7f} [{current:>5d}/{self.train_size:>5d}]")
