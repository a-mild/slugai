from enum import Enum, auto
from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader

from slugai.callback import Callback, CallbackAggregate
from slugai.phase import Phase, LoopPhase

device = "cuda" if torch.cuda.is_available() else "cpu"


class Trainer:

    def __init__(self, train_dl: DataLoader, test_dl: DataLoader,
                 model: nn.Module, loss_fn, optimizer,
                 callbacks: List[Callback]) -> None:
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.cb = CallbackAggregate(callbacks)
        self.phases = [
            Phase(LoopPhase.TRAIN, train_dl, is_train=True),
            Phase(LoopPhase.VALIDATION, test_dl, is_train=False),
        ]

    def fit(self, n_epochs: int) -> None:
        self.model.to(device)
        self.cb.training_started(phases=self.phases)
        for i in range(1, n_epochs + 1):
            self.cb.epoch_started(epoch=i)
            self._fit_one_cycle()
            self.cb.epoch_ended(phases=self.phases)
        self.cb.training_ended()

    def fit_one_cycle(self):
        self.fit(n_epochs=1)

    def _fit_one_cycle(self):
        model, loss_fn, optimizer = self.model, self.loss_fn, self.optimizer

        for phase in self.phases:
            self.cb.phase_started(phase=phase.type_)
            model.train(phase.is_train)  # sets model.eval() if phase.is_train == False

            for batch, (X, y) in enumerate(phase.dataloader):
                self.cb.batch_started()
                X, y = X.to(device), y.to(device)

                with torch.set_grad_enabled(phase.is_train):
                    pred = model(X)
                    loss = loss_fn(pred, y)

                if phase.is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    self.cb.after_backward_pass(loss=loss.item(), batch=batch)

                self.cb.batch_ended(phase=phase, pred=pred, target=y)

            self.cb.phase_ended(phase=phase.type_)
