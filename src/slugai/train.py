import torch
from torch import nn
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"


class Phase:

    def __init__(self, dataloader: DataLoader, is_train: bool = True) -> None:
        self.dataloader = dataloader
        self.is_train = is_train


def run_one_epoch(train_phase: Phase, test_phase: Phase, model, loss_fn, optimizer) -> None:
    for phase in [train_phase, test_phase]:
        size = len(phase.dataloader.dataset)
        if phase.is_train:
            model.train()
        else:
            test_loss, correct = 0, 0
            model.eval()

        for batch, (X, y) in enumerate(phase.dataloader):
            X, y = X.to(device), y.to(device)

            with torch.set_grad_enabled(phase.is_train):
                pred = model(X)
                loss = loss_fn(pred, y)

            if phase.is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if batch % 100 == 0:
                    loss, current = loss.item(), batch * len(X)
                    print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
            else:
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        if not phase.is_train:
            num_batches = len(phase.dataloader)
            test_loss = test_loss / num_batches
            correct = correct / size
            print(f"Test error: \n Accuracy: {(100 * correct):>0.2f}%, Avg loss: {test_loss:>8f} \n")


def run_epochs(n_epochs: int,
               train_dl: DataLoader, test_dl: DataLoader,
               model: nn.Module, loss_fn, optimizer) -> None:
    train_phase = Phase(train_dl, is_train=True)
    test_phase = Phase(test_dl, is_train=False)
    for i in range(n_epochs):
        run_one_epoch(train_phase, test_phase, model, loss_fn, optimizer)
    print("Done")
