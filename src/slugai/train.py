import torch
from torch import nn
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"


def run_train(train_dl: DataLoader, model: nn.Module, loss_fn, optimizer):
    size = len(train_dl.dataset)
    model.train()
    X: torch.Tensor
    y: torch.Tensor
    for batch, (X, y) in enumerate(train_dl):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch*len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def run_test(test_dl: DataLoader, model: nn.Module, loss_fn):
    size = len(test_dl.dataset)
    num_batches = len(test_dl)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in test_dl:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss = test_loss / num_batches
    correct = correct / size
    print(f"Test error: \n Accuracy: {(100*correct):>0.2f}%, Avg loss: {test_loss:>8f} \n")


def run_epochs(n_epochs: int,
               train_dl: DataLoader, test_dl: DataLoader,
               model: nn.Module, loss_fn, optimizer) -> None:
    for i in range(n_epochs):
        print(f"Epoch {i + 1} \n -----------------------")
        run_train(train_dl, model, loss_fn, optimizer)
        run_test(test_dl, model, loss_fn)
    print("Done")
