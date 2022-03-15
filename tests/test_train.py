import torch.optim
from torch import nn

from slugai.train import run_train, run_test, run_epochs


def test_train(mnist_train_dl, mnist_test_dl, model, cross_entropy_loss):
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    run_train(mnist_train_dl, model, cross_entropy_loss, optimizer)


def test_test(mnist_test_dl, model, cross_entropy_loss):
    run_test(mnist_test_dl, model, cross_entropy_loss)


def test_run_epochs(mnist_train_dl, mnist_test_dl, model, cross_entropy_loss):
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    run_epochs(5, mnist_train_dl, mnist_test_dl, model, cross_entropy_loss, optimizer)