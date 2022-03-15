import torch.optim

from slugai.train import run_epochs


def test_run_epochs(mnist_train_dl, mnist_test_dl, model, cross_entropy_loss):
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    run_epochs(5, mnist_train_dl, mnist_test_dl, model, cross_entropy_loss, optimizer)
