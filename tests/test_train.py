import torch.optim

from slugai.callback import Progress
from slugai.train import Trainer


def test_trainer(mnist_train_dl, mnist_test_dl, model, cross_entropy_loss):
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    callbacks = [Progress()]
    trainer = Trainer(mnist_train_dl, mnist_test_dl, model, cross_entropy_loss, optimizer,
                      callbacks)
    trainer.fit(n_epochs=5)
