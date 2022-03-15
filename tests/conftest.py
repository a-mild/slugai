import pytest
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as t


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


@pytest.fixture
def mnist_training_data():
    return datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=t.ToTensor()
    )


@pytest.fixture
def mnist_test_data():
    return datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=t.ToTensor()
    )


@pytest.fixture
def mnist_train_dl(mnist_training_data):
    return DataLoader(mnist_training_data, batch_size=64)


@pytest.fixture
def mnist_test_dl(mnist_test_data):
    return DataLoader(mnist_test_data, batch_size=64)


@pytest.fixture
def model():
    return NeuralNetwork()


@pytest.fixture
def cross_entropy_loss():
    return nn.CrossEntropyLoss()
