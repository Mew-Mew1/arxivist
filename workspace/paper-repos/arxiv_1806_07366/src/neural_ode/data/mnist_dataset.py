"""MNIST DataModule for the classification experiments (Table 1).

Architecture plan: src/neural_ode/data/mnist_dataset.py.
"""

from __future__ import annotations

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MNISTDataModule:
    """Thin wrapper around torchvision's MNIST with the standard normalize transform."""

    MEAN = (0.1307,)
    STD = (0.3081,)

    def __init__(self, root: str, num_workers: int = 4, download: bool = True):
        self.root = root
        self.num_workers = num_workers
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(self.MEAN, self.STD)]
        )
        self.train_set = datasets.MNIST(root=self.root, train=True, download=download, transform=self.transform)
        self.test_set = datasets.MNIST(root=self.root, train=False, download=download, transform=self.transform)

    def train_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(
            self.train_set, batch_size=batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True
        )

    def test_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(
            self.test_set, batch_size=batch_size, shuffle=False, num_workers=self.num_workers, drop_last=False
        )
