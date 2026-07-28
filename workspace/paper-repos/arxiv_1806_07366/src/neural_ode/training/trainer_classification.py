"""Trainer for the MNIST classification experiment (Table 1, Figure 3).

Architecture plan: src/neural_ode/training/trainer_classification.py.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from neural_ode.training.losses import classification_loss
from neural_ode.utils.config import Config


class ClassificationTrainer:
    """Shared training loop for ResNetClassifier / RKNetClassifier / ODENetClassifier."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        config: Config,
        checkpoint_dir: str = "./checkpoints",
    ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.device = config.resolved_device()
        self.model.to(self.device)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if config.training.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer '{config.training.optimizer}'")

        self.best_test_error = 1.0
        self.global_step = 0

    def _print_summary(self) -> None:
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_train = len(self.train_loader.dataset)
        steps_per_epoch = len(self.train_loader)
        print(
            f"[ClassificationTrainer] model={self.model!r} params={n_params:,} "
            f"train_size={n_train:,} steps/epoch={steps_per_epoch}"
        )

    @torch.no_grad()
    def evaluate(self) -> float:
        """Returns test error (fraction misclassified), matching Table 1's metric."""
        self.model.eval()
        correct, total = 0, 0
        for x, y in self.test_loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.shape[0]
        self.model.train()
        return 1.0 - correct / total

    def fit(self, epochs: Optional[int] = None, dry_run: bool = False) -> dict:
        epochs = epochs if epochs is not None else self.config.training.epochs
        self._print_summary()
        history = {"train_loss": [], "test_error": [], "nfe": []}

        if dry_run:
            print("[ClassificationTrainer] --dry-run: components built successfully, skipping training loop.")
            return history

        self.model.train()
        for epoch in range(epochs):
            epoch_start = time.time()
            running_loss = 0.0
            for step, (x, y) in enumerate(self.train_loader):
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(x)
                loss = classification_loss(logits, y)
                loss.backward()
                if self.config.training.gradient_clipping is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.gradient_clipping)
                self.optimizer.step()
                running_loss += loss.item()
                self.global_step += 1

                if self.global_step % self.config.training.log_every_n_steps == 0:
                    print(f"epoch={epoch} step={self.global_step} loss={loss.item():.4f}")
                if self.global_step % self.config.training.checkpoint_every_n_steps == 0:
                    self._save_checkpoint("last.pt")

            test_error = self.evaluate()
            nfe = self.model.nfe() if hasattr(self.model, "nfe") else 0
            history["train_loss"].append(running_loss / len(self.train_loader))
            history["test_error"].append(test_error)
            history["nfe"].append(nfe)
            print(
                f"[epoch {epoch}] loss={running_loss / len(self.train_loader):.4f} "
                f"test_error={test_error:.4%} nfe={nfe} time={time.time() - epoch_start:.1f}s"
            )
            if test_error < self.best_test_error:
                self.best_test_error = test_error
                self._save_checkpoint("best.pt")

        return history

    def _save_checkpoint(self, filename: str) -> None:
        torch.save(
            {"model_state_dict": self.model.state_dict(), "global_step": self.global_step},
            self.checkpoint_dir / filename,
        )
