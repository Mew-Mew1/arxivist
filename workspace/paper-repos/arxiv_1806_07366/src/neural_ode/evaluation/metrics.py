"""Evaluation metrics shared across the three experiment families.

Architecture plan: src/neural_ode/evaluation/metrics.py.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch import Tensor


def test_error(logits: Tensor, targets: Tensor) -> float:
    """Fraction misclassified, matching Table 1's "Test Error" column."""
    assert logits.dim() == 2, f"Expected logits of shape [B, num_classes], got {logits.shape}"
    preds = logits.argmax(dim=1)
    return 1.0 - (preds == targets).float().mean().item()


def predictive_rmse(x_hat: Tensor, x_true: Tensor) -> float:
    """Root-mean-squared error, matching Table 2's "Predictive RMSE" metric."""
    assert x_hat.shape == x_true.shape, f"x_hat shape {x_hat.shape} != x_true shape {x_true.shape}"
    return torch.sqrt(((x_hat - x_true) ** 2).mean()).item()


class MetricTracker:
    """Simple accumulator for logging arbitrary scalar metrics across a run."""

    def __init__(self):
        self._history: Dict[str, list] = {}

    def update(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            self._history.setdefault(key, []).append(value)

    def summary(self) -> Dict[str, Any]:
        return {
            key: (sum(values) / len(values) if values else None)
            for key, values in self._history.items()
        }

    def history(self) -> Dict[str, list]:
        return dict(self._history)

    def __repr__(self) -> str:  # noqa: D105
        return f"MetricTracker(keys={list(self._history.keys())})"
