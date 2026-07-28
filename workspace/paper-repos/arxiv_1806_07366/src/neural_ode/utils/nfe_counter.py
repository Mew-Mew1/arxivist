"""Forward-hook based counter for number of dynamics-function evaluations (NFE).

Reproduces the NFE tracking used to generate Figure 3 in the paper ("Statistics
of a trained ODE-Net", NFE = number of function evaluations).

Architecture plan: src/neural_ode/utils/nfe_counter.py.
"""

from __future__ import annotations

import torch


class NFECounter:
    """Attaches a forward-pre-hook to a module to count calls to it."""

    def __init__(self):
        self._count = 0
        self._handle = None

    def attach(self, module: torch.nn.Module) -> None:
        if self._handle is not None:
            self._handle.remove()

        def _hook(module, inputs):  # noqa: ANN001
            self._count += 1

        self._handle = module.register_forward_pre_hook(_hook)

    def reset(self) -> None:
        self._count = 0

    def count(self) -> int:
        return self._count

    def __repr__(self) -> str:  # noqa: D105
        return f"NFECounter(count={self._count})"
