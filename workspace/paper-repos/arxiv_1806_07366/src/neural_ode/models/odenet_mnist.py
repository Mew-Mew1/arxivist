"""ResNet / RK-Net / ODE-Net classifiers for the MNIST experiment (Table 1, Figure 3).

All three variants share the same `ConvDownsample` stem and channel width so
that parameter counts and results are directly comparable, exactly as the
paper does: "We also test a network with the same architecture but where
gradients are backpropagated directly through a Runge-Kutta integrator,
referred to as RK-Net."

Architecture plan: src/neural_ode/models/odenet_mnist.py.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from neural_ode.core.ode_block import ODEBlock
from neural_ode.models.dynamics_mnist import ConvDownsample, ConvODEFunc, ResidualBlock


class _ClassifierHead(nn.Module):
    """Shared classification head: GroupNorm -> ReLU -> global average pool -> linear."""

    def __init__(self, channels: int, num_classes: int = 10):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.norm(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


class ResNetClassifier(nn.Module):
    """Discrete-depth baseline: downsample stem + 6 stacked residual blocks (Eq. 1)."""

    def __init__(self, hidden_channels: int = 64, num_blocks: int = 6, num_classes: int = 10):
        super().__init__()
        self.downsample = ConvDownsample(out_channels=hidden_channels)
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_channels) for _ in range(num_blocks)])
        self.head = _ClassifierHead(hidden_channels, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        assert x.dim() == 4, f"Expected [B, 1, 28, 28], got {x.shape}"
        h = self.downsample(x)
        h = self.blocks(h)
        return self.head(h)

    def __repr__(self) -> str:  # noqa: D105
        return f"ResNetClassifier(blocks={len(self.blocks)})"


class RKNetClassifier(nn.Module):
    """Continuous-depth model with direct backprop through an RK4 integrator (Table 1).

    Same dynamics function as ODENetClassifier, but `ODEBlock(backprop="direct")`
    means gradients flow through every intermediate RK4 stage rather than via
    the adjoint method — this is the "RK-Net" variant.
    """

    def __init__(self, hidden_channels: int = 64, num_classes: int = 10, step_size: float = 0.1):
        super().__init__()
        self.downsample = ConvDownsample(out_channels=hidden_channels)
        self.ode_block = ODEBlock(
            ConvODEFunc(hidden_channels), solver_name="rk4", step_size=step_size, backprop="direct"
        )
        self.head = _ClassifierHead(hidden_channels, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        assert x.dim() == 4, f"Expected [B, 1, 28, 28], got {x.shape}"
        h = self.downsample(x)
        h = self.ode_block(h)
        return self.head(h)

    def __repr__(self) -> str:  # noqa: D105
        return "RKNetClassifier(backprop=direct)"


class ODENetClassifier(nn.Module):
    """Continuous-depth model with O(1)-memory adjoint backprop (Table 1, "ODE-Net").

    Replaces the 6 residual blocks of ResNetClassifier with a single ODEBlock
    call, using the adjoint sensitivity method (Eqs. 4-5) for O(1) memory cost
    with respect to the number of solver evaluations.
    """

    def __init__(
        self,
        hidden_channels: int = 64,
        num_classes: int = 10,
        solver_name: str = "dopri5",
        rtol: float = 1e-3,
        atol: float = 1e-3,
    ):
        super().__init__()
        self.downsample = ConvDownsample(out_channels=hidden_channels)
        self.ode_block = ODEBlock(
            ConvODEFunc(hidden_channels),
            solver_name=solver_name,
            rtol=rtol,
            atol=atol,
            backprop="adjoint",
        )
        self.head = _ClassifierHead(hidden_channels, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        assert x.dim() == 4, f"Expected [B, 1, 28, 28], got {x.shape}"
        h = self.downsample(x)
        h = self.ode_block(h)
        return self.head(h)

    def nfe(self) -> int:
        """Number of dynamics evaluations in the most recent forward pass (Figure 3)."""
        return self.ode_block.nfe()

    def __repr__(self) -> str:  # noqa: D105
        return f"ODENetClassifier(backprop=adjoint, solver={type(self.ode_block.solver).__name__})"
