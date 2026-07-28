"""Downsampling stem and dynamics network for the MNIST experiments (Table 1).

Section 3, "Model Architectures": "we experiment with a small residual network
which downsamples the input twice then applies 6 standard residual blocks
(He et al., 2016b), which are replaced by an ODESolve module in the ODE-Net
variant."

WARNING: low-confidence implementation (SIR architecture confidence 0.4 for
this specific module): the exact kernel sizes / strides of the downsampling
stem are not given in the parsed paper text.
TODO: SIR ambiguity — "The exact configuration (kernel sizes, strides, channel
counts) of the downsampling stem ... is not specified." We assume two stride-2
3x3 convolutions (4x total spatial downsampling), configurable via
`config.model.downsample_output_spatial_size` / `hidden_channels`.

Architecture plan: src/neural_ode/models/dynamics_mnist.py.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ConvDownsample(nn.Module):
    """Downsamples a [B, 1, 28, 28] MNIST image twice (Section 3).

    ASSUMED architecture (SIR confidence 0.4, see module docstring): two
    stride-2 3x3 convolutions with GroupNorm + ReLU, matching common
    ResNet-on-MNIST stems of the paper's era.
    """

    def __init__(self, out_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1),  # /2
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1),  # /2 -> /4 total
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        assert x.dim() == 4, f"Expected [B, 1, H, W], got {x.shape}"
        return self.net(x)


class ResidualBlock(nn.Module):
    """A single standard residual block (He et al., 2016b), used by the ResNet baseline.

    Eq. 1: h_{t+1} = h_t + f(h_t, theta_t).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        assert x.dim() == 4, f"Expected [B, C, H, W], got {x.shape}"
        out = self.conv1(torch.relu(self.norm1(x)))
        out = self.conv2(torch.relu(self.norm2(out)))
        return x + out  # Eq. 1


class ConvODEFunc(nn.Module):
    """Dynamics function f(h, t, theta) for the ODE-Net / RK-Net variants (Eq. 2).

    Structurally similar to one `ResidualBlock`, but outputs dh/dt rather than
    an incremental update, and is time-conditioned by concatenating a constant
    time channel (a standard trick for continuous-depth conv nets).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.conv1 = nn.Conv2d(channels + 1, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
        self.conv2 = nn.Conv2d(channels + 1, channels, kernel_size=3, padding=1)

    def _concat_time(self, t: Tensor, h: Tensor) -> Tensor:
        time_channel = torch.ones_like(h[:, :1, :, :]) * t
        return torch.cat([h, time_channel], dim=1)

    def forward(self, t: Tensor, h: Tensor) -> Tensor:  # noqa: D102 (see class docstring)
        assert h.dim() == 4, f"Expected [B, C, H, W], got {h.shape}"
        out = self.conv1(self._concat_time(t, torch.relu(self.norm1(h))))
        out = self.conv2(self._concat_time(t, torch.relu(self.norm2(out))))
        return out  # dh/dt, Eq. 2
