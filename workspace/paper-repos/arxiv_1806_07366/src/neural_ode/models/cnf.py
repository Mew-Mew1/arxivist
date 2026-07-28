"""Continuous Normalizing Flow (CNF): Section 4, Theorem 1, Eqs. 6-10.

Implements the "instantaneous change of variables" (Theorem 1, Eq. 8):

    d(log p(z(t)))/dt = -tr(df/dz(t))

and its planar-flow special case (Eq. 9):

    dz(t)/dt = u h(w^T z(t) + b)
    d(log p(z(t)))/dt = -u^T dh/dz(t)

as well as the linear-cost multi-unit extension (Eq. 10) for M > 1 hidden units.

Architecture plan: src/neural_ode/models/cnf.py.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn

from neural_ode.core.ode_block import ODEBlock


class PlanarCNFDynamics(nn.Module):
    """Augmented dynamics for a planar CNF with M hidden units (Eqs. 9-10).

    Given z(t) in R^D, computes:
        dz/dt        = sum_n u_n * tanh(w_n . z + b_n)                 (Eq. 10, f_n summed)
        d(logp)/dt   = -sum_n tr(d f_n / dz) = -sum_n u_n . dtanh/dz . w_n

    The trace of the Jacobian of a planar unit is computed exactly (not via a
    stochastic trace estimator), which is tractable here because the ambient
    dimension D is small (D=2 for the paper's toy 2D densities), consistent
    with the paper's own low-dimensional experiments in Section 4.1.
    """

    def __init__(self, dim: int = 2, hidden_units: int = 64):
        super().__init__()
        self.dim = dim
        self.hidden_units = hidden_units
        self.u = nn.Parameter(torch.randn(hidden_units, dim) * 0.1)
        self.w = nn.Parameter(torch.randn(hidden_units, dim) * 0.1)
        self.b = nn.Parameter(torch.zeros(hidden_units))

    def forward(self, t: Tensor, states: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        z, _logp = states
        assert z.dim() == 2, f"Expected z of shape [B, D], got {z.shape}"
        B = z.shape[0]

        # h(w_n^T z + b_n), shape [B, M]
        pre_activation = z @ self.w.t() + self.b  # [B, M]
        h = torch.tanh(pre_activation)  # [B, M]
        dz_dt = h @ self.u  # Eq. 10: sum_n u_n h_n(z)  -> [B, D]

        # dtanh/dpre = 1 - tanh^2, so d f_n/dz = u_n (1 - h_n^2) w_n  (outer product)
        # tr(d f_n/dz) = (1 - h_n^2) * (u_n . w_n)   (Eq. 9 special case: -u^T dh/dz)
        dtanh = 1 - h**2  # [B, M]
        uw_dot = (self.u * self.w).sum(dim=1)  # [M]
        trace_per_unit = dtanh * uw_dot.unsqueeze(0)  # [B, M]
        dlogp_dt = -trace_per_unit.sum(dim=1, keepdim=True)  # Eq. 8/10: -sum_n tr(df_n/dz) -> [B, 1]

        return dz_dt, dlogp_dt


class _AugmentedODEFunc(nn.Module):
    """Adapter so `ODEBlock` (which operates on a single tensor) can drive the
    (z, logp) pair required by the instantaneous change-of-variables ODE."""

    def __init__(self, dynamics: PlanarCNFDynamics):
        super().__init__()
        self.dynamics = dynamics
        self.dim = dynamics.dim

    def forward(self, t: Tensor, aug: Tensor) -> Tensor:
        z, logp = aug[:, : self.dim], aug[:, self.dim :]
        dz_dt, dlogp_dt = self.dynamics(t, (z, logp))
        return torch.cat([dz_dt, dlogp_dt], dim=1)


class ContinuousNormalizingFlow(nn.Module):
    """Full CNF model: base distribution N(0, I) <-> data distribution via ODESolve.

    `forward(z0, logp_z0)` pushes samples from the base distribution forward to
    the data distribution ("sampling" direction).
    `forward(..., reverse=True)` runs time backwards, i.e. maps data -> base
    (needed for maximum-likelihood training, Section 4.1: "we can compute the
    reverse transformation for about the same cost as the forward pass").
    """

    def __init__(self, dim: int = 2, hidden_units: int = 64, solver_name: str = "dopri5", rtol: float = 1e-5, atol: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.dynamics = PlanarCNFDynamics(dim=dim, hidden_units=hidden_units)
        self.ode_block = ODEBlock(
            _AugmentedODEFunc(self.dynamics), solver_name=solver_name, rtol=rtol, atol=atol, backprop="adjoint"
        )

    def forward(self, z0: Tensor, logp_z0: Tensor, reverse: bool = False) -> Tuple[Tensor, Tensor]:
        assert z0.dim() == 2 and z0.shape[1] == self.dim, f"Expected z0 of shape [B, {self.dim}], got {z0.shape}"
        assert logp_z0.dim() == 2 and logp_z0.shape[1] == 1, f"Expected logp_z0 of shape [B, 1], got {logp_z0.shape}"
        aug0 = torch.cat([z0, logp_z0], dim=1)
        t = torch.tensor([1.0, 0.0] if reverse else [0.0, 1.0], dtype=z0.dtype, device=z0.device)
        aug1 = self.ode_block(aug0, integration_time=t)
        z1, logp_delta = aug1[:, : self.dim], aug1[:, self.dim :]
        logp_z1 = logp_z0 - logp_delta  # Eq. 6/8 sign convention: log p(z1) = log p(z0) - integral(tr(df/dz))
        return z1, logp_z1

    def sample(self, num_samples: int, device: torch.device = torch.device("cpu")) -> Tensor:
        """Draw samples by reversing the flow from the base N(0, I) distribution."""
        z0 = torch.randn(num_samples, self.dim, device=device)
        logp_z0 = torch.full((num_samples, 1), -0.5 * self.dim * torch.log(torch.tensor(2 * torch.pi)).item(), device=device)
        logp_z0 = logp_z0 - 0.5 * (z0**2).sum(dim=1, keepdim=True)
        z1, _ = self.forward(z0, logp_z0, reverse=False)
        return z1

    def __repr__(self) -> str:  # noqa: D105
        return f"ContinuousNormalizingFlow(dim={self.dim}, hidden_units={self.dynamics.hidden_units})"
