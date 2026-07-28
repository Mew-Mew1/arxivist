"""ODEBlock: a drop-in nn.Module that replaces a stack of discrete layers.

This is the central abstraction of the paper (Section 1-3): instead of
`h_{t+1} = h_t + f(h_t, theta_t)` (Eq. 1, a discrete ResNet block), ODEBlock
solves `dh(t)/dt = f(h(t), t, theta)` (Eq. 2) with a black-box ODE solver and
returns the result at the requested time(s).

Backpropagation strategy is selectable:
  * "adjoint": use the custom `OdeintAdjointMethod` (O(1) memory, Eqs. 4-5).
  * "direct": backprop directly through the solver's operations (this is what
    makes a network using this block an "RK-Net" rather than an "ODE-Net"
    when combined with `RK4Solver`, per Table 1 / Section 3).

Architecture plan: src/neural_ode/core/ode_block.py.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from neural_ode.core.adjoint import odeint_adjoint
from neural_ode.core.ode_solvers import BaseSolver, build_solver
from neural_ode.utils.nfe_counter import NFECounter


class ODEBlock(nn.Module):
    """Wraps a dynamics network `f(t, h)` into a continuous-depth layer.

    Args:
        dynamics_fn: nn.Module implementing f(t, h) -> dh/dt.
        solver_name: one of euler/rk4/dopri5/dopri8/adaptive_heun/bosh3.
        rtol, atol: solver error tolerances (Section 6, "Setting tolerances").
        backprop: "adjoint" (Eqs. 4-5, O(1) memory) or "direct" (backprop
            through solver ops, used for the RK-Net baseline).
    """

    def __init__(
        self,
        dynamics_fn: nn.Module,
        solver_name: str = "dopri5",
        rtol: float = 1e-3,
        atol: float = 1e-3,
        step_size: float = 0.05,
        backprop: str = "adjoint",
    ):
        super().__init__()
        assert backprop in ("adjoint", "direct"), f"backprop must be 'adjoint' or 'direct', got {backprop}"
        self.dynamics_fn = dynamics_fn
        self.solver: BaseSolver = build_solver(solver_name, rtol=rtol, atol=atol, step_size=step_size)
        self.backprop = backprop
        self._nfe_counter = NFECounter()
        self._nfe_counter.attach(dynamics_fn)

    def forward(self, h0: Tensor, integration_time: Optional[Tensor] = None) -> Tensor:
        """Solve dh/dt = dynamics_fn(t, h) from integration_time[0] to integration_time[-1].

        Args:
            h0: initial state, shape [B, ...].
            integration_time: 1-D tensor of times to evaluate at; defaults to [0, 1]
                (Eq. 3: L(z(t1)) = L(ODESolve(z(t0), f, t0, t1, theta))).

        Returns:
            If len(integration_time) == 2: the state at t1, shape [B, ...].
            Otherwise: the full trajectory, shape [T, B, ...].
        """
        assert h0.dim() >= 1, f"Expected h0 with at least 1 dim (batch), got shape {h0.shape}"
        if integration_time is None:
            integration_time = torch.tensor([0.0, 1.0], dtype=h0.dtype, device=h0.device)
        self._nfe_counter.reset()

        if self.backprop == "adjoint":
            trajectory = odeint_adjoint(h0, self.dynamics_fn, integration_time, self.solver)
        else:
            trajectory = self.solver.integrate(lambda t, z: self.dynamics_fn(t, z), h0, integration_time)

        if integration_time.shape[0] == 2:
            return trajectory[-1]
        return trajectory

    def nfe(self) -> int:
        """Number of dynamics-function evaluations since the last forward call.

        Reproduces the NFE (number of function evaluations) statistics reported
        in Figure 3 of the paper.
        """
        return self._nfe_counter.count()

    def __repr__(self) -> str:  # noqa: D105
        return f"ODEBlock(solver={type(self.solver).__name__}, backprop={self.backprop})"
