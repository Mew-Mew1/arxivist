"""Pluggable ODE solver backends.

Implements the "ODESolve" building block used throughout the paper (Eq. 2-3,
Section 3 "Software"). Two families are provided:

  * Fixed-step solvers (`EulerSolver`, `RK4Solver`) reimplemented from scratch,
    useful for the from-scratch `OdeintAdjointMethod` in `adjoint.py` and for
    the RK-Net baseline (direct backprop through a Runge-Kutta integrator).
  * `TorchdiffeqAdaptiveSolver`, a thin wrapper around the community
    `torchdiffeq` package's adaptive dopri5 solver, matching the reference
    implementation the paper's authors released (github.com/rtqichen/torchdiffeq).

Architecture plan: src/neural_ode/core/ode_solvers.py.
"""

from __future__ import annotations

import abc
from typing import Callable

import torch
from torch import Tensor


class BaseSolver(abc.ABC):
    """Abstract base class for all ODE solver backends."""

    @abc.abstractmethod
    def integrate(self, func: Callable[[Tensor, Tensor], Tensor], z0: Tensor, t: Tensor) -> Tensor:
        """Integrate dz/dt = func(t, z) from t[0] to t[-1] (and all intermediate points in t).

        Args:
            func: dynamics function, func(t, z) -> dz/dt, same shape as z.
            z0: initial state, shape [B, ...].
            t: 1-D tensor of times at which to return the solution, shape [T].

        Returns:
            Tensor of shape [T, B, ...] containing z(t) at every requested time.
        """
        raise NotImplementedError


class EulerSolver(BaseSolver):
    """Fixed-step forward Euler solver.

    z(t + h) = z(t) + h * f(t, z(t))

    This is the simplest possible ODE solver, explicitly called out in the paper
    (Section "Adaptive computation": "Euler's method is perhaps the simplest
    method for solving ODEs.").
    """

    def __init__(self, step_size: float = 0.05):
        self.step_size = step_size

    def integrate(self, func: Callable[[Tensor, Tensor], Tensor], z0: Tensor, t: Tensor) -> Tensor:
        assert t.dim() == 1, f"Expected 1-D time tensor, got shape {t.shape}"
        outputs = [z0]
        z = z0
        for i in range(len(t) - 1):
            t0, t1 = t[i], t[i + 1]
            n_steps = max(1, int(torch.ceil((t1 - t0).abs() / self.step_size).item()))
            h = (t1 - t0) / n_steps
            cur_t = t0
            for _ in range(n_steps):
                z = z + h * func(cur_t, z)
                cur_t = cur_t + h
            outputs.append(z)
        return torch.stack(outputs, dim=0)


class RK4Solver(BaseSolver):
    """Fixed-step classical 4th-order Runge-Kutta solver.

    Used, in particular, to implement the "RK-Net" baseline from Table 1: a
    network with the same architecture as ODE-Net, but where gradients are
    backpropagated directly through this integrator's operations (i.e. NOT
    via the adjoint method in `adjoint.py`).
    """

    def __init__(self, step_size: float = 0.05):
        self.step_size = step_size

    def integrate(self, func: Callable[[Tensor, Tensor], Tensor], z0: Tensor, t: Tensor) -> Tensor:
        assert t.dim() == 1, f"Expected 1-D time tensor, got shape {t.shape}"
        outputs = [z0]
        z = z0
        for i in range(len(t) - 1):
            t0, t1 = t[i], t[i + 1]
            n_steps = max(1, int(torch.ceil((t1 - t0).abs() / self.step_size).item()))
            h = (t1 - t0) / n_steps
            cur_t = t0
            for _ in range(n_steps):
                k1 = func(cur_t, z)
                k2 = func(cur_t + h / 2, z + h / 2 * k1)
                k3 = func(cur_t + h / 2, z + h / 2 * k2)
                k4 = func(cur_t + h, z + h * k3)
                z = z + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
                cur_t = cur_t + h
            outputs.append(z)
        return torch.stack(outputs, dim=0)


class TorchdiffeqAdaptiveSolver(BaseSolver):
    """Adaptive-step solver backed by `torchdiffeq` (dopri5 by default).

    This is the recommended default backend (see architecture plan risk
    assessment: hand-rolled adjoint backprop is easy to get subtly wrong,
    whereas `torchdiffeq` is the well-tested community-standard reference
    implementation of this exact paper).
    """

    def __init__(self, method: str = "dopri5", rtol: float = 1e-5, atol: float = 1e-5):
        try:
            import torchdiffeq  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "TorchdiffeqAdaptiveSolver requires the `torchdiffeq` package. "
                "Install it with `pip install torchdiffeq`."
            ) from e
        self.method = method
        self.rtol = rtol
        self.atol = atol

    def integrate(self, func: Callable[[Tensor, Tensor], Tensor], z0: Tensor, t: Tensor) -> Tensor:
        import torchdiffeq

        class _FuncWrapper(torch.nn.Module):
            def __init__(self, f):
                super().__init__()
                self.f = f

            def forward(self, t, z):  # noqa: D102
                return self.f(t, z)

        return torchdiffeq.odeint(
            _FuncWrapper(func), z0, t, rtol=self.rtol, atol=self.atol, method=self.method
        )


def build_solver(name: str, rtol: float = 1e-5, atol: float = 1e-5, step_size: float = 0.05) -> BaseSolver:
    """Factory function used by `ODEBlock` to build a solver from a config string."""
    if name == "euler":
        return EulerSolver(step_size=step_size)
    if name == "rk4":
        return RK4Solver(step_size=step_size)
    if name in ("dopri5", "dopri8", "adaptive_heun", "bosh3"):
        return TorchdiffeqAdaptiveSolver(method=name, rtol=rtol, atol=atol)
    raise ValueError(f"Unknown solver name '{name}'. Expected one of euler/rk4/dopri5/dopri8/adaptive_heun/bosh3.")
