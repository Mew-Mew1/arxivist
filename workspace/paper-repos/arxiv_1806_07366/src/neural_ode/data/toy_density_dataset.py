"""Toy 2D target densities for the CNF experiments (Figures 4-5, Section 4.1).

"We first compare continuous and discrete planar flows at learning to sample
from a known distribution... Figure 4 shows that CNF generally achieves lower
loss" (density matching against "Two Circles" and "Two Moons" targets).

Architecture plan: src/neural_ode/data/toy_density_dataset.py.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


class TwoCirclesDensity:
    """Mixture of two concentric-ish rings, the 'Two Circles' target in Figure 4/5."""

    def __init__(self, noise_std: float = 0.1, radii: tuple = (1.0, 2.5)):
        self.noise_std = noise_std
        self.radii = radii

    def sample(self, n: int, rng: np.random.RandomState | None = None) -> Tensor:
        rng = rng or np.random.RandomState()
        radius_choice = rng.choice(self.radii, size=n)
        angle = rng.uniform(0, 2 * np.pi, size=n)
        x = radius_choice * np.cos(angle) + rng.normal(scale=self.noise_std, size=n)
        y = radius_choice * np.sin(angle) + rng.normal(scale=self.noise_std, size=n)
        return torch.tensor(np.stack([x, y], axis=1), dtype=torch.float32)

    def log_prob(self, z: Tensor) -> Tensor:
        """Unnormalized log-density: mixture of Gaussian rings (used as the target for
        density-matching training, minimizing KL(q(x)||p(x)))."""
        assert z.dim() == 2 and z.shape[1] == 2, f"Expected z of shape [B, 2], got {z.shape}"
        r = torch.norm(z, dim=1)  # [B]
        log_probs = []
        for radius in self.radii:
            log_probs.append(-0.5 * ((r - radius) / self.noise_std) ** 2)
        stacked = torch.stack(log_probs, dim=1)  # [B, num_rings]
        return torch.logsumexp(stacked, dim=1) - np.log(len(self.radii))


class TwoMoonsDensity:
    """The 'Two Moons' target distribution in Figure 4/5."""

    def __init__(self, noise_std: float = 0.1):
        self.noise_std = noise_std

    def sample(self, n: int, rng: np.random.RandomState | None = None) -> Tensor:
        rng = rng or np.random.RandomState()
        n_per_moon = n // 2
        theta1 = rng.uniform(0, np.pi, size=n_per_moon)
        moon1_x = np.cos(theta1)
        moon1_y = np.sin(theta1)
        theta2 = rng.uniform(0, np.pi, size=n - n_per_moon)
        moon2_x = 1 - np.cos(theta2)
        moon2_y = 1 - np.sin(theta2) - 0.5
        x = np.concatenate([moon1_x, moon2_x]) + rng.normal(scale=self.noise_std, size=n)
        y = np.concatenate([moon1_y, moon2_y]) + rng.normal(scale=self.noise_std, size=n)
        return torch.tensor(np.stack([x, y], axis=1), dtype=torch.float32)

    def log_prob(self, z: Tensor) -> Tensor:
        """Approximate log-density via a kernel-density-style Gaussian mixture over a
        fixed reference sample (sufficient for training the density-matching KL loss)."""
        assert z.dim() == 2 and z.shape[1] == 2, f"Expected z of shape [B, 2], got {z.shape}"
        ref = self.sample(2000).to(z.device)
        diffs = z.unsqueeze(1) - ref.unsqueeze(0)  # [B, 2000, 2]
        sq_dists = (diffs**2).sum(dim=-1)  # [B, 2000]
        log_kernel = -0.5 * sq_dists / (self.noise_std**2)
        return torch.logsumexp(log_kernel, dim=1) - np.log(ref.shape[0])
