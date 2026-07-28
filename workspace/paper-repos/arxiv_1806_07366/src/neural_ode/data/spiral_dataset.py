"""Synthetic bi-directional spiral dataset (Section 5.1).

"We generated a dataset of 1000 2-dimensional spirals, each starting at a
different point, sampled at 100 equally-spaced timesteps. The dataset
contains two types of spirals: half are clockwise while the other half
counter-clockwise. To make the task more realistic, we add gaussian noise to
the observations."

Architecture plan: src/neural_ode/data/spiral_dataset.py.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


def _generate_spiral(num_timesteps: int, clockwise: bool, rng: np.random.RandomState) -> np.ndarray:
    t = np.linspace(0, 4 * np.pi, num_timesteps)
    start_radius = rng.uniform(0.5, 1.5)
    growth = rng.uniform(0.1, 0.3)
    radius = start_radius + growth * t
    angle = t if not clockwise else -t
    angle = angle + rng.uniform(0, 2 * np.pi)  # random starting point (paper: "each starting at a different point")
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    return np.stack([x, y], axis=1)  # [T, 2]


class SpiralDataset(Dataset):
    """1000 bi-directional 2D spirals, 100 timesteps each, with Gaussian observation noise.

    ASSUMED (SIR: qualitative-only spec): observation_noise_std defaults to
    0.1 (config.data.spiral_observation_noise_std) since the paper states
    noise is added but does not give a numeric value.
    """

    def __init__(
        self,
        num_trajectories: int = 1000,
        num_timesteps: int = 100,
        observation_noise_std: float = 0.1,
        seed: int = 0,
        _data: Optional[np.ndarray] = None,
        _t: Optional[np.ndarray] = None,
    ):
        self.num_timesteps = num_timesteps
        self.observation_noise_std = observation_noise_std
        if _data is not None:
            self.data = _data
            self.t = _t
            return
        rng = np.random.RandomState(seed)
        clean = np.stack(
            [
                _generate_spiral(num_timesteps, clockwise=(i % 2 == 0), rng=rng)
                for i in range(num_trajectories)
            ],
            axis=0,
        )  # [N, T, 2]
        noise = rng.normal(scale=observation_noise_std, size=clean.shape)
        self.data = (clean + noise).astype(np.float32)  # [N, T, 2]
        self.t = np.linspace(0, 1, num_timesteps).astype(np.float32)  # [T]

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return torch.from_numpy(self.data[idx]), torch.from_numpy(self.t)

    def subsample(self, n_obs: int, seed: int = 0) -> "SpiralDataset":
        """Randomly retain `n_obs` of the `num_timesteps` points per trajectory (Table 2:
        predictive RMSE reported for n_obs in {30, 50, 100})."""
        assert 0 < n_obs <= self.num_timesteps, f"n_obs must be in (0, {self.num_timesteps}], got {n_obs}"
        rng = np.random.RandomState(seed)
        idx = np.sort(rng.choice(self.num_timesteps, size=n_obs, replace=False))
        sub_data = self.data[:, idx, :]
        sub_t = self.t[idx]
        new_ds = SpiralDataset(
            num_trajectories=0,
            num_timesteps=n_obs,
            observation_noise_std=self.observation_noise_std,
            _data=sub_data,
            _t=sub_t,
        )
        return new_ds

    def __repr__(self) -> str:  # noqa: D105
        return f"SpiralDataset(n={len(self)}, timesteps={self.num_timesteps})"
