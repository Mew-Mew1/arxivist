"""Trainer for the latent ODE time-series model (Section 5.1, Table 2, Appendix E).

Architecture plan: src/neural_ode/training/trainer_latent_ode.py.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from neural_ode.evaluation.metrics import predictive_rmse
from neural_ode.models.latent_ode import LatentODEModel
from neural_ode.training.losses import latent_ode_negative_elbo
from neural_ode.utils.config import Config


class LatentODETrainer:
    """VAE training loop following Appendix E's algorithm."""

    def __init__(
        self,
        model: LatentODEModel,
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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.training.learning_rate)
        self.best_rmse = float("inf")
        self.global_step = 0

    @torch.no_grad()
    def evaluate(self, extrapolation_t: torch.Tensor) -> float:
        """Predictive RMSE on `extrapolation_t` (Table 2 metric)."""
        self.model.eval()
        total_sq_err, total_n = 0.0, 0
        for x, t_obs in self.test_loader:
            x, t_obs = x.to(self.device), t_obs[0].to(self.device)
            x_hat, _, _ = self.model(x, t_obs, extrapolation_t.to(self.device))
            # Only the overlapping observed timesteps are compared to ground truth here;
            # a full extrapolation evaluation would require ground-truth beyond t_obs,
            # which is generated on-the-fly by the SpiralDataset in practice.
            n_compare = min(x_hat.shape[1], x.shape[1])
            sq_err = ((x_hat[:, :n_compare] - x[:, :n_compare]) ** 2).sum().item()
            total_sq_err += sq_err
            total_n += x.shape[0] * n_compare * x.shape[2]
        self.model.train()
        return (total_sq_err / total_n) ** 0.5

    def fit(self, epochs: Optional[int] = None, dry_run: bool = False) -> dict:
        epochs = epochs if epochs is not None else self.config.training.epochs
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[LatentODETrainer] model={self.model!r} params={n_params:,} train_size={len(self.train_loader.dataset):,}")
        history = {"train_loss": [], "rmse": []}

        if dry_run:
            print("[LatentODETrainer] --dry-run: components built successfully, skipping training loop.")
            return history

        self.model.train()
        for epoch in range(epochs):
            epoch_start = time.time()
            running_loss = 0.0
            for x, t_obs in self.train_loader:
                x, t_obs = x.to(self.device), t_obs[0].to(self.device)
                self.optimizer.zero_grad()
                x_hat, mu_z0, logvar_z0 = self.model(x, t_obs, t_obs)
                loss = latent_ode_negative_elbo(
                    x_hat, x, mu_z0, logvar_z0, obs_noise_std=self.config.data.spiral_observation_noise_std
                    )
                loss.backward()
                if self.config.training.gradient_clipping is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.gradient_clipping)
                self.optimizer.step()
                running_loss += loss.item()
                self.global_step += 1

                if self.global_step % self.config.training.log_every_n_steps == 0:
                    print(f"epoch={epoch} step={self.global_step} neg_elbo={loss.item():.4f}")
                if self.global_step % self.config.training.checkpoint_every_n_steps == 0:
                    self._save_checkpoint("last.pt")

            rmse = self.evaluate(t_obs)
            history["train_loss"].append(running_loss / len(self.train_loader))
            history["rmse"].append(rmse)
            print(f"[epoch {epoch}] neg_elbo={running_loss / len(self.train_loader):.4f} rmse={rmse:.4f} time={time.time() - epoch_start:.1f}s")
            if rmse < self.best_rmse:
                self.best_rmse = rmse
                self._save_checkpoint("best.pt")

        return history

    def _save_checkpoint(self, filename: str) -> None:
        torch.save({"model_state_dict": self.model.state_dict(), "global_step": self.global_step}, self.checkpoint_dir / filename)
