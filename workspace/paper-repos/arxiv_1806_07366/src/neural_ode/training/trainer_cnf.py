"""Trainer for the Continuous Normalizing Flow experiments (Section 4.1, Figures 4-5).

Architecture plan: src/neural_ode/training/trainer_cnf.py.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from neural_ode.models.cnf import ContinuousNormalizingFlow
from neural_ode.training.losses import kl_density_matching_loss, max_likelihood_loss
from neural_ode.utils.config import Config


class CNFTrainer:
    """Two training modes reproducing Section 4.1's two experiments."""

    def __init__(self, model: ContinuousNormalizingFlow, config: Config, checkpoint_dir: str = "./checkpoints"):
        self.model = model
        self.config = config
        self.device = config.resolved_device()
        self.model.to(self.device)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # Section 4.1: "we train for 10,000 iterations using Adam" — explicit for CNF (SIR confidence 0.95).
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.training.learning_rate)

    def _base_log_prob(self, z0: torch.Tensor) -> torch.Tensor:
        dim = z0.shape[1]
        return (-0.5 * (z0**2).sum(dim=1, keepdim=True) - 0.5 * dim * torch.log(torch.tensor(2 * torch.pi)))

    def fit_density_matching(self, target, iterations: int = 10000, batch_size: int = 512, dry_run: bool = False) -> dict:
        """Section 4.1 "Density matching": minimize KL(q(x)||p(x)) where q is the CNF
        pushed forward from the base distribution and p is a known target density
        (Two Circles / Two Moons)."""
        history = {"loss": []}
        if dry_run:
            print("[CNFTrainer] --dry-run: components built successfully, skipping training loop.")
            return history

        self.model.train()
        start = time.time()
        for it in range(iterations):
            z0 = torch.randn(batch_size, self.model.dim, device=self.device)
            logp_z0 = self._base_log_prob(z0)
            z1, logp_z1 = self.model(z0, logp_z0, reverse=False)
            target_log_prob = target.log_prob(z1)
            loss = kl_density_matching_loss(logp_z1.squeeze(-1), target_log_prob)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            history["loss"].append(loss.item())

            if (it + 1) % self.config.training.log_every_n_steps == 0:
                print(f"[density_matching] iter={it + 1}/{iterations} loss={loss.item():.4f}")
            if (it + 1) % self.config.training.checkpoint_every_n_steps == 0:
                self._save_checkpoint("last.pt")
        print(f"[CNFTrainer] density matching finished in {time.time() - start:.1f}s")
        return history

    def fit_maximum_likelihood(self, target, iterations: int, batch_size: int = 512, dry_run: bool = False) -> dict:
        """Section 4.1 "Maximum Likelihood Training": maximize E_p(x)[log q(x)] by
        running the flow in reverse (data -> base) and evaluating the induced density."""
        history = {"loss": []}
        if dry_run:
            print("[CNFTrainer] --dry-run: components built successfully, skipping training loop.")
            return history

        self.model.train()
        start = time.time()
        for it in range(iterations):
            x = target.sample(batch_size).to(self.device)
            logp_x_placeholder = torch.zeros(batch_size, 1, device=self.device)  # accumulator starts at 0
            z0, logp_z0_under_model = self.model(x, logp_x_placeholder, reverse=True)
            base_log_prob = self._base_log_prob(z0)
            log_q_x = base_log_prob - logp_z0_under_model  # change of variables composed both ways
            loss = max_likelihood_loss(log_q_x)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            history["loss"].append(loss.item())

            if (it + 1) % self.config.training.log_every_n_steps == 0:
                print(f"[maximum_likelihood] iter={it + 1}/{iterations} loss={loss.item():.4f}")
            if (it + 1) % self.config.training.checkpoint_every_n_steps == 0:
                self._save_checkpoint("last.pt")
        print(f"[CNFTrainer] maximum likelihood finished in {time.time() - start:.1f}s")
        return history

    def _save_checkpoint(self, filename: str) -> None:
        torch.save({"model_state_dict": self.model.state_dict()}, self.checkpoint_dir / filename)
