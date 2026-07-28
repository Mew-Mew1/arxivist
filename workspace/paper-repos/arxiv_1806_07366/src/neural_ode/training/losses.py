"""Loss functions for the classification, CNF, and latent-ODE experiments.

Architecture plan: src/neural_ode/training/losses.py.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def classification_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Standard cross-entropy loss for the MNIST experiment (Table 1)."""
    assert logits.dim() == 2, f"Expected logits of shape [B, num_classes], got {logits.shape}"
    return F.cross_entropy(logits, targets)


def kl_density_matching_loss(model_log_prob: Tensor, target_log_prob: Tensor) -> Tensor:
    """KL(q(x) || p(x)) via importance-weighted Monte Carlo estimate (Section 4.1,
    "we minimize KL(q(x)||p(x)) as the loss function where q is the flow model
    and the target density p(.) can be evaluated").

    KL(q||p) = E_q[log q(x) - log p(x)], estimated with samples x ~ q.
    """
    return (model_log_prob - target_log_prob).mean()


def max_likelihood_loss(log_prob_under_model: Tensor) -> Tensor:
    """Negative log-likelihood for the maximum-likelihood CNF training mode
    (Section 4.1: "performing maximum likelihood estimation, which maximizes
    E_p(x)[log q(x)]")."""
    return -log_prob_under_model.mean()


def latent_ode_negative_elbo(
    x_hat: Tensor,
    x_true: Tensor,
    mu_z0: Tensor,
    logvar_z0: Tensor,
    obs_noise_std: float = 0.3,
) -> Tensor:
    """Negative ELBO for the latent ODE VAE (Appendix E):

        ELBO = sum_i log p(x_ti | z_ti, theta_x) + log p(z_t0) - log q(z_t0 | {x_ti,ti}, phi)

    Reconstruction term uses a fixed-variance Gaussian likelihood
    (log p(x|z) = Gaussian log-density), and the prior is p(z_t0) = N(0, I) as
    stated explicitly in Appendix E.
    """
    assert x_hat.shape == x_true.shape, f"x_hat shape {x_hat.shape} != x_true shape {x_true.shape}"
    recon_log_prob = -0.5 * (((x_hat - x_true) / obs_noise_std) ** 2 + torch.log(
        torch.tensor(2 * torch.pi) * obs_noise_std**2
    ))
    recon_log_prob = recon_log_prob.sum(dim=(1, 2))  # sum over time and obs_dim -> [B]

    # KL(q(z0|x) || N(0, I)), closed form for diagonal Gaussians (equivalent to
    # "log p(z_t0) - log q(z_t0|...)" in expectation under the reparameterized sample).
    kl = -0.5 * torch.sum(1 + logvar_z0 - mu_z0.pow(2) - logvar_z0.exp(), dim=1)  # [B]

    elbo = recon_log_prob - kl  # [B]
    return -elbo.mean()
