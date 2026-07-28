"""Latent ODE model for irregularly-sampled time series (Section 5, Appendix E).

Pipeline (Figure 6):
    RNN encoder (reversed sequence) -> q(z_t0 | x) -> sample z_t0
        -> ODESolve(z_t0, f, theta_f, t0...tM) -> z_t1...z_tM
        -> decoder -> reconstructed/extrapolated x_hat

Trained as a VAE (Appendix E): maximize
    ELBO = sum_i log p(x_ti | z_ti, theta_x) + log p(z_t0) - log q(z_t0 | {x_ti, ti}, phi)

Architecture plan: src/neural_ode/models/latent_ode.py.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn

from neural_ode.core.ode_block import ODEBlock


class RecognitionRNN(nn.Module):
    """RNN encoder producing q(z_t0 | x_t0...x_tN) (Section 5.1, Appendix E step 1).

    "Our recognition net is an RNN, which consumes the data sequentially
    backwards in time, and outputs q_phi(z0|x1, x2, . . . , xN)."

    WARNING: low-confidence implementation detail (SIR ambiguity, confidence
    0.4): the RNN cell type (vanilla/GRU/LSTM) is not specified in the paper.
    TODO: SIR ambiguity — "RNN cell type for the encoder/decoder ... not
    specified." Defaults to GRU, controlled by `config.model.rnn_cell_type`.
    """

    def __init__(self, obs_dim: int = 2, hidden_units: int = 25, latent_dim: int = 4, cell_type: str = "gru"):
        super().__init__()
        assert cell_type in ("gru", "lstm", "rnn"), f"Unknown cell_type {cell_type}"
        rnn_cls = {"gru": nn.GRU, "lstm": nn.LSTM, "rnn": nn.RNN}[cell_type]
        # +1 input dim for the time delta between consecutive (reversed) observations,
        # a standard trick (and one explicitly used for the RNN *baseline* in Section 5.1:
        # "concatenated with the time difference to the next observation").
        self.rnn = rnn_cls(input_size=obs_dim + 1, hidden_size=hidden_units, batch_first=True)
        self.to_latent_params = nn.Linear(hidden_units, 2 * latent_dim)
        self.latent_dim = latent_dim

    def forward(self, x: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: observations, shape [B, N, obs_dim].
            t: observation times, shape [B, N] or [N].
        Returns:
            (mu_z0, logvar_z0), each shape [B, latent_dim].
        """
        assert x.dim() == 3, f"Expected x of shape [B, N, obs_dim], got {x.shape}"
        B, N, _ = x.shape
        if t.dim() == 1:
            t = t.unsqueeze(0).expand(B, -1)
        dt = torch.diff(t, dim=1, prepend=t[:, :1])  # [B, N]
        x_with_dt = torch.cat([x, dt.unsqueeze(-1)], dim=-1)  # [B, N, obs_dim+1]

        x_reversed = torch.flip(x_with_dt, dims=[1])  # consume backwards in time
        rnn_out = self.rnn(x_reversed)
        h_n = rnn_out[1][0] if isinstance(self.rnn, nn.LSTM) else rnn_out[1]
        h_n = h_n.squeeze(0)  # [B, hidden_units]

        params = self.to_latent_params(h_n)  # [B, 2*latent_dim]
        mu, logvar = params[:, : self.latent_dim], params[:, self.latent_dim :]
        return mu, logvar


class LatentODEFunc(nn.Module):
    """Dynamics function dz/dt = f(z, theta_f) for the latent trajectory (Section 5.1).

    "We parameterize the dynamics function f with a one-hidden-layer network
    with 20 hidden units." Note f is stated to be time-invariant in the paper.
    """

    def __init__(self, latent_dim: int = 4, hidden_units: int = 20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, latent_dim),
        )

    def forward(self, t: Tensor, z: Tensor) -> Tensor:  # noqa: D102 (see class docstring; t unused: time-invariant f)
        assert z.dim() == 2, f"Expected z of shape [B, latent_dim], got {z.shape}"
        return self.net(z)


class Decoder(nn.Module):
    """p(x_ti | z_ti): "another neural network with one hidden layer with 20 hidden units"."""

    def __init__(self, latent_dim: int = 4, obs_dim: int = 2, hidden_units: int = 20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_units),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_units, obs_dim),
        )

    def forward(self, z: Tensor) -> Tensor:
        assert z.dim() == 2, f"Expected z of shape [B, latent_dim], got {z.shape}"
        return self.net(z)


class LatentODEModel(nn.Module):
    """Full latent ODE model tying RecognitionRNN + ODEBlock(LatentODEFunc) + Decoder together."""

    def __init__(
        self,
        obs_dim: int = 2,
        latent_dim: int = 4,
        encoder_hidden_units: int = 25,
        dynamics_hidden_units: int = 20,
        decoder_hidden_units: int = 20,
        rnn_cell_type: str = "gru",
        solver_name: str = "dopri5",
        rtol: float = 1.5e-8,
        atol: float = 1.5e-8,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = RecognitionRNN(obs_dim, encoder_hidden_units, latent_dim, cell_type=rnn_cell_type)
        self.ode_block = ODEBlock(
            LatentODEFunc(latent_dim, dynamics_hidden_units),
            solver_name=solver_name,
            rtol=rtol,
            atol=atol,
            backprop="adjoint",
        )
        self.decoder = Decoder(latent_dim, obs_dim, decoder_hidden_units)

    def forward(self, x: Tensor, t_obs: Tensor, t_query: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            x: observed values, shape [B, N, obs_dim].
            t_obs: observation times, shape [N] (assumed shared across the batch).
            t_query: times to decode at (observed + extrapolated), shape [M].
        Returns:
            (x_hat [B, M, obs_dim], mu_z0 [B, latent_dim], logvar_z0 [B, latent_dim])
        """
        mu_z0, logvar_z0 = self.encoder(x, t_obs)
        std_z0 = torch.exp(0.5 * logvar_z0)
        eps = torch.randn_like(std_z0)
        z_t0 = mu_z0 + eps * std_z0  # reparameterization trick (Appendix E step 2)

        z_traj = self.ode_block(z_t0, integration_time=t_query)  # [M, B, latent_dim] (Appendix E step 3)
        M, B, D = z_traj.shape
        x_hat = self.decoder(z_traj.reshape(M * B, D)).reshape(M, B, -1).transpose(0, 1)  # [B, M, obs_dim]
        return x_hat, mu_z0, logvar_z0

    def __repr__(self) -> str:  # noqa: D105
        return f"LatentODEModel(latent_dim={self.latent_dim})"
