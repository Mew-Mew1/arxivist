#!/usr/bin/env python
"""Training entrypoint for the latent ODE time-series model (Table 2, Figures 8-10).

Usage:
    python train_latent_ode.py --num-observations 100
    python train_latent_ode.py --num-observations 30 --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from torch.utils.data import DataLoader, random_split  # noqa: E402

from neural_ode.data.spiral_dataset import SpiralDataset  # noqa: E402
from neural_ode.models.latent_ode import LatentODEModel  # noqa: E402
from neural_ode.training.trainer_latent_ode import LatentODETrainer  # noqa: E402
from neural_ode.utils.config import Config, set_seed  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-observations", type=int, default=100, help="Sub-sampled points per trajectory (30/50/100, Table 2)")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=None, help="Override config.training.seed")
    parser.add_argument("--epochs", type=int, default=None, help="Override config.training.epochs")
    parser.add_argument("--debug", action="store_true", help="Reduce dataset size/steps for a quick local test")
    parser.add_argument("--dry-run", action="store_true", help="Build all components but do not train")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    if args.seed is not None:
        config.training.seed = args.seed
    set_seed(config.training.seed, deterministic=config.training.deterministic)

    full_dataset = SpiralDataset(
        num_trajectories=config.data.spiral_num_trajectories if not args.debug else 32,
        num_timesteps=config.data.spiral_timesteps,
        observation_noise_std=config.data.spiral_observation_noise_std,
        seed=config.training.seed,
    )
    if args.num_observations < full_dataset.num_timesteps:
        full_dataset = full_dataset.subsample(args.num_observations, seed=config.training.seed)

    n_train = int(0.8 * len(full_dataset))
    n_test = len(full_dataset) - n_train
    train_set, test_set = random_split(full_dataset, [n_train, n_test])
    train_loader = DataLoader(train_set, batch_size=min(config.training.batch_size, len(train_set)), shuffle=True)
    test_loader = DataLoader(test_set, batch_size=min(config.training.batch_size, len(test_set)), shuffle=False)

    model = LatentODEModel(
        obs_dim=2,
        latent_dim=config.model.latent_dim,
        encoder_hidden_units=config.model.encoder_hidden_units,
        dynamics_hidden_units=config.model.dynamics_hidden_units,
        decoder_hidden_units=config.model.decoder_hidden_units,
        rnn_cell_type=config.model.rnn_cell_type,
        solver_name=config.model.ode_solver,
        rtol=config.model.rtol_sequence,
        atol=config.model.atol_sequence,
    )

    if args.resume is not None:
        import torch

        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Resumed from checkpoint: {args.resume}")

    trainer = LatentODETrainer(model, train_loader, test_loader, config)
    epochs = args.epochs if args.epochs is not None else (1 if args.debug else None)
    trainer.fit(epochs=epochs, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
