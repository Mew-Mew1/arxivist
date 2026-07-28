#!/usr/bin/env python
"""Evaluation entrypoint: load a checkpoint and reproduce the relevant paper table/figure.

Usage:
    python evaluate.py --checkpoint checkpoints/best.pt --experiment classification
    python evaluate.py --checkpoint checkpoints/best.pt --experiment latent_ode
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch  # noqa: E402
from torch.utils.data import DataLoader, random_split  # noqa: E402

from neural_ode.data.mnist_dataset import MNISTDataModule  # noqa: E402
from neural_ode.data.spiral_dataset import SpiralDataset  # noqa: E402
from neural_ode.models.cnf import ContinuousNormalizingFlow  # noqa: E402
from neural_ode.models.latent_ode import LatentODEModel  # noqa: E402
from neural_ode.models.odenet_mnist import ODENetClassifier, RKNetClassifier, ResNetClassifier  # noqa: E402
from neural_ode.training.trainer_classification import ClassificationTrainer  # noqa: E402
from neural_ode.training.trainer_latent_ode import LatentODETrainer  # noqa: E402
from neural_ode.utils.config import Config, set_seed  # noqa: E402


def evaluate_classification(config: Config, checkpoint: str) -> None:
    data = MNISTDataModule(root=config.data.mnist_path, num_workers=config.data.num_workers)
    test_loader = data.test_loader(batch_size=config.training.batch_size)
    variant_to_cls = {"resnet": ResNetClassifier, "rknet": RKNetClassifier, "odenet": ODENetClassifier}
    model = variant_to_cls[config.model.variant](hidden_channels=config.model.hidden_channels)
    ckpt = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    trainer = ClassificationTrainer(model, data.train_loader(config.training.batch_size), test_loader, config)
    print(f"Test error: {trainer.evaluate():.4%}  (Table 1 reports 0.42% for ODE-Net)")


def evaluate_latent_ode(config: Config, checkpoint: str, num_observations: int) -> None:
    dataset = SpiralDataset(
        num_trajectories=config.data.spiral_num_trajectories,
        num_timesteps=config.data.spiral_timesteps,
        observation_noise_std=config.data.spiral_observation_noise_std,
        seed=config.training.seed,
    )
    if num_observations < dataset.num_timesteps:
        dataset = dataset.subsample(num_observations, seed=config.training.seed)
    n_train = int(0.8 * len(dataset))
    _, test_set = random_split(dataset, [n_train, len(dataset) - n_train])
    test_loader = DataLoader(test_set, batch_size=min(config.training.batch_size, len(test_set)))

    model = LatentODEModel(
        latent_dim=config.model.latent_dim,
        encoder_hidden_units=config.model.encoder_hidden_units,
        dynamics_hidden_units=config.model.dynamics_hidden_units,
        decoder_hidden_units=config.model.decoder_hidden_units,
        rnn_cell_type=config.model.rnn_cell_type,
    )
    ckpt = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    trainer = LatentODETrainer(model, test_loader, test_loader, config)
    _, t_obs = next(iter(test_loader))
    rmse = trainer.evaluate(t_obs[0])
    print(f"Predictive RMSE (n_obs={num_observations}): {rmse:.4f}  (Table 2 reports 0.1346-0.1642 across n_obs)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a saved model checkpoint")
    parser.add_argument("--experiment", type=str, required=True, choices=["classification", "cnf", "latent_ode"])
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--num-observations", type=int, default=100, help="Only used for --experiment latent_ode")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    set_seed(config.training.seed, deterministic=config.training.deterministic)

    if args.experiment == "classification":
        evaluate_classification(config, args.checkpoint)
    elif args.experiment == "latent_ode":
        evaluate_latent_ode(config, args.checkpoint, args.num_observations)
    elif args.experiment == "cnf":
        raise NotImplementedError(
            "CNF evaluation is qualitative (density plots, Figures 4-5); use inference.py --experiment cnf "
            "to sample from a trained flow and visualize it in notebooks/explore_arxiv_1806_07366.ipynb."
        )


if __name__ == "__main__":
    main()
