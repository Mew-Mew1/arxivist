#!/usr/bin/env python
"""Single-sample inference entrypoint.

Usage:
    python inference.py --checkpoint checkpoints/best.pt --experiment classification --input path/to/image.png
    python inference.py --checkpoint checkpoints/best.pt --experiment cnf
    python inference.py --checkpoint checkpoints/best.pt --experiment latent_ode --input path/to/trajectory.npy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from neural_ode.models.cnf import ContinuousNormalizingFlow  # noqa: E402
from neural_ode.models.latent_ode import LatentODEModel  # noqa: E402
from neural_ode.models.odenet_mnist import ODENetClassifier, RKNetClassifier, ResNetClassifier  # noqa: E402
from neural_ode.utils.config import Config  # noqa: E402


def infer_classification(config: Config, checkpoint: str, input_path: str) -> None:
    from PIL import Image
    from torchvision import transforms

    variant_to_cls = {"resnet": ResNetClassifier, "rknet": RKNetClassifier, "odenet": ODENetClassifier}
    model = variant_to_cls[config.model.variant](hidden_channels=config.model.hidden_channels)
    ckpt = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    transform = transforms.Compose([transforms.Grayscale(), transforms.Resize((28, 28)), transforms.ToTensor()])
    img = Image.open(input_path)
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
    pred = logits.argmax(dim=1).item()
    print(f"Predicted digit: {pred}  (logits: {logits.squeeze().tolist()})")


def infer_cnf(config: Config, checkpoint: str, num_samples: int = 16) -> None:
    model = ContinuousNormalizingFlow(dim=2, hidden_units=config.model.cnf_hidden_units, solver_name=config.model.ode_solver)
    ckpt = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    with torch.no_grad():
        samples = model.sample(num_samples)
    print(f"Sampled {num_samples} points from the trained CNF (shape {tuple(samples.shape)}):")
    print(samples)


def infer_latent_ode(config: Config, checkpoint: str, input_path: str) -> None:
    model = LatentODEModel(
        latent_dim=config.model.latent_dim,
        encoder_hidden_units=config.model.encoder_hidden_units,
        dynamics_hidden_units=config.model.dynamics_hidden_units,
        decoder_hidden_units=config.model.decoder_hidden_units,
        rnn_cell_type=config.model.rnn_cell_type,
    )
    ckpt = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    data = np.load(input_path)  # expects shape [N_obs, 2]
    x = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
    t_obs = torch.linspace(0, 1, x.shape[1])
    t_query = torch.linspace(0, 1.5, int(x.shape[1] * 1.5))  # extrapolate 50% beyond observed range
    with torch.no_grad():
        x_hat, _, _ = model(x, t_obs, t_query)
    print(f"Extrapolated trajectory shape: {tuple(x_hat.shape)} (observed {x.shape[1]} -> queried {t_query.shape[0]} points)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a saved model checkpoint")
    parser.add_argument("--experiment", type=str, required=True, choices=["classification", "cnf", "latent_ode"])
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--input", type=str, default=None, help="Path to a single input sample (image/.npy), if applicable")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)

    if args.experiment == "classification":
        if args.input is None:
            raise ValueError("--input (path to an image) is required for --experiment classification")
        infer_classification(config, args.checkpoint, args.input)
    elif args.experiment == "cnf":
        infer_cnf(config, args.checkpoint)
    elif args.experiment == "latent_ode":
        if args.input is None:
            raise ValueError("--input (path to a .npy trajectory) is required for --experiment latent_ode")
        infer_latent_ode(config, args.checkpoint, args.input)


if __name__ == "__main__":
    main()
