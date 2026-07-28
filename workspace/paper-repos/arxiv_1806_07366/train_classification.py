#!/usr/bin/env python
"""Training entrypoint for the MNIST classification experiment (Table 1, Figure 3).

Usage:
    python train_classification.py --variant odenet --config configs/config.yaml
    python train_classification.py --variant resnet --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from neural_ode.data.mnist_dataset import MNISTDataModule  # noqa: E402
from neural_ode.models.odenet_mnist import ODENetClassifier, RKNetClassifier, ResNetClassifier  # noqa: E402
from neural_ode.training.trainer_classification import ClassificationTrainer  # noqa: E402
from neural_ode.utils.config import Config, set_seed  # noqa: E402


def build_model(variant: str, hidden_channels: int, num_blocks: int):
    if variant == "resnet":
        return ResNetClassifier(hidden_channels=hidden_channels, num_blocks=num_blocks)
    if variant == "rknet":
        return RKNetClassifier(hidden_channels=hidden_channels)
    if variant == "odenet":
        return ODENetClassifier(hidden_channels=hidden_channels)
    raise ValueError(f"Unknown variant '{variant}'. Expected resnet/rknet/odenet.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--variant", type=str, default=None, help="Override config.model.variant: resnet/rknet/odenet")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=None, help="Override config.training.seed")
    parser.add_argument("--epochs", type=int, default=None, help="Override config.training.epochs")
    parser.add_argument("--debug", action="store_true", help="Reduce dataset size/steps for a quick local test")
    parser.add_argument("--dry-run", action="store_true", help="Build all components but do not train")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    if args.variant is not None:
        config.model.variant = args.variant
    if args.seed is not None:
        config.training.seed = args.seed
    set_seed(config.training.seed, deterministic=config.training.deterministic)

    data = MNISTDataModule(root=config.data.mnist_path, num_workers=config.data.num_workers)
    train_loader = data.train_loader(batch_size=config.training.batch_size)
    test_loader = data.test_loader(batch_size=config.training.batch_size)

    if args.debug:
        # Shrink to a handful of batches for a fast smoke test.
        from torch.utils.data import DataLoader, Subset

        train_loader = DataLoader(Subset(data.train_set, range(256)), batch_size=32, shuffle=True)
        test_loader = DataLoader(Subset(data.test_set, range(128)), batch_size=32, shuffle=False)

    model = build_model(config.model.variant, config.model.hidden_channels, config.model.num_residual_blocks)

    if args.resume is not None:
        import torch

        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Resumed from checkpoint: {args.resume}")

    trainer = ClassificationTrainer(model, train_loader, test_loader, config)
    epochs = args.epochs if args.epochs is not None else (1 if args.debug else None)
    trainer.fit(epochs=epochs, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
