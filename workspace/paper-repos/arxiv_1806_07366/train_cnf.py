#!/usr/bin/env python
"""Training entrypoint for the Continuous Normalizing Flow experiments (Figures 4-5).

Usage:
    python train_cnf.py --target two_circles --mode density_matching
    python train_cnf.py --target two_moons --mode maximum_likelihood --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from neural_ode.data.toy_density_dataset import TwoCirclesDensity, TwoMoonsDensity  # noqa: E402
from neural_ode.models.cnf import ContinuousNormalizingFlow  # noqa: E402
from neural_ode.training.trainer_cnf import CNFTrainer  # noqa: E402
from neural_ode.utils.config import Config, set_seed  # noqa: E402


def build_target(name: str):
    if name == "two_circles":
        return TwoCirclesDensity()
    if name == "two_moons":
        return TwoMoonsDensity()
    raise ValueError(f"Unknown target '{name}'. Expected two_circles/two_moons.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=str, default="two_circles", help="One of: two_circles, two_moons")
    parser.add_argument("--mode", type=str, default="density_matching", help="One of: density_matching, maximum_likelihood")
    parser.add_argument("--hidden-units", type=int, default=None, help="Override config.model.cnf_hidden_units (M)")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=None, help="Override config.training.seed")
    parser.add_argument("--debug", action="store_true", help="Reduce iterations for a quick local test")
    parser.add_argument("--dry-run", action="store_true", help="Build all components but do not train")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    if args.seed is not None:
        config.training.seed = args.seed
    set_seed(config.training.seed, deterministic=config.training.deterministic)

    hidden_units = args.hidden_units if args.hidden_units is not None else config.model.cnf_hidden_units
    model = ContinuousNormalizingFlow(
        dim=2, hidden_units=hidden_units, solver_name=config.model.ode_solver,
        rtol=config.model.rtol_density, atol=config.model.atol_density,
    )

    if args.resume is not None:
        import torch

        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Resumed from checkpoint: {args.resume}")

    target = build_target(args.target)
    trainer = CNFTrainer(model, config)

    iterations = 20 if args.debug else config.training.cnf_density_matching_iterations
    if args.mode == "density_matching":
        trainer.fit_density_matching(target, iterations=iterations, dry_run=args.dry_run)
    elif args.mode == "maximum_likelihood":
        trainer.fit_maximum_likelihood(target, iterations=iterations, dry_run=args.dry_run)
    else:
        raise ValueError(f"Unknown mode '{args.mode}'. Expected density_matching/maximum_likelihood.")


if __name__ == "__main__":
    main()
