#!/usr/bin/env python
"""Data setup script for the three experiment families.

- MNIST: publicly available; downloaded automatically by torchvision on first
  use via `MNISTDataModule`, but this script can pre-fetch it.
- Two Circles / Two Moons: synthetic, generated on-the-fly, nothing to download.
- Bi-directional spiral dataset: NOT publicly released by the paper's authors
  (SIR evaluation_protocol: publicly_available=false). This script regenerates
  a statistically equivalent version from `SpiralDataset` rather than
  downloading an original file — see data/README_data.md for details.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


MNIST_RAW_FILES = {
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
}


def mnist_is_present(mnist_dir: Path) -> bool:
    raw_dir = mnist_dir / "MNIST" / "raw"
    return raw_dir.exists() and all((raw_dir / name).exists() for name in MNIST_RAW_FILES)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mnist-path", type=str, default="./data/mnist")
    parser.add_argument("--generate-spirals", action="store_true", help="Also pre-generate the spiral dataset")
    args = parser.parse_args()

    mnist_dir = Path(args.mnist_path)
    if mnist_is_present(mnist_dir):
        print(f"MNIST already present at {mnist_dir}, skipping download.")
    else:
        print(f"Downloading MNIST to {mnist_dir} (~11 MB)...")
        from torchvision import datasets

        mnist_dir.mkdir(parents=True, exist_ok=True)
        datasets.MNIST(root=str(mnist_dir), train=True, download=True)
        datasets.MNIST(root=str(mnist_dir), train=False, download=True)
        print("Done.")

    print("Two Circles / Two Moons: synthetic, generated on-the-fly. Nothing to download.")

    if args.generate_spirals:
        from neural_ode.data.spiral_dataset import SpiralDataset

        print("Generating the synthetic bi-directional spiral dataset (see data/README_data.md)...")
        ds = SpiralDataset(num_trajectories=1000, num_timesteps=100, observation_noise_std=0.1, seed=0)
        print(f"Generated {len(ds)} trajectories of {ds.num_timesteps} timesteps each.")


if __name__ == "__main__":
    main()
