#!/usr/bin/env python
"""Top-level training dispatcher.

The paper covers three distinct experiment families, each with its own
dedicated entrypoint (see architecture plan): `train_classification.py`,
`train_cnf.py`, `train_latent_ode.py`. This file exists to satisfy the
standard ArXivist repo contract (`train.py`) by dispatching to the right
one based on `--experiment`; all actual training logic lives in the
experiment-specific scripts.

Usage:
    python train.py --experiment classification -- --variant odenet
    python train.py --experiment cnf -- --target two_circles --mode density_matching
    python train.py --experiment latent_ode -- --num-observations 100
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

_SCRIPTS = {
    "classification": "train_classification.py",
    "cnf": "train_cnf.py",
    "latent_ode": "train_latent_ode.py",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", type=str, required=True, choices=list(_SCRIPTS.keys()))
    args, remaining = parser.parse_known_args()

    script_path = Path(__file__).parent / _SCRIPTS[args.experiment]
    sys.argv = [str(script_path)] + remaining
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()
