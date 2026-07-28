"""Config loading and reproducibility utilities.

Implements the "Configuration Schema" section of the ArXivist architecture plan
for arXiv:1806.07366 (Neural Ordinary Differential Equations). All hyperparameters
used anywhere in this repository must be threaded through the `Config` object
produced here rather than hardcoded, per the plan's reproducibility requirements.
"""

from __future__ import annotations

import dataclasses
import random
from typing import Any, Optional

import numpy as np
import torch
import yaml


@dataclasses.dataclass
class ModelConfig:
    variant: str = "odenet"
    hidden_channels: int = 64
    downsample_output_channels: int = 64
    downsample_output_spatial_size: int = 6
    num_residual_blocks: int = 6
    latent_dim: int = 4
    encoder_hidden_units: int = 25
    rnn_cell_type: str = "gru"
    dynamics_hidden_units: int = 20
    decoder_hidden_units: int = 20
    cnf_hidden_units: int = 64
    ode_solver: str = "dopri5"
    rtol_classification: float = 1e-3
    atol_classification: float = 1e-3
    rtol_density: float = 1e-5
    atol_density: float = 1e-5
    rtol_sequence: float = 1.5e-8
    atol_sequence: float = 1.5e-8


@dataclasses.dataclass
class TrainingConfig:
    optimizer: str = "adam"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 128
    epochs: int = 50
    cnf_density_matching_iterations: int = 10000
    nf_baseline_iterations: int = 500000
    gradient_clipping: Optional[float] = None
    mixed_precision: bool = False
    log_every_n_steps: int = 50
    checkpoint_every_n_steps: int = 500
    seed: int = 0
    deterministic: bool = False


@dataclasses.dataclass
class DataConfig:
    mnist_path: str = "./data/mnist"
    spiral_num_trajectories: int = 1000
    spiral_timesteps: int = 100
    spiral_observation_noise_std: float = 0.1
    spiral_subsample_sizes: list = dataclasses.field(default_factory=lambda: [30, 50, 100])
    num_workers: int = 4


@dataclasses.dataclass
class EvaluationConfig:
    metrics: list = dataclasses.field(default_factory=lambda: ["test_error", "kl_divergence", "predictive_rmse", "nfe"])
    mnist_test_split: str = "standard"


@dataclasses.dataclass
class HardwareConfig:
    device: str = "cuda_if_available"
    precision: str = "float32"


@dataclasses.dataclass
class Config:
    """Top-level config aggregating all sub-configs.

    Mirrors the `config_schema` object in `architecture_plan.json` exactly, so
    every field there has a corresponding dataclass field here.
    """

    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    training: TrainingConfig = dataclasses.field(default_factory=TrainingConfig)
    data: DataConfig = dataclasses.field(default_factory=DataConfig)
    evaluation: EvaluationConfig = dataclasses.field(default_factory=EvaluationConfig)
    hardware: HardwareConfig = dataclasses.field(default_factory=HardwareConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
        cfg = cls(
            model=ModelConfig(**raw.get("model", {})),
            training=TrainingConfig(**raw.get("training", {})),
            data=DataConfig(**raw.get("data", {})),
            evaluation=EvaluationConfig(**raw.get("evaluation", {})),
            hardware=HardwareConfig(**raw.get("hardware", {})),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        """Basic sanity checks on config values, raising ValueError with a helpful message."""
        if self.model.variant not in ("resnet", "rknet", "odenet"):
            raise ValueError(
                f"config.model.variant must be one of resnet/rknet/odenet, got '{self.model.variant}'"
            )
        if self.model.rnn_cell_type not in ("gru", "lstm", "rnn"):
            raise ValueError(
                f"config.model.rnn_cell_type must be one of gru/lstm/rnn, got '{self.model.rnn_cell_type}'"
            )
        if self.training.batch_size <= 0:
            raise ValueError("config.training.batch_size must be positive")
        if self.hardware.device not in ("cuda_if_available", "cuda", "cpu"):
            raise ValueError(
                f"config.hardware.device must be one of cuda_if_available/cuda/cpu, got '{self.hardware.device}'"
            )

    def resolved_device(self) -> torch.device:
        if self.hardware.device == "cpu":
            return torch.device("cpu")
        if self.hardware.device == "cuda":
            return torch.device("cuda")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch RNGs (reproducibility requirement).

    Args:
        seed: the RNG seed to use everywhere.
        deterministic: if True, enables `torch.use_deterministic_algorithms(True)`
            and disables cuDNN benchmarking. This can noticeably slow training,
            hence it is opt-in via config rather than always-on.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
