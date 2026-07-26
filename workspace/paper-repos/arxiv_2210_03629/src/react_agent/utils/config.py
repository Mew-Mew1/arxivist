"""
utils/config.py
================
Config loading, validation, and reproducibility (seeding) utilities.

Implements the reproducibility requirements from ArXivist's code-generation
standard: a single seeding entrypoint that seeds Python's `random`, NumPy, and
PyTorch, plus a deterministic-mode flag mirroring configs/config.yaml's
`deterministic` field.

Paper section: N/A (infrastructure only, ReAct itself does not specify a
seeding protocol).
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Any

import yaml

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a transitive dependency of torch
    np = None  # type: ignore[assignment]

try:
    import torch
except ImportError:  # pragma: no cover - torch only required for the finetuning path
    torch = None  # type: ignore[assignment]


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python's `random`, NumPy, and PyTorch (if installed) for reproducibility.

    Args:
        seed: Integer seed applied to all RNGs.
        deterministic: If True and PyTorch is installed, force deterministic
            algorithms via `torch.use_deterministic_algorithms(True)`. This can
            noticeably slow down the optional finetuning path (Appendix B.1)
            and has no effect on the core API-driven ReAct loop.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if np is not None:
        np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True)


@dataclass
class AgentConfig:
    """Typed wrapper around the YAML config schema defined in
    architecture_plan.json::config_schema.

    This class intentionally keeps the raw nested dict (`raw`) alongside a few
    commonly-accessed convenience fields, rather than fully flattening the
    schema, since downstream modules (ReactAgent, environments, training)
    each read a different subset of the config.
    """

    raw: dict[str, Any] = field(default_factory=dict)

    # --- convenience accessors -------------------------------------------------
    @property
    def seed(self) -> int:
        return int(self.raw.get("seed", 42))

    @property
    def deterministic(self) -> bool:
        return bool(self.raw.get("deterministic", False))

    @property
    def model(self) -> dict[str, Any]:
        return self.raw.get("model", {})

    @property
    def routing(self) -> dict[str, Any]:
        return self.raw.get("routing", {})

    @property
    def training(self) -> dict[str, Any]:
        return self.raw.get("training", {})

    @property
    def data(self) -> dict[str, Any]:
        return self.raw.get("data", {})

    @property
    def evaluation(self) -> dict[str, Any]:
        return self.raw.get("evaluation", {})

    @property
    def paths(self) -> dict[str, Any]:
        return self.raw.get("paths", {})

    @property
    def hardware(self) -> dict[str, Any]:
        return self.raw.get("hardware", {})

    # --- construction ------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str) -> "AgentConfig":
        """Load and validate a config YAML file.

        Args:
            path: Path to a config YAML file (e.g. configs/config.yaml).

        Returns:
            A validated AgentConfig instance.

        Raises:
            FileNotFoundError: if `path` does not exist.
            ValueError: if required top-level sections are missing or a field
                fails a basic sanity check.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Config file not found: {path!r}. All paths must be passed "
                f"explicitly via --config; nothing is hardcoded."
            )

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        cfg = cls(raw=raw)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        """Basic sanity checks on required fields.

        Raises:
            ValueError: with a helpful, specific message if a required field
                is missing or out of range.
        """
        required_sections = ["model", "routing", "training", "data", "evaluation", "paths", "hardware"]
        for section in required_sections:
            if section not in self.raw:
                raise ValueError(
                    f"Config is missing required top-level section '{section}'. "
                    f"See configs/config.yaml for the full expected schema."
                )

        backend = self.model.get("llm_backend")
        if backend not in ("openai", "hf_local"):
            raise ValueError(
                f"model.llm_backend must be one of ['openai', 'hf_local'], got {backend!r}."
            )
        if backend == "hf_local" and not self.model.get("hf_model_name_or_path"):
            raise ValueError(
                "model.llm_backend == 'hf_local' requires model.hf_model_name_or_path to be set."
            )

        temp = self.model.get("temperature_cotsc")
        if temp is not None and not (0.0 <= float(temp) <= 2.0):
            raise ValueError(f"model.temperature_cotsc must be in [0, 2], got {temp}.")

        n_samples = self.model.get("cotsc_n_samples")
        if n_samples is not None and int(n_samples) < 1:
            raise ValueError(f"model.cotsc_n_samples must be >= 1, got {n_samples}.")

    def __repr__(self) -> str:  # pragma: no cover - trivial
        backend = self.model.get("llm_backend")
        model_name = self.model.get("model_name") or self.model.get("hf_model_name_or_path")
        return f"AgentConfig(seed={self.seed}, llm_backend={backend!r}, model={model_name!r})"
