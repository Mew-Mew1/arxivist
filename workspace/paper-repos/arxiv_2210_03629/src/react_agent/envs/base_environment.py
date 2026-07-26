"""
envs/base_environment.py
=========================
Abstract environment interface implemented by every task adapter
(WikipediaEnvironment, ALFWorldEnvironment, WebShopEnvironment).

Defines the o_t <-> a_t contract used throughout the SIR: the agent calls
`step(action)` and receives back an observation o_{t+1}, matching the
context-update recurrence c_t = (o_1, a_1, ..., o_{t-1}, a_{t-1}, o_t)
from the SIR mathematical_spec.

Paper section: Section 2 (general agent-environment setup); Section 3.1
(Wikipedia action space); Section 4 (ALFWorld / WebShop).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field


@dataclass
class ActionSpace:
    """Describes the set of valid domain actions A for a given environment."""

    action_names: list[str]
    description: str = ""
    extra: dict = field(default_factory=dict)


class BaseEnvironment(abc.ABC):
    """Abstract base class all task environment adapters must implement."""

    @abc.abstractmethod
    def reset(self, task_instance: dict) -> str:
        """Reset the environment to a new task instance.

        Args:
            task_instance: Task-specific instance data (e.g. a HotpotQA
                question dict, an ALFWorld game id, a WebShop instruction).

        Returns:
            The initial observation o_1 (e.g. the ALFWorld room description,
            or an empty string for HotpotQA/FEVER where o_1 is the question
            itself and is handled by the prompt template).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, action: str) -> tuple[str, bool, dict]:
        """Execute one domain action and return the resulting observation.

        Args:
            action: The raw action string produced by ActionParser, e.g.
                "search[Apple Remote]" or "go to drawer 1".

        Returns:
            A tuple (observation, done, info):
                observation: text describing the result of the action.
                done: whether the episode has terminated (success or failure).
                info: environment-specific metadata (e.g. success flag,
                    WebShop attribute-coverage score).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def action_space(self) -> ActionSpace:
        """Return a description of this environment's valid domain actions."""
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{self.__class__.__name__}()"
