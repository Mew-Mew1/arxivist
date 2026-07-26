"""
envs/alfworld_env.py
=====================
Thin adapter wrapping the external ALFWorld TextWorld engine to the
BaseEnvironment contract.

ALFWorld (Shridhar et al., 2020b) is not re-implemented here -- this module
only translates between react_agent's (observation, done, info) contract and
the `alfworld` package's native TextWorld API, per architecture_plan.json's
"external dependency" risk mitigation (pin version, document data-download
step, degrade gracefully if not installed).

Paper section: Section 4 ("ALFWorld").
"""

from __future__ import annotations

from react_agent.envs.base_environment import ActionSpace, BaseEnvironment


class ALFWorldEnvironment(BaseEnvironment):
    """Adapter around the `alfworld` package's TextWorld-based environment.

    Requires the `alfworld` package and its associated game data to be
    installed/downloaded first -- see data/README_data.md for setup
    instructions (`alfworld-download`).
    """

    def __init__(self, config_path: str = "configs/config.yaml") -> None:
        try:
            import alfworld.agents.environment as alfworld_environment
        except ImportError as e:
            raise ImportError(
                "ALFWorldEnvironment requires the 'alfworld' package and its "
                "downloaded game data. Install via `pip install alfworld` and "
                "run `alfworld-download`. See data/README_data.md for details."
            ) from e

        self._alfworld_environment_module = alfworld_environment
        self._config_path = config_path
        self._env = None  # lazily constructed on first reset(); requires alfworld's own config format
        self._n_env_steps = 0

    def reset(self, task_instance: dict) -> str:
        """Reset to a new ALFWorld task instance.

        Args:
            task_instance: dict with at least a "game_file" or task id
                identifying which of the 134 unseen evaluation games to load
                (SIR evaluation_protocol.datasets, ALFWorld n=134).

        Returns:
            Initial room description text (o_1).
        """
        if self._env is None:
            # NOTE: alfworld's AlfredTWEnv expects its own nested config
            # object (loaded via alfworld.agents.utils.misc). Wiring our
            # unified configs/config.yaml -> alfworld's config format is
            # environment-setup glue code, not part of the ReAct algorithm
            # itself; see data/README_data.md for the expected alfworld
            # config file layout.
            self._env = self._alfworld_environment_module.AlfredTWEnv(
                config=task_instance.get("alfworld_config", {}), train_eval="eval_out_of_distribution"
            )
        self._n_env_steps = 0
        obs, _info = self._env.reset()
        return obs[0] if isinstance(obs, (list, tuple)) else obs

    def step(self, action: str) -> tuple[str, bool, dict]:
        obs, scores, dones, infos = self._env.step([action])
        self._n_env_steps += 1
        observation = obs[0] if isinstance(obs, (list, tuple)) else obs
        done = bool(dones[0]) if isinstance(dones, (list, tuple)) else bool(dones)
        info = {
            "n_env_steps": self._n_env_steps,
            "success": bool(infos.get("won", [False])[0]) if isinstance(infos, dict) else None,
        }
        return observation, done, info

    def action_space(self) -> ActionSpace:
        return ActionSpace(
            action_names=[],  # ALFWorld's action space is open-ended natural language, not a fixed enum
            description="ALFWorld free-text actions, e.g. 'go to drawer 1', "
            "'take X from Y', 'clean X with Y', 'put X in/on Y' (Section 4).",
        )
