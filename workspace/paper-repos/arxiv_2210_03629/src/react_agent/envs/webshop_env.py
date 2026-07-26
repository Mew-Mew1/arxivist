"""
envs/webshop_env.py
====================
Thin adapter wrapping the external WebShop environment (Yao et al., 2022) to
the BaseEnvironment contract.

think[...] actions are handled upstream by ActionParser/ReactAgent as
Thoughts (SIR "Action Executor" module note: "think[...] is treated as a
thought, not an action") and never reach this environment's step(); only
search[...], click[...], and equivalent WebShop-native actions are executed
here.

Paper section: Section 4 ("WebShop").
"""

from __future__ import annotations

from react_agent.envs.base_environment import ActionSpace, BaseEnvironment


class WebShopEnvironment(BaseEnvironment):
    """Adapter around the WebShop simulator (github.com/princeton-nlp/webshop).

    Requires the WebShop repository/server to be set up separately -- see
    data/README_data.md for setup instructions.
    """

    def __init__(self, server_url: str = "http://localhost:3000") -> None:
        try:
            import requests
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "WebShopEnvironment requires the 'requests' package to talk "
                "to a running WebShop server. Install via "
                "`pip install -r requirements.txt`."
            ) from e

        self._requests = requests
        self.server_url = server_url
        self._session_id: str | None = None

    def reset(self, task_instance: dict) -> str:
        """Reset to a new WebShop instruction instance.

        Args:
            task_instance: dict with an "instruction_id" or "instruction"
                key identifying which of the 500 test instructions to load
                (SIR evaluation_protocol.datasets, WebShop n=500).

        Returns:
            Initial search page / instruction text (o_1).
        """
        resp = self._requests.post(
            f"{self.server_url}/reset", json={"instruction_id": task_instance.get("instruction_id")}
        )
        resp.raise_for_status()
        payload = resp.json()
        self._session_id = payload.get("session_id")
        return payload.get("observation", "")

    def step(self, action: str) -> tuple[str, bool, dict]:
        if self._session_id is None:
            raise RuntimeError("WebShopEnvironment.step() called before reset().")
        resp = self._requests.post(
            f"{self.server_url}/step", json={"session_id": self._session_id, "action": action}
        )
        resp.raise_for_status()
        payload = resp.json()
        observation = payload.get("observation", "")
        done = bool(payload.get("done", False))
        info = {
            # ASSUMED normalization for the WebShop attribute-coverage Score;
            # see SIR ambiguities[2] and configs/config.yaml::
            # evaluation.webshop_score_normalization (confidence 0.55).
            "score": payload.get("score"),
            "success": payload.get("success"),
        }
        return observation, done, info

    def action_space(self) -> ActionSpace:
        return ActionSpace(
            action_names=["search", "click"],
            description="WebShop actions: search[query], click[button/option] "
            "(Section 4); think[...] is routed as a Thought upstream, not "
            "executed here.",
        )
