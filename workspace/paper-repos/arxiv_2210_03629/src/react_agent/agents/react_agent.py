"""
agents/react_agent.py
======================
Core ReAct orchestration loop: implements the augmented-action-space policy
pi(a_t | c_t) and the Thought <-> Action <-> Observation cycle.

Implements SIR mathematical_spec:
    - "Policy over augmented action space":
          a_t ~ pi(a_t | c_t),  c_t = (o_1, a_1, ..., o_{t-1}, a_{t-1}, o_t)
    - "Augmented action space definition":  Ahat = A union L
    - "Context update rule after a thought":  c_{t+1} = (c_t, ahat_t)

Paper section: Section 2 ("ReAct: Synergizing Reasoning + Acting").
"""

from __future__ import annotations

from dataclasses import dataclass, field

from react_agent.envs.base_environment import BaseEnvironment
from react_agent.llm.action_parser import ActionParser, DomainAction, Thought
from react_agent.llm.llm_client import LLMClient
from react_agent.prompts.prompt_templates import PromptTemplate


@dataclass
class TrajectoryContext:
    """Working memory / context accumulator: c_t from the SIR mathematical_spec.

    c_t = (o_1, a_1, ..., o_{t-1}, a_{t-1}, o_t), serialized as text and
    re-fed into the LLM prompt at every step (Section 2). Thoughts are
    interleaved into this same serialized stream but do not count as
    (action, observation) environment steps.
    """

    lines: list[str] = field(default_factory=list)
    n_env_steps_taken: int = 0

    def append_thought(self, thought: str) -> None:
        """c_{t+1} = (c_t, ahat_t) for a thought ahat_t in L (no observation)."""
        step_idx = self._next_step_index()
        self.lines.append(f"Thought {step_idx}: {thought}")

    def append_action_observation(self, action: str, observation: str) -> None:
        """c_{t+1} appends (a_t, o_{t+1}) for a domain action a_t in A."""
        step_idx = self._next_step_index()
        self.lines.append(f"Action {step_idx}: {action}")
        self.lines.append(f"Observation {step_idx}: {observation}")
        self.n_env_steps_taken += 1

    def _next_step_index(self) -> int:
        # Count only prior Action lines to number steps 1, 2, 3, ... matching
        # the paper's "Thought i / Action i / Observation i" numbering.
        return sum(1 for line in self.lines if line.startswith("Action")) + 1

    def to_prompt_string(self) -> str:
        """Serialize c_t as text, ready to be appended after the few-shot prefix."""
        return "\n".join(self.lines) + ("\n" if self.lines else "")

    def n_env_steps(self) -> int:
        """Number of (action, observation) pairs so far.

        Per configs/config.yaml::evaluation.step_counting_mode
        ("action_observation_pairs_only"), this counts only
        environment-interacting steps, not Thoughts, matching the paper's
        step-cap language in Section 3.2 (SIR implementation_assumptions[6]).
        """
        return self.n_env_steps_taken

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"TrajectoryContext(n_lines={len(self.lines)}, n_env_steps={self.n_env_steps_taken})"


@dataclass
class EpisodeResult:
    """Result of running one full ReAct episode to completion or step-cap."""

    trajectory: TrajectoryContext
    final_answer: str | None
    success: bool | None  # None if unknown (e.g. no ground-truth label available at run time)
    n_steps: int
    finished: bool  # True if the episode ended via finish[...] / env done=True, False if step-capped


class ReactAgent:
    """Implements the ReAct policy: interleaves Thought/Action generation with
    environment interaction, per Section 2 of the paper.

    Args:
        llm_client: Backend used to generate raw text continuations (the
            "Frozen Policy LLM" SIR module).
        prompt_template: Supplies the few-shot prefix for (task_name, method).
        max_steps: Maximum number of (action, observation) environment steps
            before the episode is force-terminated (SIR implementation_
            assumptions[6]: 7 for HotpotQA, 5 for FEVER; environment-specific
            caps for ALFWorld/WebShop).
        thought_mode: "dense" (alternate Thought-Action every step, Section
            3.1) or "sparse" (LLM decides thought frequency itself, Section
            4). See SIR architecture.variants.
        task_name: one of "hotpotqa", "fever", "alfworld", "webshop".
        method: one of "standard", "cot", "act", "react", "react_im"
            (selects which few-shot prefix / thought policy to use).
    """

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_template: PromptTemplate,
        max_steps: int,
        task_name: str,
        method: str = "react",
        thought_mode: str = "dense",
        temperature: float = 0.0,
        max_tokens_per_step: int = 256,
    ) -> None:
        if method not in ("standard", "cot", "act", "react", "react_im"):
            raise ValueError(f"Unknown method: {method!r}")
        if thought_mode not in ("dense", "sparse"):
            raise ValueError(f"thought_mode must be 'dense' or 'sparse', got {thought_mode!r}")

        self.llm_client = llm_client
        self.prompt_template = prompt_template
        self.max_steps = max_steps
        self.task_name = task_name
        self.method = method
        self.thought_mode = thought_mode
        self.temperature = temperature
        self.max_tokens_per_step = max_tokens_per_step
        self._parser = ActionParser()

    def step(self, context: TrajectoryContext) -> Thought | DomainAction:
        """Generate the next Thought or DomainAction given the current context.

        Args:
            context: The TrajectoryContext accumulated so far this episode.

        Returns:
            A parsed Thought or DomainAction (SIR "Thought Generator" /
            "Action Executor" modules).
        """
        prompt = self.prompt_template.format_step(self.task_name, self.method, context)
        raw = self.llm_client.generate(
            prompt=prompt,
            stop_sequences=["\n"],
            temperature=self.temperature,
            max_tokens=self.max_tokens_per_step,
        )
        return self._parser.parse(raw)

    def run_episode(self, task_input: str, env: BaseEnvironment, task_instance: dict | None = None) -> EpisodeResult:
        """Run one full episode: alternate Thought/Action generation with
        environment interaction until finish[...]/done or max_steps.

        Args:
            task_input: The question/claim/instruction text for this episode
                (used only for logging here; the actual o_1 handling is
                environment- and prompt-template-specific).
            env: The BaseEnvironment adapter for this task
                (WikipediaEnvironment / ALFWorldEnvironment / WebShopEnvironment).
            task_instance: Optional environment-specific instance data passed
                through to env.reset().

        Returns:
            An EpisodeResult with the full trajectory and outcome.
        """
        context = TrajectoryContext()
        env.reset(task_instance or {"question": task_input, "claim": task_input})

        # HotpotQA/FEVER: o_1 is the question/claim itself, and must be
        # written into the context, or the LLM never actually sees it.
        if self.task_name == "hotpotqa":
            context.lines.append(f"Question: {task_input}")
        elif self.task_name == "fever":
            context.lines.append(f"Claim: {task_input}")

        final_answer: str | None = None
        finished = False

        while context.n_env_steps() < self.max_steps:
            augmented_action = self.step(context)

            if isinstance(augmented_action, Thought):
                # aihat_t in L: no environment effect, only updates context
                # (SIR mathematical_spec "Context update rule after a thought").
                context.append_thought(augmented_action.text)
                if self.method == "cot" or self.method == "standard":
                    # Pure reasoning-only / no-loop baselines: a single Thought
                    # (or none) is followed directly by a Finish-equivalent
                    # answer extraction rather than further env interaction.
                    break
                continue

           # Validate the action name against this environment's action space
            # before dispatching, so we can give the model a clearer corrective
            # message than a generic environment-level rejection (e.g. when the
            # model hallucinates an action name, or emits free-text instead of
            # a name[args] action entirely). ALFWorld's action space is
            # intentionally open-ended (action_names == []), so skip there.
            valid_names = set(env.action_space().action_names)
            if valid_names and augmented_action.name not in valid_names:
                observation = (
                    f"Invalid action '{augmented_action.name}'. You must use one "
                    f"of these exact action names: {', '.join(sorted(valid_names))}. "
                    f"For example: search[entity], lookup[string], finish[answer]."
                )
                context.append_action_observation(augmented_action.raw_text, observation)
                continue

            # DomainAction: dispatch to the environment.
            observation, done, info = env.step(augmented_action.raw_text)
            context.append_action_observation(augmented_action.raw_text, observation)

            if augmented_action.name == "finish":
                final_answer = augmented_action.args
                finished = True
                break
            if done:
                final_answer = info.get("answer")
                finished = True
                break
        return EpisodeResult(
            trajectory=context,
            final_answer=final_answer,
            success=None,  # populated by the caller once compared against ground truth (evaluation/metrics.py)
            n_steps=context.n_env_steps(),
            finished=finished,
        )

    def reset(self) -> None:
        """No persistent agent-level state to reset; each run_episode() call
        builds a fresh TrajectoryContext. Provided for interface symmetry
        with stateful agent implementations."""
        return None

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"ReactAgent(task_name={self.task_name!r}, method={self.method!r}, "
            f"thought_mode={self.thought_mode!r}, max_steps={self.max_steps})"
        )
