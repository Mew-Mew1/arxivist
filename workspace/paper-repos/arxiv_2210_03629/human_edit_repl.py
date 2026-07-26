#!/usr/bin/env python
"""
human_edit_repl.py
===================
Interactive REPL reproducing the Appendix A.3 human-in-the-loop
thought-editing demo: lets a human inspect and edit ReAct's "thoughts"
mid-trajectory on ALFWorld, then resume the episode.

From the paper: "Figure 5 shows that by simply removing a hallucinating
sentence in Act 17 and adding some hints in Act 23, ReAct can be made to
change its behavior drastically to align with these human thought edits and
succeed in the task" (Appendix A.3).

This script runs the ReAct loop one step at a time, printing each generated
Thought/Action before executing it, and lets the user optionally overwrite
the most recent Thought before continuing -- since, per the paper, editing a
few thoughts is far cheaper than editing a full action sequence or model
parameters.

Paper section: Appendix A.3 ("Human-in-the-Loop Behavior Correction on ALFWorld").
"""

from __future__ import annotations

import argparse

from react_agent.agents.react_agent import ReactAgent, TrajectoryContext
from react_agent.envs.alfworld_env import ALFWorldEnvironment
from react_agent.llm.action_parser import DomainAction, Thought
from react_agent.llm.llm_client import build_llm_client
from react_agent.prompts.prompt_templates import PromptTemplate
from react_agent.utils.config import AgentConfig, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive human-in-the-loop ReAct thought editing on ALFWorld.")
    parser.add_argument("--task-instance", type=str, required=True, help="ALFWorld task instance id.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML.")
    parser.add_argument("--seed", type=int, default=None, help="Override config.seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = AgentConfig.from_yaml(args.config)
    seed = args.seed if args.seed is not None else cfg.seed
    set_seed(seed, deterministic=cfg.deterministic)

    llm_client = build_llm_client(cfg.model)
    prompt_template = PromptTemplate()
    env = ALFWorldEnvironment()

    agent = ReactAgent(
        llm_client=llm_client,
        prompt_template=prompt_template,
        max_steps=50,
        task_name="alfworld",
        method="react",
        thought_mode="sparse",
        temperature=cfg.model.get("temperature_main_prompting", 0.0),
        max_tokens_per_step=cfg.model.get("max_tokens_per_step", 256),
    )

    context = TrajectoryContext()
    env.reset({"task_instance": args.task_instance})

    print("=" * 70)
    print("Human-in-the-loop ReAct (Appendix A.3 reproduction)")
    print("At each Thought step, you may accept it as-is or type a replacement.")
    print("Press Ctrl+C at any time to stop.")
    print("=" * 70)

    while context.n_env_steps() < agent.max_steps:
        augmented_action = agent.step(context)

        if isinstance(augmented_action, Thought):
            print(f"\n[model thought] {augmented_action.text}")
            edited = input("  Accept? Press Enter to keep, or type a replacement thought: ").strip()
            final_thought = edited if edited else augmented_action.text
            context.append_thought(final_thought)
            continue

        assert isinstance(augmented_action, DomainAction)
        print(f"\n[model action] {augmented_action.raw_text}")
        observation, done, info = env.step(augmented_action.raw_text)
        context.append_action_observation(augmented_action.raw_text, observation)
        print(f"[observation]  {observation}")

        if augmented_action.name == "finish" or done:
            print("\nEpisode finished.")
            break

    print("=" * 70)
    print("Full trajectory:")
    print(context.to_prompt_string())
    print("=" * 70)


if __name__ == "__main__":
    main()
