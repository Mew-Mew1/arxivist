#!/usr/bin/env python
"""
inference.py
=============
Runs a single ReAct episode on one user-supplied question/claim/task
instance and prints the interleaved Thought/Action/Observation trace,
matching the illustrative trajectories shown in Figure 1 of the paper.

Paper section: Figure 1; Section 2.
"""

from __future__ import annotations

import argparse
import json

from react_agent.agents.react_agent import ReactAgent
from react_agent.envs.wikipedia_env import WikipediaEnvironment
from react_agent.llm.llm_client import build_llm_client
from react_agent.prompts.prompt_templates import PromptTemplate
from react_agent.utils.config import AgentConfig, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single ReAct episode and print its trajectory.")
    parser.add_argument("--task", type=str, required=True, choices=["hotpotqa", "fever", "alfworld", "webshop"])
    parser.add_argument("--input", type=str, required=True, help="Question text, claim text, or task instance id.")
    parser.add_argument("--method", type=str, default="react", choices=["standard", "cot", "act", "react", "react_im"])
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

    if args.task in ("hotpotqa", "fever"):
        env = WikipediaEnvironment()
        max_steps = cfg.routing.get(f"{args.task}_max_react_steps", 7)
        thought_mode = "dense"
    elif args.task == "alfworld":
        from react_agent.envs.alfworld_env import ALFWorldEnvironment

        env = ALFWorldEnvironment()
        max_steps = 50
        thought_mode = "sparse"
    else:
        from react_agent.envs.webshop_env import WebShopEnvironment

        env = WebShopEnvironment()
        max_steps = 30
        thought_mode = "sparse"

    agent = ReactAgent(
        llm_client=llm_client,
        prompt_template=prompt_template,
        max_steps=max_steps,
        task_name=args.task,
        method=args.method,
        thought_mode=thought_mode,
        temperature=cfg.model.get("temperature_main_prompting", 0.0),
        max_tokens_per_step=cfg.model.get("max_tokens_per_step", 256),
    )

    task_instance = {"question": args.input, "claim": args.input} if args.task in ("hotpotqa", "fever") else json.loads(args.input) if args.input.strip().startswith("{") else {"raw_input": args.input}

    result = agent.run_episode(task_input=args.input, env=env, task_instance=task_instance)

    print("=" * 70)
    print(f"Task: {args.task} | Method: {args.method}")
    print(f"Input: {args.input}")
    print("-" * 70)
    print(result.trajectory.to_prompt_string())
    print("-" * 70)
    print(f"Final answer: {result.final_answer}")
    print(f"Finished via terminal action: {result.finished} | Steps: {result.n_steps}")
    print("=" * 70)


if __name__ == "__main__":
    main()
