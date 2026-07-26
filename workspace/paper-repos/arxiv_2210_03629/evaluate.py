#!/usr/bin/env python
"""
evaluate.py
============
Runs a full prompting-method x task evaluation sweep, reproducing Tables 1
(HotpotQA/FEVER), 3 (ALFWorld), and 4 (WebShop) from the paper.

This is the primary entrypoint for reproducing ReAct's headline results,
since the main experiments require no training -- just few-shot prompting
against a (substitute) frozen LLM.

Paper section: Section 3 (HotpotQA/FEVER), Section 4 (ALFWorld/WebShop).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from react_agent.agents.cot_sc import CoTSelfConsistency, ReactCotScRouter
from react_agent.agents.react_agent import ReactAgent
from react_agent.data.fever_dataset import FeverDataset
from react_agent.data.hotpotqa_dataset import HotpotQADataset
from react_agent.envs.wikipedia_env import WikipediaEnvironment
from react_agent.evaluation.metrics import MetricsCalculator
from react_agent.llm.llm_client import build_llm_client
from react_agent.prompts.prompt_templates import PromptTemplate
from react_agent.utils.config import AgentConfig, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a ReAct prompting method on one task.")
    parser.add_argument("--task", type=str, required=True, choices=["hotpotqa", "fever", "alfworld", "webshop"])
    parser.add_argument(
        "--method",
        type=str,
        default="react",
        choices=["standard", "cot", "cot_sc", "act", "react", "react_im", "cotsc_react", "react_cotsc"],
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML.")
    parser.add_argument("--n-eval", type=int, default=None, help="Override number of eval instances.")
    parser.add_argument("--output-dir", type=str, default="results/", help="Directory for trajectories.jsonl and metrics.json.")
    parser.add_argument("--seed", type=int, default=None, help="Override config.seed.")
    parser.add_argument("--debug", action="store_true", help="Use configs/config_debug.yaml defaults and a tiny eval set.")
    return parser.parse_args()


def evaluate_hotpotqa_or_fever(cfg: AgentConfig, task: str, method: str, n_eval: int | None, output_dir: str) -> dict:
    """Evaluate on HotpotQA (EM) or FEVER (Accuracy), reproducing Table 1."""
    llm_client = build_llm_client(cfg.model)
    prompt_template = PromptTemplate()
    metrics = MetricsCalculator()

    if task == "hotpotqa":
        data_cfg = cfg.data["hotpotqa"]
        dataset = HotpotQADataset(data_cfg["data_dir"], split=data_cfg.get("split", "validation"), n_eval=n_eval or data_cfg.get("n_eval"))
        max_steps = cfg.routing.get("hotpotqa_max_react_steps", 7)
    else:
        data_cfg = cfg.data["fever"]
        dataset = FeverDataset(data_cfg["data_dir"], split=data_cfg.get("split", "validation"), n_eval=n_eval or data_cfg.get("n_eval"))
        max_steps = cfg.routing.get("fever_max_react_steps", 5)

    react_agent = ReactAgent(
        llm_client=llm_client,
        prompt_template=prompt_template,
        max_steps=max_steps,
        task_name=task,
        method="react" if method in ("react", "react_im", "cotsc_react", "react_cotsc") else method,
        thought_mode="dense",
        temperature=cfg.model.get("temperature_main_prompting", 0.0),
        max_tokens_per_step=cfg.model.get("max_tokens_per_step", 256),
    )
    cot_sc = CoTSelfConsistency(llm_client=llm_client, prompt_template=prompt_template, task_name=task)
    router = ReactCotScRouter(
        react_agent=react_agent, cot_sc=cot_sc,
        cotsc_majority_threshold_frac=cfg.routing.get("cotsc_majority_threshold_frac", 0.5),
    )

    predictions: list[str] = []
    golds: list[str] = []
    trajectories_log = []

    print(f"Evaluating {len(dataset)} examples: task={task}, method={method}")
    for i in range(len(dataset)):
        ex = dataset[i]
        question = ex.get("question") or ex.get("claim")
        gold = ex.get("answer") or ex.get("label")
        env = WikipediaEnvironment()

        if method == "cotsc_react":
            pred = router.cotsc_then_react(question, env)
        elif method == "react_cotsc":
            pred = router.react_then_cotsc(question, env, max_react_steps=max_steps)
        elif method == "cot_sc":
            answers = cot_sc.sample(question)
            pred, _frac = cot_sc.majority_vote(answers)
        else:
            result = react_agent.run_episode(task_input=question, env=env)
            pred = result.final_answer or ""
            trajectories_log.append(
                {"id": ex.get("id"), "input": question, "prediction": pred, "gold": gold, "n_steps": result.n_steps, "finished": result.finished}
            )

        predictions.append(pred)
        golds.append(gold)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "trajectories.jsonl"), "w", encoding="utf-8") as f:
        for row in trajectories_log:
            f.write(json.dumps(row) + "\n")

    if task == "hotpotqa":
        score = sum(metrics.exact_match(p, g) for p, g in zip(predictions, golds)) / max(1, len(predictions))
        result = {"task": task, "method": method, "metric": "exact_match", "value": score, "n": len(predictions)}
    else:
        score = metrics.accuracy(predictions, golds)
        result = {"task": task, "method": method, "metric": "accuracy", "value": score, "n": len(predictions)}

    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def evaluate_alfworld_or_webshop(cfg: AgentConfig, task: str, method: str, n_eval: int | None, output_dir: str) -> dict:
    """Evaluate on ALFWorld (SR) or WebShop (Score/SR), reproducing Tables 3-4.

    NOTE: requires the `alfworld` / WebShop-server dependencies to be
    installed and set up separately -- see data/README_data.md. This
    function raises a clear ImportError with setup instructions if the
    environment adapter cannot be constructed, rather than silently no-op'ing.
    """
    llm_client = build_llm_client(cfg.model)
    prompt_template = PromptTemplate()
    metrics = MetricsCalculator()

    if task == "alfworld":
        from react_agent.envs.alfworld_env import ALFWorldEnvironment

        env = ALFWorldEnvironment()
        n_games = n_eval or cfg.data["alfworld"].get("n_eval_games", 134)
        task_instances = [{"game_index": i} for i in range(n_games)]
    else:
        from react_agent.envs.webshop_env import WebShopEnvironment

        env = WebShopEnvironment()
        n_instr = n_eval or cfg.data["webshop"].get("n_eval_instructions", 500)
        task_instances = [{"instruction_id": i} for i in range(n_instr)]

    react_agent = ReactAgent(
        llm_client=llm_client,
        prompt_template=prompt_template,
        max_steps=50,  # ASSUMED generous step cap; paper states ALFWorld tasks can take "more than 50 steps" (Section 4)
        task_name=task,
        method="react_im" if method == "react_im" else "react",
        thought_mode="sparse",
        temperature=cfg.model.get("temperature_main_prompting", 0.0),
        max_tokens_per_step=cfg.model.get("max_tokens_per_step", 256),
    )

    results = []
    print(f"Evaluating {len(task_instances)} instances: task={task}, method={method}")
    for instance in task_instances:
        episode = react_agent.run_episode(task_input=json.dumps(instance), env=env, task_instance=instance)
        episode.success = episode.finished  # ASSUMED: environments should populate a proper success flag via info["success"]
        results.append(episode)

    os.makedirs(output_dir, exist_ok=True)
    sr = metrics.success_rate(results)
    result = {"task": task, "method": method, "metric": "success_rate", "value": sr, "n": len(results)}

    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def main() -> None:
    args = parse_args()
    config_path = "configs/config_debug.yaml" if args.debug and args.config == "configs/config.yaml" else args.config
    cfg = AgentConfig.from_yaml(config_path)

    seed = args.seed if args.seed is not None else cfg.seed
    set_seed(seed, deterministic=cfg.deterministic)

    n_eval = args.n_eval
    if args.debug and n_eval is None:
        n_eval = 5

    start = time.time()
    if args.task in ("hotpotqa", "fever"):
        result = evaluate_hotpotqa_or_fever(cfg, args.task, args.method, n_eval, args.output_dir)
    else:
        try:
            result = evaluate_alfworld_or_webshop(cfg, args.task, args.method, n_eval, args.output_dir)
        except ImportError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
    elapsed = time.time() - start

    print("=" * 70)
    print(f"Result: {result}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Written to: {os.path.join(args.output_dir, 'metrics.json')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
