#!/usr/bin/env python
"""
train.py
========
Entrypoint for the OPTIONAL Appendix B.1 finetuning experiment: bootstraps
ReAct-generated trajectories and finetunes a substitute open-weight causal LM
to imitate them.

The paper's MAIN results (Tables 1, 3, 4) require no training at all -- see
evaluate.py for the few-shot prompting evaluation. This script only
reproduces the secondary finetuning ablation described in Section 3.3 /
Appendix B.1.

Paper section: Section 3.3 "Finetuning"; Appendix B.1.
"""

from __future__ import annotations

import argparse
import json
import sys

from react_agent.training.finetune import TrainingConfig, TrajectoryFinetuner
from react_agent.utils.config import AgentConfig, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finetune a substitute LM on bootstrapped ReAct trajectories.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML.")
    parser.add_argument("--method", type=str, required=True, choices=["standard", "cot", "act", "react"], help="Which trajectory format to finetune on.")
    parser.add_argument("--model-size", type=str, default="8b", help="Substitute model size tag, e.g. 8b|62b.")
    parser.add_argument(
        "--trajectories-path",
        type=str,
        required=True,
        help="Path to a JSON file of bootstrapped trajectories (list of serialized EpisodeResult dicts).",
    )
    parser.add_argument("--hf-model-name-or-path", type=str, required=True, help="Base HF model to finetune.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training.finetune_batch_size from config.")
    parser.add_argument("--steps", type=int, default=None, help="Override training step count from config.")
    parser.add_argument("--output-dir", type=str, required=True, help="Checkpoint output directory.")
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint to resume from.")
    parser.add_argument("--seed", type=int, default=None, help="Override config.seed.")
    parser.add_argument("--debug", action="store_true", help="Reduce dataset size / steps for a quick local smoke test.")
    parser.add_argument("--dry-run", action="store_true", help="Build all components but do not actually train.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = "configs/config_debug.yaml" if args.debug and args.config == "configs/config.yaml" else args.config
    cfg = AgentConfig.from_yaml(config_path)

    seed = args.seed if args.seed is not None else cfg.seed
    set_seed(seed, deterministic=cfg.deterministic)

    if not cfg.training.get("finetune_enabled", False) and not args.dry_run:
        print(
            "WARNING: training.finetune_enabled is false in the loaded config. "
            "Proceeding anyway since train.py was invoked explicitly, but "
            "double-check configs/config.yaml if this is unexpected.",
            file=sys.stderr,
        )

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print(
            "ERROR: train.py requires 'transformers' and 'torch'. Install via "
            "`pip install -r requirements.txt`.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("=" * 70)
    print("ReAct finetuning (Appendix B.1 reproduction)")
    print(f"  method:              {args.method}")
    print(f"  model_size tag:      {args.model_size}")
    print(f"  base model:          {args.hf_model_name_or_path}")
    print(f"  trajectories_path:   {args.trajectories_path}")
    print(f"  seed:                {seed}")
    print("=" * 70)

    with open(args.trajectories_path, "r", encoding="utf-8") as f:
        raw_trajectories = json.load(f)
    n_trajectories = cfg.training.get("n_bootstrapped_trajectories", 3000)
    if args.debug:
        n_trajectories = min(n_trajectories, 8)
    raw_trajectories = raw_trajectories[:n_trajectories]

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.resume or args.hf_model_name_or_path)

    finetuner = TrajectoryFinetuner(tokenizer=tokenizer)
    dataset = finetuner.build_dataset(raw_trajectories)

    steps_key = f"{'react_act' if args.method in ('react', 'act') else 'standard_cot'}_{args.model_size}"
    default_steps = cfg.training.get("finetune_steps", {}).get(steps_key, 4000)
    steps = args.steps if args.steps is not None else default_steps
    batch_size = args.batch_size if args.batch_size is not None else cfg.training.get("finetune_batch_size", 64)

    print(f"Training summary: {len(dataset)} trajectories, batch_size={batch_size}, steps={steps}")
    print(f"  steps/epoch (approx): {max(1, len(dataset) // batch_size)}")
    try:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  model param count:    {n_params:,}")
    except Exception:  # pragma: no cover - best-effort logging only
        pass

    train_config = TrainingConfig(
        batch_size=batch_size,
        steps=steps,
        optimizer=cfg.training.get("optimizer", "adamw"),
        learning_rate=cfg.training.get("learning_rate", 1e-5),
        weight_decay=cfg.training.get("weight_decay", 0.01),
        lr_schedule=cfg.training.get("lr_schedule", "linear_warmup_then_constant"),
        warmup_steps=cfg.training.get("warmup_steps", 100),
        gradient_clipping=cfg.training.get("gradient_clipping", 1.0),
        log_every_n_steps=cfg.training.get("log_every_n_steps", 50),
        save_every_n_steps=cfg.training.get("save_every_n_steps", 500),
    )

    if args.dry_run:
        print("--dry-run set: components built successfully, skipping actual training.")
        return

    finetuner.train(model, dataset, train_config)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved finetuned checkpoint to {args.output_dir}")


if __name__ == "__main__":
    main()
