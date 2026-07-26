"""
training/finetune.py
=====================
Bootstraps 3,000 ReAct-generated trajectories to finetune a substitute
open-weight causal LM (standing in for the paper's PaLM-8B/62B) to decode
full (thought, action, observation) trajectories conditioned on the input
question/claim.

WARNING: low-confidence implementation. The paper's Appendix B.1 states
batch size 64 and a training-step count per (method, model size), but does
NOT specify an optimizer, learning rate, or LR schedule (SIR
training_pipeline confidence 0.5; SIR implementation_assumptions[4],
confidence 0.55). The optimizer/LR/schedule below are marked `# ASSUMED` and
should be tuned; they are standard defaults, not paper-derived values.

Paper section: Section 3.3 "Finetuning"; Appendix B.1.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch.utils.data import Dataset


@dataclass
class TrainingConfig:
    """Finetuning hyperparameters, mirroring configs/config.yaml::training.

    All fields except `batch_size`, `steps`, and `n_bootstrapped_trajectories`
    are ASSUMED (not specified in the parsed paper text) -- see module
    docstring.
    """

    batch_size: int = 64  # explicitly stated, Appendix B.1
    steps: int = 4000  # explicitly stated per (method, model size), Appendix B.1
    optimizer: str = "adamw"  # ASSUMED
    learning_rate: float = 1.0e-5  # ASSUMED -- NOT stated anywhere in the paper
    weight_decay: float = 0.01  # ASSUMED
    lr_schedule: str = "linear_warmup_then_constant"  # ASSUMED
    warmup_steps: int = 100  # ASSUMED
    gradient_clipping: float = 1.0  # ASSUMED
    log_every_n_steps: int = 50
    save_every_n_steps: int = 500


class TrajectoryFinetuner:
    """Bootstraps trajectories and finetunes a causal LM to imitate them.

    Args:
        tokenizer: A HuggingFace tokenizer for the substitute model.
        max_seq_length: Maximum tokenized sequence length (prompt + target
            trajectory). ASSUMED: not specified in the paper.
    """

    def __init__(self, tokenizer, max_seq_length: int = 2048) -> None:
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def build_dataset(self, trajectories: list) -> Dataset:
        """Tokenize (prompt, target trajectory) pairs for supervised finetuning.

        Args:
            trajectories: List of react_agent.agents.react_agent.EpisodeResult
                objects with `.trajectory.to_prompt_string()` used as the
                supervised target (all thoughts, actions, observations, per
                Section 3.3: "to decode trajectories (all thoughts, actions,
                observations) conditioned on input questions/claims").

        Returns:
            A torch.utils.data.Dataset yielding tokenized
            {"input_ids", "attention_mask", "labels"} dicts, with prompt
            tokens masked out of the loss (label = -100) so only the target
            trajectory tokens contribute to the cross-entropy loss.
        """
        return _TrajectoryDataset(trajectories, self.tokenizer, self.max_seq_length)

    def train(self, model, dataset: Dataset, config: TrainingConfig):
        """Finetune `model` on `dataset` following `config`.

        Args:
            model: A HuggingFace PreTrainedModel (causal LM) to finetune.
            dataset: Output of `build_dataset()`.
            config: TrainingConfig with optimizer/schedule/step hyperparameters.

        Returns:
            The finetuned model (mutated in place and also returned).

        Note:
            This function performs a standard supervised finetuning loop
            (masked next-token cross-entropy) using HuggingFace's Trainer.
            The training *objective* (predict the full trajectory) is
            paper-grounded (Section 3.3); the optimizer/LR/schedule choices
            below are `# ASSUMED` per the module docstring.
        """
        try:
            import torch
            from transformers import Trainer, TrainingArguments
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "TrajectoryFinetuner.train() requires 'torch' and "
                "'transformers'. Install via `pip install -r requirements.txt`."
            ) from e

        training_args = TrainingArguments(
            output_dir="checkpoints/finetune",
            per_device_train_batch_size=config.batch_size,
            max_steps=config.steps,
            learning_rate=config.learning_rate,  # ASSUMED
            weight_decay=config.weight_decay,  # ASSUMED
            warmup_steps=config.warmup_steps,  # ASSUMED
            lr_scheduler_type="linear",  # ASSUMED (approximates linear_warmup_then_constant)
            max_grad_norm=config.gradient_clipping,  # ASSUMED
            logging_steps=config.log_every_n_steps,
            save_steps=config.save_every_n_steps,
            optim="adamw_torch" if config.optimizer == "adamw" else config.optimizer,  # ASSUMED
            report_to=[],
        )

        trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
        trainer.train()
        return model

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"TrajectoryFinetuner(max_seq_length={self.max_seq_length})"


class _TrajectoryDataset(Dataset):
    """Internal Dataset wrapper tokenizing (prompt, target) trajectory pairs."""

    def __init__(self, trajectories: list, tokenizer, max_seq_length: int) -> None:
        self._trajectories = trajectories
        self._tokenizer = tokenizer
        self._max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self._trajectories)

    def __getitem__(self, idx: int) -> dict:
        episode = self._trajectories[idx]
        target_text = episode.trajectory.to_prompt_string()
        encoded = self._tokenizer(
            target_text,
            truncation=True,
            max_length=self._max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"][0]
        attention_mask = encoded["attention_mask"][0]
        # ASSUMED: no prompt/target masking distinction implemented at this
        # stub level (full loss over the whole sequence); a production
        # implementation should mask the few-shot-prefix tokens with -100
        # per Section 3.3's "conditioned on input questions/claims" framing.
        labels = input_ids.clone()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
