"""
data/fever_dataset.py
======================
Loads the FEVER claim-only validation split (Section 3.1: same question-only
/ claim-only setup as HotpotQA, no access to supporting evidence documents).

Paper section: Section 3.1 ("Domains").
"""

from __future__ import annotations

import json
import os

from torch.utils.data import Dataset


class FeverDataset(Dataset):
    """FEVER claim-only dataset.

    Expects a JSONL file at `{data_dir}/{split}.jsonl` following the official
    FEVER release format (one JSON object per line with "id", "claim",
    "label" keys, label in {SUPPORTS, REFUTES, NOT ENOUGH INFO}); see
    data/README_data.md for download instructions.

    Args:
        data_dir: Directory containing the downloaded FEVER JSONL file.
            Never hardcoded -- always read from configs/config.yaml::
            data.fever.data_dir.
        split: Dataset split, e.g. "validation" (mapped to FEVER's official
            "shared_task_dev" file).
        n_eval: If set, truncate to the first n_eval examples. ASSUMED: the
            paper does not state an exact FEVER eval subset size in the
            parsed text (SIR data.fever.n_eval is null by default).
    """

    def __init__(self, data_dir: str, split: str = "validation", n_eval: int | None = None) -> None:
        self.data_dir = data_dir
        self.split = split
        filename = "shared_task_dev.jsonl" if split == "validation" else f"{split}.jsonl"
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"FEVER data file not found at {path!r}. Run "
                f"`python data/download.py --dataset fever` first, or see "
                f"data/README_data.md."
            )

        examples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))

        if n_eval is not None:
            examples = examples[:n_eval]
        self._examples = examples

    def __getitem__(self, idx: int) -> dict:
        ex = self._examples[idx]
        return {"claim": ex["claim"], "label": ex["label"], "id": ex.get("id", str(idx))}

    def __len__(self) -> int:
        return len(self._examples)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"FeverDataset(split={self.split!r}, n={len(self)})"
