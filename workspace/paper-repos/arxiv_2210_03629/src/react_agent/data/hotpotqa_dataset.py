"""
data/hotpotqa_dataset.py
=========================
Loads the HotpotQA question-only validation split (Section 3.1: "we operate
in a question-only setup ... where models only receive the question/claim as
input without access to support paragraphs").

Paper section: Section 3.1 ("Domains").
"""

from __future__ import annotations

import json
import os

from torch.utils.data import Dataset


class HotpotQADataset(Dataset):
    """HotpotQA question-only dataset.

    Expects a JSON file at `{data_dir}/hotpot_{split}_v1.1.json` following the
    official HotpotQA release format (a list of dicts with "_id", "question",
    "answer" keys); see data/README_data.md for download instructions.

    Args:
        data_dir: Directory containing the downloaded HotpotQA JSON file.
            Never hardcoded -- always read from configs/config.yaml::
            data.hotpotqa.data_dir.
        split: Dataset split, e.g. "validation". The official HotpotQA
            distributed dev file is used as the "validation" split.
        n_eval: If set, truncate to the first n_eval examples (SIR
            evaluation_protocol: 500-question subset used in Appendix A.1's
            GPT-3 comparison).
    """

    def __init__(self, data_dir: str, split: str = "validation", n_eval: int | None = None) -> None:
        self.data_dir = data_dir
        self.split = split
        filename = "hotpot_dev_distractor_v1.json" if split == "validation" else f"hotpot_{split}_v1.1.json"
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"HotpotQA data file not found at {path!r}. Run "
                f"`python data/download.py --dataset hotpotqa` first, or see "
                f"data/README_data.md."
            )

        with open(path, "r", encoding="utf-8") as f:
            examples = json.load(f)

        if n_eval is not None:
            examples = examples[:n_eval]
        self._examples = examples

    def __getitem__(self, idx: int) -> dict:
        ex = self._examples[idx]
        return {"question": ex["question"], "answer": ex["answer"], "id": ex.get("_id", str(idx))}

    def __len__(self) -> int:
        return len(self._examples)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"HotpotQADataset(split={self.split!r}, n={len(self)})"
