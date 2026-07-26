"""
evaluation/metrics.py
======================
Computes the four metrics reported in the paper's result tables:
  - Exact Match (EM) for HotpotQA (Table 1)
  - Accuracy (Acc) for FEVER (Table 1)
  - Success Rate (SR) for ALFWorld (Table 3) and WebShop (Table 4)
  - Average Score (attribute coverage) for WebShop (Table 4)

Paper section: Table 1 caption; Section 4 "WebShop" paragraph.
"""

from __future__ import annotations

import re
import string


class MetricsCalculator:
    """Metric implementations matching the paper's evaluation protocol."""

    def exact_match(self, pred: str, gold: str) -> float:
        """HotpotQA Exact Match (SIR mathematical_spec: EM = 1[y_hat == y]).

        ASSUMED normalization: lowercase, strip articles/punctuation, and
        collapse whitespace before comparing, matching the standard SQuAD/
        HotpotQA EM normalization convention (not restated in the parsed
        paper text; SIR mathematical_spec confidence 0.55 for this metric).

        Args:
            pred: Predicted answer string (e.g. from finish[answer]).
            gold: Ground-truth answer string.

        Returns:
            1.0 if normalized pred == normalized gold, else 0.0.
        """
        return 1.0 if _normalize_answer(pred) == _normalize_answer(gold) else 0.0

    def accuracy(self, preds: list[str], golds: list[str]) -> float:
        """FEVER label accuracy over a batch of predictions.

        Args:
            preds: Predicted labels, e.g. "SUPPORTS" / "REFUTES" / "NOT ENOUGH INFO".
            golds: Ground-truth labels.

        Returns:
            Fraction of exact label matches, in [0, 1].

        Raises:
            ValueError: if preds and golds have different lengths.
        """
        if len(preds) != len(golds):
            raise ValueError(f"preds (n={len(preds)}) and golds (n={len(golds)}) must have equal length.")
        if not preds:
            return 0.0
        matches = sum(1 for p, g in zip(preds, golds) if _normalize_label(p) == _normalize_label(g))
        return matches / len(preds)

    def success_rate(self, episode_results: list) -> float:
        """ALFWorld / WebShop task-level success rate.

        Args:
            episode_results: List of react_agent.agents.react_agent.EpisodeResult,
                each with a `.success` field populated by the caller (e.g.
                from env info["won"] for ALFWorld, or info["success"] for
                WebShop).

        Returns:
            Fraction of episodes with `.success is True`, in [0, 1].
        """
        if not episode_results:
            return 0.0
        successes = sum(1 for r in episode_results if getattr(r, "success", None) is True)
        return successes / len(episode_results)

    def webshop_score(self, chosen_attrs: set, requested_attrs: set) -> float:
        """WebShop attribute-coverage Score for a single episode.

        ASSUMED normalization (SIR ambiguities[2], confidence 0.55): the
        paper describes Score as "the percentage of desired attributes
        covered by the chosen product averaged across all episodes" without
        giving the precise per-episode formula. We implement:

            score_i = |chosen_attrs ∩ requested_attrs| / |requested_attrs|

        and leave averaging across episodes to the caller. Cross-check
        against the official WebShop repository's scoring function
        (Yao et al., 2022) before trusting absolute numbers -- see
        architecture_plan.json risk_assessment (Medium severity item).

        Args:
            chosen_attrs: Set of attributes present on the product the agent
                ultimately bought.
            requested_attrs: Set of attributes specified in the user
                instruction.

        Returns:
            Coverage fraction in [0, 1]. Returns 0.0 if requested_attrs is empty
            (undefined otherwise; treated as no requested attributes to satisfy).
        """
        if not requested_attrs:
            return 0.0
        return len(chosen_attrs & requested_attrs) / len(requested_attrs)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return "MetricsCalculator()"


def _normalize_answer(s: str) -> str:
    """Lowercase, remove punctuation/articles, collapse whitespace. ASSUMED
    convention (SIR mathematical_spec confidence 0.55), not restated verbatim
    in the paper."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def _normalize_label(label: str) -> str:
    return label.strip().upper()
