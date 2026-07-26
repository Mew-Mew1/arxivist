"""
llm/action_parser.py
=====================
Parses raw LLM text output into typed AugmentedAction objects.

Implements the augmented action space split from the SIR mathematical_spec
("Augmented action space definition"): Ahat = A union L, where L is the
language space of free-form "thoughts" (no environment effect) and A is the
task-specific domain action space (search/lookup/finish for HotpotQA/FEVER,
or environment-native text actions for ALFWorld/WebShop).

Paper section: Section 2, paragraph 3 ("The idea of ReAct is simple...").
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Thought:
    """A free-form reasoning trace: aihat_t in L. Produces no observation."""

    text: str


@dataclass
class DomainAction:
    """A task-specific action: a_t in A, e.g. search[Apple Remote], go to drawer 1."""

    name: str
    args: str
    raw_text: str


AugmentedAction = Thought | DomainAction


# Matches the paper's `name[args]` action syntax used for HotpotQA/FEVER/WebShop,
# e.g. "Search[Apple Remote]", "Lookup[Front Row]", "Finish[keyboard function keys]".
_BRACKET_ACTION_RE = re.compile(r"^\s*([A-Za-z_]+)\s*\[(.*)\]\s*$", re.DOTALL)

# Strips a leading "Action i:" numbering prefix, per the paper's prompt format
# (Section 2 / Appendix C), before bracket-matching.
_ACTION_PREFIX_RE = re.compile(r"^\s*Action\s*\d*\s*:\s*", re.IGNORECASE)

# ALFWorld actions are free-form imperative text without brackets, e.g.
# "go to drawer 1", "take keychain 2 from dresser 1", "clean lettuce 1 with sinkbasin 1".
# We treat any non-"Thought:"-prefixed, non-bracket line as a raw ALFWorld action.
_THOUGHT_PREFIX_RE = re.compile(r"^\s*(Thought\s*\d*\s*:|think\s*\[)", re.IGNORECASE)


class ActionParser:
    """Classifies one line of raw LLM output as a Thought or a DomainAction."""

    def parse(self, raw_text: str) -> AugmentedAction:
        """Parse a single generated line into a Thought or DomainAction.

        Args:
            raw_text: Raw text returned by LLMClient.generate() for one step,
                expected to be a single "Thought i: ..." or "Action i: ..."
                line per the paper's prompt format (Section 2, Appendix C).

        Returns:
            A Thought if the text is a reasoning trace, else a DomainAction.
        """
        stripped = raw_text.strip()

        # WebShop's think[...] actions are semantically Thoughts, not
        # environment-affecting actions (SIR module "Action Executor" notes:
        # "think[...] actions are treated as a thought, not an action").
        if _THOUGHT_PREFIX_RE.match(stripped):
            content = re.sub(r"^\s*Thought\s*\d*\s*:\s*", "", stripped, flags=re.IGNORECASE)
            content = re.sub(r"^\s*think\s*\[(.*)\]\s*$", r"\1", content, flags=re.IGNORECASE | re.DOTALL)
            return Thought(text=content.strip())

        # Strip a leading "Action i:" prefix (Section 2 / Appendix C prompt
        # format) before attempting to match the name[args] bracket syntax.
        action_body = _ACTION_PREFIX_RE.sub("", stripped)

        bracket_match = _BRACKET_ACTION_RE.match(action_body)
        if bracket_match:
            name, args = bracket_match.group(1), bracket_match.group(2)
            return DomainAction(name=name.lower(), args=args.strip(), raw_text=action_body.strip())

        # No brackets and no "Thought:" prefix -> treat as a raw ALFWorld-style
        # free-text domain action (e.g. "go to drawer 1"), after stripping any
        # "Action i:" prefix the model may still have emitted.
        return DomainAction(name="raw_text_action", args=action_body.strip(), raw_text=action_body.strip())

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return "ActionParser()"
