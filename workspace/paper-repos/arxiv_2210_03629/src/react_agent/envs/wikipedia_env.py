"""
envs/wikipedia_env.py
======================
Simplified Wikipedia API environment used for HotpotQA/FEVER, implementing
the three-action space search[entity] / lookup[string] / finish[answer].

Deliberately weaker than a state-of-the-art neural/lexical retriever, per the
paper's own design rationale: "The purpose is to simulate how humans would
interact with Wikipedia, and force models to retrieve via explicit reasoning
in language" (Section 3.1).

Implementation note (SIR ambiguities[0], confidence 0.6): the exact
sentence-segmentation and similar-entity-suggestion algorithm behind
`search[entity]`'s fallback path is not specified in the paper. We use the
`wikipedia` package's built-in search-suggest, which closely mirrors the
described behavior (see architecture_plan.json risk_assessment, "Low"
severity item).

Paper section: Section 3.1 ("Action Space").
"""

from __future__ import annotations

import re

from react_agent.envs.base_environment import ActionSpace, BaseEnvironment


class WikipediaEnvironment(BaseEnvironment):
    """Implements search[entity], lookup[string], finish[answer] over Wikipedia.

    Args:
        n_sentences: Number of leading sentences to return from
            `search[entity]` when the entity page exists. Explicitly stated
            in the paper as 5 (Section 3.1).
        n_similar: Number of similar-entity suggestions to return when the
            entity page does not exist. Explicitly stated in the paper as 5
            (Section 3.1).
    """

    def __init__(self, n_sentences: int = 5, n_similar: int = 5) -> None:
        try:
            import wikipedia
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "WikipediaEnvironment requires the 'wikipedia' package. "
                "Install it via `pip install -r requirements.txt`."
            ) from e

        self._wikipedia = wikipedia
        self.n_sentences = n_sentences
        self.n_similar = n_similar
        self._current_page = None  # cached page for lookup[]'s "next sentence" behavior
        self._lookup_cursor = 0

    def reset(self, task_instance: dict) -> str:
        """No environment state to reset besides the cached page/cursor.

        Args:
            task_instance: dict with a "question" or "claim" key; only used
                for logging, since HotpotQA/FEVER o_1 is the question/claim
                text itself, formatted directly by the prompt template.

        Returns:
            Empty string (o_1 is supplied by the prompt template, not the
            environment, for this task family).
        """
        self._current_page = None
        self._lookup_cursor = 0
        return ""

    def step(self, action: str) -> tuple[str, bool, dict]:
        match = re.match(r"^\s*(search|lookup|finish)\s*\[(.*)\]\s*$", action, re.IGNORECASE | re.DOTALL)
        if not match:
            return (
                f"Invalid action: {action!r}. Valid actions: search[entity], "
                f"lookup[string], finish[answer].",
                False,
                {"error": "unparsable_action"},
            )
        name, arg = match.group(1).lower(), match.group(2).strip()

        if name == "search":
            return self._search(arg)
        if name == "lookup":
            return self._lookup(arg)
        if name == "finish":
            return f"Episode finished with answer: {arg}", True, {"answer": arg, "success": None}

        return f"Unknown action {name!r}.", False, {"error": "unknown_action"}  # pragma: no cover

    def _search(self, entity: str) -> tuple[str, bool, dict]:
        """search[entity]: first n_sentences of the entity page, or n_similar suggestions."""
        try:
            page = self._wikipedia.page(entity, auto_suggest=False)
        except self._wikipedia.exceptions.PageError:
            suggestions = self._safe_search_suggestions(entity)
            return (
                f"Could not find [{entity}]. Similar: {suggestions}.",
                False,
                {"found": False, "similar": suggestions},
            )
        except self._wikipedia.exceptions.DisambiguationError as e:
            options = list(e.options)[: self.n_similar]
            return (
                f"Could not find [{entity}]. Similar: {options}.",
                False,
                {"found": False, "similar": options},
            )
        except Exception as e:
            # Transport-level failures (rate limiting, transient network errors,
            # malformed API responses) are not modeled by the `wikipedia`
            # package's own exception hierarchy -- treat them as a soft,
            # recoverable search failure rather than crashing the episode,
            # so a single flaky request doesn't abort an entire eval batch.
            return (
                f"Search for [{entity}] temporarily failed ({type(e).__name__}). "
                f"Try a different search term or try again.",
                False,
                {"found": False, "error": str(e)},
            )

        self._current_page = page
        self._lookup_cursor = 0
        sentences = _split_sentences(page.content)[: self.n_sentences]
        return " ".join(sentences), False, {"found": True, "title": page.title}

    def _safe_search_suggestions(self, entity: str) -> list:
        """Wraps wikipedia.search() so a transport failure here doesn't also crash the episode."""
        try:
            return self._wikipedia.search(entity, results=self.n_similar)
        except Exception:
            return []

    def _lookup(self, query: str) -> tuple[str, bool, dict]:
        """lookup[string]: emulate browser Ctrl+F -- next sentence containing `query`."""
        if self._current_page is None:
            return "No page currently open. Use search[entity] first.", False, {"error": "no_page"}

        sentences = _split_sentences(self._current_page.content)
        matches = [i for i, s in enumerate(sentences) if query.lower() in s.lower()]
        remaining = [i for i in matches if i >= self._lookup_cursor]

        if not remaining:
            return "No more results.", False, {"found": False}

        idx = remaining[0]
        self._lookup_cursor = idx + 1
        result_num = matches.index(idx) + 1
        return (
            f"(Result {result_num} / {len(matches)}) {sentences[idx]}",
            False,
            {"found": True, "result_index": result_num, "total_results": len(matches)},
        )

    def action_space(self) -> ActionSpace:
        return ActionSpace(
            action_names=["search", "lookup", "finish"],
            description="Simplified Wikipedia web API (Section 3.1): "
            "search[entity], lookup[string], finish[answer].",
        )


def _split_sentences(text: str) -> list[str]:
    """Naive sentence splitter.

    ASSUMED: the paper does not specify a sentence-segmentation algorithm
    for the 'first 5 sentences' extraction (SIR ambiguities[0]). A proper
    implementation should swap this for nltk.sent_tokenize or spacy's
    sentencizer for higher fidelity on abbreviations/edge cases.
    """
    # Split on '.', '!', '?' followed by whitespace, keeping the punctuation.
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if s.strip()]