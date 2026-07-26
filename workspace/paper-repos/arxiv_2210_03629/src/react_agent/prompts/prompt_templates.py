"""
prompts/prompt_templates.py
============================
Stores and formats the few-shot in-context exemplar trajectories used to
prompt the LLM for each (task, method) pair.

STUB / KNOWN GAP: The paper's full few-shot exemplar prompts live in
Appendix C, which was not present in the parsed PDF text supplied to
ArXivist Stage 1 (see SIR ambiguities[0] and ambiguities[1], confidence 0.5).
The `_EXEMPLARS` dict below is hand-authored from the illustrative fragments
shown in Figure 1 and the excerpted HotpotQA/FEVER/WebShop prompt tables in
the visible paper text (Tables 6-9). It follows the documented thought
taxonomy (decompose goal, extract information, commonsense/arithmetic
reasoning, track progress, handle exceptions, reformulate search) but is NOT
guaranteed to match Appendix C verbatim.

# TODO: verify against paper Appendix C / official repo (https://react-lm.github.io/)
before running large-scale evaluation and trusting absolute score comparisons.

Paper section: Section 3.2 ("we randomly select 6 and 3 cases..."); Section 4
("we randomly annotate three trajectories..."); Appendix C (not available).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Import only for type checking to avoid a circular import: agents.react_agent
    # imports PromptTemplate, and PromptTemplate.format_step() takes a
    # TrajectoryContext, so a runtime import here would create a cycle.
    from react_agent.agents.react_agent import TrajectoryContext

# One entry per (task_name, method). `method` in {"standard","cot","act","react","react_im"}.
# Each value is the literal few-shot prefix text prepended before the live question/task.
_EXEMPLARS: dict[tuple[str, str], str] = {
    ("hotpotqa", "react"): (
        "Question: What is the elevation range for the area that the eastern sector of the "
        "Colorado orogeny extends into?\n"
        "Thought 1: I need to search Colorado orogeny, find the area that the eastern sector of "
        "the Colorado orogeny extends into, then find the elevation range of the area.\n"
        "Action 1: Search[Colorado orogeny]\n"
        "Observation 1: The Colorado orogeny was an episode of mountain building (an orogeny) in "
        "Colorado and surrounding areas.\n"
        "Thought 2: It does not mention the eastern sector. So I need to look up eastern sector.\n"
        "Action 2: Lookup[eastern sector]\n"
        "Observation 2: (Result 1 / 1) The eastern sector extends into the High Plains and is "
        "called the Central Plains orogeny.\n"
        "Thought 3: The eastern sector of Colorado orogeny extends into the High Plains. So I "
        "need to search High Plains and find its elevation range.\n"
        "Action 3: Search[High Plains]\n"
        "Observation 3: High Plains rise in elevation from around 1,800 to 7,000 ft.\n"
        "Thought 4: High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer "
        "is 1,800 to 7,000 ft.\n"
        "Action 4: Finish[1,800 to 7,000 ft]\n\n"
        # ASSUMED: paper uses 6 exemplars for HotpotQA (Section 3.2); only 1 shown here for
        # brevity. Add the remaining 5 by following this same Thought/Action/Observation format.
    ),
    ("hotpotqa", "cot"): (
        "Question: What is the elevation range for the area that the eastern sector of the "
        "Colorado orogeny extends into?\n"
        "Thought: Let's think step by step. The eastern sector of Colorado orogeny extends into "
        "the High Plains. High Plains rise in elevation from around 1,800 to 7,000 ft, so the "
        "answer is 1,800 to 7,000 ft.\n"
        "Answer: 1,800 to 7,000 ft\n\n"
    ),
    ("hotpotqa", "act"): (
        "Question: What is the elevation range for the area that the eastern sector of the "
        "Colorado orogeny extends into?\n"
        "Action 1: Search[Colorado orogeny]\n"
        "Observation 1: The Colorado orogeny was an episode of mountain building.\n"
        "Action 2: Lookup[eastern sector]\n"
        "Observation 2: The eastern sector extends into the High Plains.\n"
        "Action 3: Search[High Plains]\n"
        "Observation 3: High Plains rise in elevation from around 1,800 to 7,000 ft.\n"
        "Action 4: Finish[1,800 to 7,000 ft]\n\n"
    ),
    ("hotpotqa", "standard"): (
        "Question: What is the elevation range for the area that the eastern sector of the "
        "Colorado orogeny extends into?\n"
        "Answer: 1,800 to 7,000 ft\n\n"
    ),
    ("fever", "react"): (
        "Claim: Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.\n"
        "Thought 1: I need to search Nikolaj Coster-Waldau and find if he has worked with the "
        "Fox Broadcasting Company.\n"
        "Action 1: Search[Nikolaj Coster-Waldau]\n"
        "Observation 1: ...appeared in the 2009 Fox television film Virtuality...\n"
        "Thought 2: Because he appeared in a Fox television film, he should have worked with "
        "the Fox Broadcasting Company.\n"
        "Action 2: Finish[SUPPORTS]\n\n"
        # ASSUMED: paper uses 3 exemplars for FEVER (Section 3.2); only 1 shown here for brevity.
    ),
    ("webshop", "react"): (
        "Instruction: i would like a 3 ounce bottle of bright citrus deodorant for sensitive "
        "skin, and price lower than 50.00 dollars\n"
        "Action: search[3 ounce bright citrus deodorant sensitive skin]\n"
        "Observation: [... search results omitted ...]\n"
        "Action: think[B078GWRC1J is a 3 ounce bright citrus deodorant, seems like a good match. "
        "I can check it first.]\n"
        "Action: click[B078GWRC1J]\n"
        "Observation: [... product page omitted ...]\n"
        "Action: think[For 3 ounce bottle of bright citrus deodorant for sensitive skin, the "
        "item has options 'bright citrus' and '3 ounce (pack of 1)' and seems good to buy.]\n"
        "Action: click[bright citrus]\n"
        "Action: click[3 ounce (pack of 1)]\n"
        "Action: click[Buy Now]\n\n"
    ),
}


class PromptTemplate:
    """Loads and formats few-shot prompts per (task, method).

    See module docstring for the known Appendix-C reconstruction gap.
    """

    def __init__(self, exemplars: dict[tuple[str, str], str] | None = None) -> None:
        self._exemplars = exemplars if exemplars is not None else _EXEMPLARS

    def load(self, task_name: str, method: str) -> str:
        """Return the formatted few-shot prefix string for (task_name, method).

        Args:
            task_name: one of "hotpotqa", "fever", "alfworld", "webshop".
            method: one of "standard", "cot", "act", "react", "react_im".

        Returns:
            The literal few-shot prefix text to prepend before the live
            question/claim/task instance.

        Raises:
            KeyError: if no exemplar is registered for (task_name, method).
                Register additional exemplars in `_EXEMPLARS` above, ideally
                sourced from the official paper repository
                (https://react-lm.github.io/) rather than hand-authored.
        """
        key = (task_name.lower(), method.lower())
        if key not in self._exemplars:
            raise KeyError(
                f"No few-shot exemplar registered for (task={task_name!r}, "
                f"method={method!r}). See prompts/prompt_templates.py module "
                f"docstring: full Appendix C prompts were not available at "
                f"parse time and must be added manually or sourced from "
                f"https://react-lm.github.io/."
            )
        return self._exemplars[key]

    def format_step(self, task_name: str, method: str, context: "TrajectoryContext") -> str:
        """Build the full prompt for the next generation step.

        Args:
            task_name: as in `load`.
            method: as in `load`.
            context: the TrajectoryContext accumulated so far for this episode.

        Returns:
            few_shot_prefix + serialized trajectory, ready for
            LLMClient.generate().
        """
        prefix = self.load(task_name, method)
        return prefix + context.to_prompt_string()

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"PromptTemplate(n_registered={len(self._exemplars)})"
