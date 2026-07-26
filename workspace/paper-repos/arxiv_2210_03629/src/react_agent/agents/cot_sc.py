"""
agents/cot_sc.py
=================
Chain-of-Thought Self-Consistency (CoT-SC) baseline, and the ReAct<->CoT-SC
backoff routers described in Section 3.2 ("Combining Internal and External
Knowledge").

Two heuristics, both explicitly stated in the paper:
    A) ReAct -> CoT-SC: if ReAct fails to return an answer within
       max_react_steps (7 for HotpotQA, 5 for FEVER), back off to CoT-SC.
    B) CoT-SC -> ReAct: if the majority answer among n CoT-SC samples occurs
       less than n/2 times, back off to ReAct.

Paper section: Section 3.2.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from react_agent.agents.react_agent import ReactAgent
from react_agent.envs.base_environment import BaseEnvironment
from react_agent.llm.llm_client import LLMClient
from react_agent.prompts.prompt_templates import PromptTemplate


@dataclass
class CoTSCResult:
    """Result of a CoT-SC sampling round."""

    answer: str
    vote_fraction: float
    all_answers: list[str]


class CoTSelfConsistency:
    """Samples n CoT trajectories at temperature > 0 and takes a majority vote.

    Args:
        llm_client: Backend used to generate raw text continuations.
        prompt_template: Supplies the CoT few-shot prefix for a given task.
        task_name: one of "hotpotqa", "fever" (CoT-SC is only used for the
            knowledge-intensive reasoning tasks in the paper, Section 3).
    """

    def __init__(self, llm_client: LLMClient, prompt_template: PromptTemplate, task_name: str) -> None:
        self.llm_client = llm_client
        self.prompt_template = prompt_template
        self.task_name = task_name

    def sample(self, question: str, n_samples: int = 21, temperature: float = 0.7) -> list[str]:
        """Sample n_samples independent CoT completions and extract each final answer.

        Args:
            question: The HotpotQA question or FEVER claim text.
            n_samples: Number of samples. Explicitly stated as 21 in the paper.
            temperature: Sampling temperature. Explicitly stated as 0.7.

        Returns:
            A list of n_samples extracted final answers (one per sample).
        """
        prefix = self.prompt_template.load(self.task_name, "cot")
        prompt = prefix + f"Question: {question}\nThought:" if self.task_name == "hotpotqa" else prefix + f"Claim: {question}\nThought:"

        answers: list[str] = []
        for _ in range(n_samples):
            raw = self.llm_client.generate(
                prompt=prompt, stop_sequences=["\n\n"], temperature=temperature, max_tokens=256
            )
            answers.append(_extract_answer(raw))
        return answers

    def majority_vote(self, answers: list[str]) -> tuple[str, float]:
        """Return the majority answer and the fraction of samples that agree with it.

        Args:
            answers: List of extracted answers from `sample()`.

        Returns:
            (majority_answer, vote_fraction) where vote_fraction is in [0, 1].
        """
        if not answers:
            raise ValueError("majority_vote() called with an empty answers list.")
        counts = Counter(answers)
        top_answer, top_count = counts.most_common(1)[0]
        return top_answer, top_count / len(answers)


def _extract_answer(raw_cot_text: str) -> str:
    """Extract the final 'Answer: ...' line from a raw CoT completion.

    ASSUMED: simple substring extraction after the literal 'Answer:' marker,
    matching the few-shot exemplar format in prompts/prompt_templates.py.
    """
    marker = "Answer:"
    idx = raw_cot_text.find(marker)
    if idx == -1:
        return raw_cot_text.strip()
    return raw_cot_text[idx + len(marker) :].strip().split("\n")[0]


class ReactCotScRouter:
    """Implements the two backoff heuristics from Section 3.2.

    Args:
        react_agent: A configured ReactAgent (method="react").
        cot_sc: A configured CoTSelfConsistency instance.
        cotsc_majority_threshold_frac: Threshold below which CoT-SC's
            majority vote is considered low-confidence (SIR: "less than n/2
            times" -> exactly 0.5).
    """

    def __init__(
        self,
        react_agent: ReactAgent,
        cot_sc: CoTSelfConsistency,
        cotsc_majority_threshold_frac: float = 0.5,
    ) -> None:
        self.react_agent = react_agent
        self.cot_sc = cot_sc
        self.cotsc_majority_threshold_frac = cotsc_majority_threshold_frac

    def react_then_cotsc(self, question: str, env: BaseEnvironment, max_react_steps: int) -> str:
        """Heuristic A: run ReAct; if it fails to Finish within max_react_steps, back off to CoT-SC.

        Args:
            question: The HotpotQA question or FEVER claim text.
            env: The environment to run the ReAct episode against.
            max_react_steps: Step cap (7 for HotpotQA, 5 for FEVER per SIR).

        Returns:
            The final answer, either from ReAct or the CoT-SC fallback.
        """
        result = self.react_agent.run_episode(task_input=question, env=env)
        if result.finished and result.final_answer is not None:
            return result.final_answer

        answers = self.cot_sc.sample(question)
        answer, _frac = self.cot_sc.majority_vote(answers)
        return answer

    def cotsc_then_react(self, question: str, env: BaseEnvironment, n_samples: int = 21) -> str:
        """Heuristic B: run CoT-SC; if majority vote fraction < threshold, back off to ReAct.

        Args:
            question: The HotpotQA question or FEVER claim text.
            env: The environment to run the ReAct fallback episode against.
            n_samples: Number of CoT-SC samples (default 21, per paper).

        Returns:
            The final answer, either from CoT-SC or the ReAct fallback.
        """
        answers = self.cot_sc.sample(question, n_samples=n_samples)
        answer, frac = self.cot_sc.majority_vote(answers)
        if frac >= self.cotsc_majority_threshold_frac:
            return answer

        result = self.react_agent.run_episode(task_input=question, env=env)
        return result.final_answer if result.final_answer is not None else answer

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"ReactCotScRouter(threshold={self.cotsc_majority_threshold_frac})"
