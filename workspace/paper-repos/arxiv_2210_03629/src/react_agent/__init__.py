"""
react_agent
===========

Reproduction package for "ReAct: Synergizing Reasoning and Acting in Language
Models" (Yao et al., ICLR 2023, arXiv:2210.03629).

ReAct is a prompting paradigm that interleaves free-form natural-language
"thoughts" with task-specific "actions" when querying a large language model,
so that reasoning can guide acting (thought -> action) and acting can ground
reasoning in fresh external information (action -> observation -> thought).

This package implements the core agent loop (react_agent.agents.react_agent),
task environments (react_agent.envs), a pluggable LLM backend
(react_agent.llm), and the evaluation / optional finetuning machinery
described in Sections 2-4 and Appendix B.1 of the paper.

See paper-repos/arxiv_2210_03629/README.md for quick start instructions and
Reproducibility Notes.
"""

__version__ = "0.1.0"
