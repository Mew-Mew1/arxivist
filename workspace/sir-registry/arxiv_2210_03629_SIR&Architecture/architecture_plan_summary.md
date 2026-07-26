# Architecture Plan Summary — ReAct (arXiv:2210.03629)

## Framework
**PyTorch + HuggingFace Transformers** (for the optional finetuning path), YAML config, no hard CUDA requirement.
ReAct's core contribution is a *prompting/orchestration loop* over a frozen LLM, not a trainable
neural architecture, so the repo is structured as an **LLM-agnostic agent framework** with a
pluggable `LLMClient` (OpenAI-style API or local HF model), plus a separate, optional finetuning
module mirroring Appendix B.1.

## Module Hierarchy (15 files)
- `agents/react_agent.py` — the core Thought↔Action↔Observation loop (`ReactAgent`, `TrajectoryContext`)
- `agents/cot_sc.py` — Self-Consistency baseline + ReAct↔CoT-SC backoff routers
- `envs/base_environment.py`, `wikipedia_env.py`, `alfworld_env.py`, `webshop_env.py` — one adapter per benchmark, all conforming to a common `reset/step` contract
- `llm/llm_client.py`, `llm/action_parser.py` — swappable LLM backend + Thought-vs-Action text parsing
- `prompts/prompt_templates.py` — few-shot exemplar management per task/method
- `training/finetune.py` — Appendix B.1 trajectory-bootstrapped finetuning
- `data/hotpotqa_dataset.py`, `data/fever_dataset.py` — dataset loaders
- `evaluation/metrics.py` — EM / Accuracy / Success Rate / WebShop Score
- `utils/config.py` — YAML config loader

## Tensor / Data Flows (4 documented)
1. **Dense-thought ReAct step** (HotpotQA/FEVER): serialize context → LLM generate → parse Thought vs Action → environment step if Action → repeat up to step cap (7 / 5).
2. **Sparse-thought ReAct step** (ALFWorld/WebShop): same loop, but the LLM decides Thought frequency itself.
3. **CoT-SC ↔ ReAct routing**: two backoff heuristics exactly as specified in the paper (fail-to-finish → CoT-SC; low-agreement majority vote → ReAct).
4. **Finetuning pass**: tokenize 3,000 bootstrapped trajectories → masked causal-LM loss → AdamW (assumed) for 4,000/2,000/1,000 steps depending on method/model size, batch size 64 (paper-stated).

## Config Highlights (see `config_schema`)
- `model.llm_backend`, `model.model_name` are explicitly marked **ASSUMED** — PaLM-540B is not public.
- `training.learning_rate` is **unspecified in the paper** (confidence 0.2) — left `null`, must be tuned.
- All paper-stated numbers (batch_size=64, cotsc_n_samples=21, temperature=0.7, step caps 7/5, alfworld eval games=134, webshop eval instructions=500) are wired in directly, no `ASSUMED` tag.

## Entrypoints
- `train.py` — optional finetuning (Appendix B.1 reproduction)
- `evaluate.py` — full method × task sweep, reproduces Tables 1/3/4
- `inference.py` — single-episode trace for one question/claim/task instance
- `human_edit_repl.py` — reproduces the Appendix A.3 human-in-the-loop thought-editing demo

## Docker
`pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime` base, git/build-essential/curl system deps,
default CMD runs the HotpotQA ReAct evaluation.

## Risk Assessment (7 risks: 2 High, 3 Medium, 2 Low)
| Severity | Risk | Mitigation |
|---|---|---|
| High | PaLM-540B not public — exact score reproduction unreachable | Pluggable `LLMClient`, compare relative trends, mirror paper's own GPT-3 ablation |
| High | Full Appendix C few-shot prompts not in parsed text | Hand-author exemplars from documented thought taxonomy + excerpted figures, flag with TODOs |
| Medium | Finetuning hyperparameters (optimizer/LR/schedule) unspecified | AdamW + standard schedule default, all exposed as `ASSUMED` config values |
| Medium | WebShop Score normalization not fully specified | Assumed matched/requested ratio, swappable, cross-check against official WebShop repo |
| Medium | ALFWorld/WebShop external simulator dependencies (large downloads, version drift) | Pin versions, document manual data-download step, graceful degradation if uninstalled |
| Low | Wikipedia fallback "similar entities" algorithm unspecified | Use `wikipedia` package's built-in search-suggest |
| Low | Step-cap counting ambiguity (Actions only vs all turns) | Count only Action/Observation pairs by default; alternate mode behind config flag |

**Bottom line:** the plan treats ReAct as an agent-orchestration framework, not a model-training
repo — the highest-risk items are all about *faithfully reconstructing missing prompts and
substituting an inaccessible base model*, not about numerical/tensor correctness.
