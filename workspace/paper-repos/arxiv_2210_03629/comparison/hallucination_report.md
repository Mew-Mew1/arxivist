# Hallucination Report

**Paper**: ReAct: Synergizing Reasoning and Acting in Language Models
**Paper ID**: arxiv_2210_03629
**Comparison Date**: 2026-07-27

This report classifies deviations between what `sir.json` / `architecture_plan.json` specify and what the generated + subsequently hand-edited code (`paper-repos/arxiv_2210_03629/`) actually does. Findings are cross-referenced with the Root Cause Analysis in `benchmark_comparison.md` where they directly explain observed score deviations.

---

## Structural Hallucinations
*(Components present in the code that are NOT in the SIR/architecture plan)*

### S1 — Action-name validation branch added to `ReactAgent.run_episode()`
- **Component**: A pre-dispatch check (`valid_names = set(env.action_space().action_names); if valid_names and augmented_action.name not in valid_names: ...`) that intercepts a parsed `DomainAction` before it reaches `env.step()`, returning a corrective observation if the action name isn't recognized.
- **Location in code**: `src/react_agent/agents/react_agent.py`, inside `run_episode()`, between the Thought-handling branch and the "DomainAction: dispatch to the environment" comment.
- **Severity**: Minor
- **Evidence**: `architecture_plan.json`'s `tensor_flows[0]` ("ReAct single-step decision") describes the loop as: parse → if Thought, append; if DomainAction, dispatch directly to `BaseEnvironment.step()`. No validation-before-dispatch step is specified there or in the SIR architecture graph — this was added during interactive debugging (this session) after observing `WikipediaEnvironment` reject a hallucinated action name (`"Research[...]"`) with only a generic message, at the cost of a full environment step with no useful signal.
- **Suggested fix**: None required — this is a legitimate, evidence-driven improvement over the original plan, not an error. Recommend formalizing it: add this validation step to `architecture_plan.json::tensor_flows[0]` and `sir.json` on the next SIR revision so future regenerations don't silently drop it (this exact fix was lost and had to be manually reapplied twice during this session after the code was copied between folders).

---

## Parametric Hallucinations
*(Hyperparameters/config values that were `# ASSUMED` and may be wrong — especially where they coincide with a deviation)*

### P1 — Substitute model identity (`model_name`) — Critical
- **Hyperparameter**: `configs/config.yaml::model.model_name`
- **Assumed value**: Evolved over this session from `gpt-4o-mini` → `deepseek-chat` → `deepseek-v4-flash` (final value that the live DeepSeek-compatible endpoint actually accepted).
- **Severity**: Critical
- **Evidence**: SIR `implementation_assumptions[0]` (confidence 0.95) explicitly flags that PaLM-540B is not publicly accessible and any substitute is an assumption. Directly implicated in Root Cause Analysis Cause 4 (`benchmark_comparison.md`) — DeepSeek was observed reasoning from parametric memory rather than grounding via `search[...]` in manual single-question tests, a real behavioral gap from what the paper documents. Not the dominant cause of the score gap (Causes 1–2 are), but a contributing, unavoidable one.
- **Suggested fix**: None available (paper's own model is inaccessible). If comparing model quality specifically, re-run the same 50-example batch with `deepseek-v4-pro` and with the `hf_local` backend (e.g. `Qwen2.5-7B-Instruct`, already validated working earlier in this session) to see how much of the gap is model-capability-driven vs. prompt/step-budget-driven (Causes 1–2).

### P2 — Exact-match normalization convention — Significant
- **Hyperparameter**: `src/react_agent/evaluation/metrics.py::_normalize_answer` (lowercase, strip punctuation, strip articles a/an/the, collapse whitespace)
- **Assumed value**: Standard SQuAD/HotpotQA-style normalization, since the paper doesn't restate its own EM formula.
- **Severity**: Significant
- **Evidence**: SIR `mathematical_spec[3]` ("HotpotQA exact-match evaluation metric (implicit)") is marked confidence 0.55 for exactly this reason. Directly implicated in Root Cause Analysis Cause 3 — at least 2 of the 27 finished-and-wrong episodes in the sampled trajectories are near-misses purely on formatting (e.g. `"1986 to 2013"` vs gold `"from 1986 to 2013"`; a full-sentence `"Yes, both..."` vs gold `"yes"`).
- **Suggested fix**: Consider extending `_normalize_answer` to strip a small set of connector words (`"from"`, `"the"` variants already handled) and/or add a terse-answer-extraction heuristic for yes/no questions. Treat as optional — this is a defensible implementation choice, not a paper-fidelity violation, and richer few-shot exemplars (Recommended Action 1 in `benchmark_comparison.md`) address the root behavioral cause more directly than patching the metric would.

---

## Omission Hallucinations
*(Components present in the SIR/paper spec but absent or stubbed in the generated code)*

### O1 — Full Appendix C few-shot exemplar set — Critical
- **Missing component**: 6 HotpotQA exemplars / 3 FEVER exemplars per the paper (Section 3.2: "we randomly select 6 and 3 cases..."). The generated code ships exactly 1 hand-authored exemplar per (task, method) in `src/react_agent/prompts/prompt_templates.py`.
- **SIR location**: `sir.json → ambiguities[0]` and `ambiguities[1]` (confidence 0.5) already flag this as a known gap — Appendix C's full text was not present in the parsed PDF at Stage 1.
- **Severity**: Critical
- **Evidence**: This is the leading root cause identified in `benchmark_comparison.md` (Cause 2) for the dominant failure mode (Cause 1: 54% non-termination within the 7-step budget). Directly testable and directly fixable.
- **Suggested fix**: Already the #1 Recommended Action in `benchmark_comparison.md` — source the full exemplar set from https://react-lm.github.io/ rather than the current single hand-authored example.

### O2 — Per-step trajectory logging in `evaluate.py` — Significant
- **Missing component**: `evaluate_hotpotqa_or_fever()`'s `trajectories_log` (in `evaluate.py`) stores only `{id, input, prediction, gold, n_steps, finished}` per episode — it does not persist the full `Thought i / Action i / Observation i` sequence that `TrajectoryContext.to_prompt_string()` already has in memory at the end of each episode.
- **SIR location**: Not explicitly specified as a logging requirement anywhere in `sir.json`/`architecture_plan.json`, but implied by Stage 6's own methodology (`06_results_comparator.md`, Step 5 "Root Cause Analysis") which expects enough evidence to distinguish hypotheses — and by the fact that this comparison had to *infer* the non-termination root cause (Cause 1) indirectly from `n_steps`/`finished`/empty-`prediction` fields rather than being able to directly inspect which action types were being emitted and rejected.
- **Severity**: Significant
- **Evidence**: This comparison run has zero visibility into *why* 27 episodes didn't finish (malformed actions repeatedly triggering the S1 corrective loop? genuinely difficult questions needing more search steps? something else?) beyond inference from aggregate counts — and zero trajectory data at all for the `act` method run, since only the `react` run's `trajectories.jsonl` was retained/uploaded.
- **Suggested fix**: Change `trajectories_log.append(...)` in `evaluate.py` to also store `result.trajectory.to_prompt_string()` (or `result.trajectory.lines`) per episode. Low-cost change, high value for any future Stage 6 pass.

---

## Summary

| Type | Count | Critical | Significant | Minor |
|---|---|---|---|---|
| Structural | 1 | 0 | 0 | 1 |
| Parametric | 2 | 1 | 1 | 0 |
| Omission | 2 | 1 | 1 | 0 |

Two of the five findings (P1 model substitution, O1 exemplar gap) are Critical and both trace directly to the two dominant root causes behind the measured score deviation. The single Structural finding (S1) is a net-positive deviation from plan, not a defect. Addressing O1 first is the highest-leverage next step per `benchmark_comparison.md`'s Recommended Actions.
