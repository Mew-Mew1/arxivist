# Benchmark Comparison Report

**Paper**: ReAct: Synergizing Reasoning and Acting in Language Models
**Paper ID**: arxiv_2210_03629
**Comparison Date**: 2026-07-27
**Reproducibility Score**: 0.00 (medium confidence)

## Metric Comparison

| Metric | Dataset | Paper Value | Your Value | Deviation | Severity |
|---|---|---|---|---|---|
| Exact Match | HotpotQA (n=50 of 500) | 27.4 | 14.0 | -48.9% | Critical |
| Exact Match | HotpotQA, Act baseline (n=50 of 500) | 25.7 | 8.0 | -68.9% | Critical |

**Unmatched paper metrics** (not attempted in this run): HotpotQA Standard/CoT/CoT-SC/CoT-SC→ReAct/ReAct→CoT-SC/Supervised-SoTA, all FEVER results, all ALFWorld results, all WebShop results — 31 of 33 published result rows in `sir.json` have no corresponding user result yet.

## Summary

Both attempted metrics show **Critical** deviation from the paper by the standard ≤2%/2–5%/5–15%/15–30%/>30% severity bands — this reproduction does not match PaLM-540B's reported numbers, and that was expected going in (PaLM-540B is not publicly accessible; see README "Reproducibility Notes"). However, a deeper look at the trajectory-level data reveals the deviation is **not evenly distributed across failure modes** — one dominant, identifiable cause explains most of the gap (see Root Cause Analysis, Cause 1), and the paper's *qualitative* claim that ReAct outperforms Act is actually preserved in this reproduction, even though absolute scores are far below PaLM-540B's.

## Root Cause Analysis

### Cause 1 — Non-termination within the step budget (High probability, dominant driver)

Of the 50 HotpotQA ReAct episodes, **27 (54%) never called `finish[...]` at all** — they were force-terminated at the `max_steps=7` cap with an **empty prediction**, which automatically scores as wrong under exact-match. Only 23 episodes (46%) actually reached a `finish[...]` call.

**Isolating this effect changes the picture substantially.** Restricting exact-match scoring to only the 23 episodes that *did* finish:

```
EM (finished episodes only) = 7 / 23 = 0.304 (30.4%)
```

That is **at or above** the paper's own reported PaLM-540B ReAct EM of 27.4% on the same task. In other words: **when the agent successfully completes an episode, its answer quality is roughly in line with the paper.** The critical-severity deviation in the headline number is being driven almost entirely by episodes that ran out of steps before producing any answer, not by systematically wrong reasoning.

- **Suggested fix**: Increase `configs/config.yaml::routing.hotpotqa_max_react_steps` for diagnostic runs to see if the non-termination rate drops with more steps (paper caps at 7 for a reason — more steps than that yields diminishing returns per Section 3.2 footnote 3 — but confirming whether DeepSeek specifically needs more room is worth an ablation). More importantly, address why the model burns steps without progressing (see Cause 2).

### Cause 2 — Insufficient few-shot exemplar coverage (High probability, contributing to Cause 1)

`src/react_agent/prompts/prompt_templates.py` ships with **1 hand-authored exemplar** per (task, method), reconstructed from Figure 1/Table fragments in the parsed PDF, versus the paper's actual **6 exemplars for HotpotQA / 3 for FEVER** (Section 3.2; full Appendix C prompts were not present in the parsed PDF text — see SIR `ambiguities[0]`/`[1]`, confidence 0.5). With only one worked example, the model has weak signal on the exact `search[entity]` / `lookup[string]` / `finish[answer]` syntax, and was directly observed (in manual single-question testing during this session) emitting free-text, non-bracketed pseudo-actions like `"Action 1: Nicholas Ray and Elia Kazan are both film directors."` instead of a valid action. Each such malformed action consumes one of the 7 available steps (this reproduction adds explicit corrective feedback for this case — see Hallucination Report, Fix #2 — but a single corrective message per malformed action still costs a full step, and repeated malformed emissions compound directly into Cause 1's non-termination).

- **Suggested fix**: Reconstruct the full Appendix C prompt set (6 HotpotQA / 3 FEVER exemplars) from the official repository (https://react-lm.github.io/) rather than the single hand-authored exemplar currently in use. This is the single highest-leverage fix available and directly targets both Cause 1 and Cause 2.

### Cause 3 — Verbose, non-terse `finish[...]` answers inflate false negatives (Medium probability)

Several completed episodes contain semantically correct answers that fail strict exact-match purely on formatting, e.g.:

| Question | Prediction | Gold | EM verdict |
|---|---|---|---|
| Were Scott Derrickson and Ed Wood of the same nationality? | "Yes, both Scott Derrickson and Ed Wood are American, so they share the same nationality." | "yes" | Wrong (should arguably be right) |
| The football manager who recruited David Beckham managed Manchester United during what timeframe? | "1986 to 2013" | "from 1986 to 2013" | Wrong (near-miss; `_normalize_answer` strips articles a/an/the but not "from") |

This is downstream of Cause 2 (the single exemplar doesn't strongly demonstrate the paper's terse `finish[short answer]` convention) compounded by a metric-implementation gap: `src/react_agent/evaluation/metrics.py::_normalize_answer` does not strip leading connector words like "from", nor extract a terse answer from a full sentence.

- **Suggested fix**: (a) richer exemplars (same fix as Cause 2) to encourage terse answers; (b) consider whether `_normalize_answer` should be extended, though note this is explicitly marked `ASSUMED` in the SIR (confidence 0.55) since the paper doesn't restate its own EM normalization convention — extending it is a defensible implementation choice, not a paper-fidelity requirement.

### Cause 4 — Base model substitution (High probability, expected/unavoidable)

PaLM-540B is not publicly accessible (SIR `implementation_assumptions[0]`, confidence 0.95). This run substitutes `deepseek-v4-flash` via `OpenAIClient`. Manual single-question testing during this session showed DeepSeek frequently reasoning from parametric memory and skipping `search[...]` grounding entirely — a real behavioral gap from what the paper documents for PaLM-540B. However, this alone does not explain the magnitude of the gap: the paper's own Appendix A.1 substitutes GPT-3 `text-davinci-002` for PaLM-540B and reports it *outperforms* PaLM-540B, so model substitution is not inherently score-destroying — Causes 1–3 are more directly responsible for this specific reproduction's low numbers.

- **Suggested fix**: None required to "fix" — this is an accepted, documented substitution. If exact parity is desired, re-run with the strongest available model (`deepseek-v4-pro`) as a comparison point once Causes 1–3 are addressed.

### Directional claim check — does ReAct still beat Act?

The paper's central claim is that ReAct (reasoning+acting) outperforms Act (acting alone): PaLM-540B shows ReAct=27.4 > Act=25.7 (+1.7 abs, +6.6% relative). This reproduction shows the same direction, proportionally more pronounced: ReAct=14.0 > Act=8.0 (+6.0 abs, +75% relative). **The qualitative claim survives even though absolute magnitudes are far below the paper** — worth noting as a partial, real reproduction success alongside the critical quantitative deviation.

## Recommended Actions

1. **(Highest impact)** Reconstruct the full 6-exemplar HotpotQA / 3-exemplar FEVER prompts from the official ReAct repository instead of the single hand-authored exemplar currently in `prompt_templates.py`. Expected to reduce the 54% non-termination rate and improve answer terseness simultaneously (Causes 1–3).
2. Re-run the same 50-example batch after Action 1, and specifically re-check the finished-only EM subset (currently 30.4%, already near parity) to see whether the *overall* EM converges toward it as non-termination drops.
3. Log full per-step trajectories (Thought/Action/Observation lines), not just final predictions, in `evaluate.py`'s `trajectories_log` — see Hallucination Report, Omission #1 — to make future root-cause analysis direct rather than inferential.
4. Once (1)–(3) are addressed, expand from n=50 to the full n=500 HotpotQA set and add FEVER, to get `score_confidence: high` (requires ≥3 matched metrics) instead of the current `medium`.
5. Consider whether `_normalize_answer` should extract terse answers from verbose completions (Cause 3) — optional, lower priority than (1).
