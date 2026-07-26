# Verification Log

**Paper ID**: arxiv_2210_03629
**Comparison run timestamp**: 2026-07-27T00:00:00Z
**SIR version used**: 1 (`sir-registry/arxiv_2210_03629/sir.json`)
**Architecture plan version used**: 1 (`sir-registry/arxiv_2210_03629/architecture_plan.json`)

## Input Provenance

| Item | Source | Notes |
|---|---|---|
| HotpotQA ReAct metrics | User-run `evaluate.py --task hotpotqa --method react --n-eval 50`, uploaded `metrics.json` | value=0.14, n=50, elapsed ~820s |
| HotpotQA ReAct trajectories | User-uploaded `trajectories.jsonl` (50 lines) | Only `{id, input, prediction, gold, n_steps, finished}` per line — see Hallucination Report O2 for the logging gap this creates |
| HotpotQA Act metrics | User-run `evaluate.py --task hotpotqa --method act --n-eval 50`, pasted inline | value=0.08, n=50 |
| HotpotQA Act trajectories | Not provided | Root-cause analysis for the Act run is limited to the aggregate metric only |
| Paper ground truth | `sir.json → evaluation_protocol.reported_results` (SIR v1) | HotpotQA ReAct=27.4, Act=25.7 (both EM, Table 1) |

## Data Integrity Checks Performed

- Confirmed `hotpot_dev_distractor_v1.json` used for this evaluation was fetched from an alternate HuggingFace mirror (`namlh2004/hotpotqa`, 58MB via `wget`) after the primary source (`curtis.ml.cmu.edu`) and the paper's own official parquet mirror (`hotpotqa/hotpot_qa`) were both attempted at different points in this session. **This specific 50-example run's exact provenance file was not independently re-validated against the earlier-confirmed 14,810-example official parquet download** (that validation happened in a different, now-abandoned working folder earlier in the session). Flagged for manual review.
- Recomputed exact-match on the uploaded `trajectories.jsonl` independently using the same `_normalize_answer` logic as `src/react_agent/evaluation/metrics.py` — confirmed 7/50 = 0.14, matching the reported `metrics.json` value exactly. No discrepancy found.
- Computed the finished-episodes-only subset metric (7/23 = 0.304) as an independent diagnostic not present in the original `metrics.json` — this is a derived statistic for root-cause analysis, not a claim about the paper's own reported metric.

## Metrics Compared

- **Total paper metrics available** (all reported_results rows in `sir.json`, across HotpotQA/FEVER/ALFWorld/WebShop, all methods): 33
- **User results provided**: 2 (HotpotQA ReAct EM, HotpotQA Act EM)
- **Matched pairs**: 2
- **Unmatched paper metrics**: 31 (not attempted this run — includes all of FEVER, ALFWorld, WebShop, and the remaining HotpotQA methods: Standard, CoT, CoT-SC, CoT-SC→ReAct, ReAct→CoT-SC, Supervised SoTA)

## User-Reported Config Modifications (this session)

1. `model.llm_backend`/`model.provider`/`model.model_name`/`model.base_url`/`model.api_key_env_var` added to `configs/config.yaml` to support DeepSeek as an OpenAI-API-compatible provider (originally OpenAI-only; user requested this migration mid-session).
2. `model.model_name` changed twice post-deployment based on live API error messages: `deepseek-chat` (initial default) → `deepseek-v4-flash` (actual accepted value for the user's API key/provider).
3. Manual fix applied to `src/react_agent/agents/react_agent.py::run_episode()`: injection of `Question:`/`Claim:` line into `TrajectoryContext` at episode start (original generated code never wrote the question/claim into context at all — a functional bug, not a config change, found and fixed collaboratively during this session).
4. Manual fix applied to `src/react_agent/agents/react_agent.py::run_episode()`: action-name validation against `env.action_space().action_names` before dispatch (see Hallucination Report S1).
5. Manual fix applied to `src/react_agent/envs/wikipedia_env.py::_search()`: broadened exception handling for transport-level failures (rate limiting / malformed JSON responses from the `wikipedia` package), previously uncaught and crashing the whole eval batch on a single bad request.
6. `--n-eval 50` used for both runs (10% subsample of the paper's full n=500 HotpotQA set) rather than the full set, for cost/time reasons.

## Requires Manual Review: Yes

**Reasons**:
- Only 2 of 33 published paper metrics were attempted (6% coverage) — insufficient to characterize overall reproducibility of the paper's claims beyond the single HotpotQA ReAct-vs-Act comparison.
- HotpotQA data file provenance for this specific 50-example run was not independently re-verified in this session (see Data Integrity Checks above).
- No per-step trajectory data available for the Act run — root cause analysis for Act relies on directional inference from the ReAct trajectory data plus the aggregate EM score alone.
- Fixes #3–#5 above were applied manually mid-session and, per this session's own history, were twice lost when code was copied between folders and had to be reapplied — recommend a final diff/verification pass against the zip artifact currently held by the user before treating the code as stable.

## Traceability

- **User results hash** (SHA-256 over concatenated `trajectories.jsonl` + both `metrics.json` values as provided): `7fd3aefe6548e87c65f3bb1b2fd3aa32897761d55fd5a3a6d114ca4ea66b229f`
