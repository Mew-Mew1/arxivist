# Verification Log (v2) — arxiv_1806_07366, Stage 6 (Results Comparator)

**Run timestamp**: 2026-07-27T00:20:00Z
**Comparison version**: 2 (previous cycle archived in `v1_archive/`)
**SIR version compared against**: `sir-registry/arxiv_1806_07366/sir.json`, version 1
**Architecture plan compared against**: `sir-registry/arxiv_1806_07366/architecture_plan.json`, version 1

---

## Inputs received (this cycle)

| Source | Content | SHA-256 |
|---|---|---|
| `trainlatentodenums100_50_30.txt` (v2 upload) | Full stdout of `train_latent_ode.py --num-observations {100,50,30} --epochs 300`, post `obs_noise_std` fix, 300 epochs each | `b8487fcd71a49d9da971130f156cd34693e75d769b0aaa584b85761dfd9e987e` |

Classification results were **not** resupplied this cycle; v1's classification numbers (test_error
for ResNet/RK-Net/ODE-Net) are carried forward unchanged into the v2 comparison table.

---

## Code change verified

User reported applying this change to `src/neural_ode/training/trainer_latent_ode.py`:
```python
loss = latent_ode_negative_elbo(
    x_hat, x, mu_z0, logvar_z0, obs_noise_std=self.config.data.spiral_observation_noise_std
)
```
This was not independently re-verified against a fresh copy of the user's local repo file (no file
upload of the changed source was provided), but the resulting `neg_elbo` magnitudes in the new logs
(~5,000-30,000 range vs. ~950-3,300 in v1 logs, roughly a 9x scale shift) are consistent with the
expected effect of changing `obs_noise_std` from 0.3 to 0.1 in the Gaussian log-likelihood term
(`1/obs_noise_std²` scaling), corroborating that the fix is active.

---

## Extraction method

Same regex-based parsing as v1:
```
r'\[epoch (\d+)\] neg_elbo=([\d.\-]+) rmse=([\d.]+)'
```
applied separately to each of the three `--num-observations` blocks (split on the invocation
command line). **Final epoch (299)** values were used for the v2 comparison table, rather than
"best across the run," because none of the three runs had plateaued — using "best" would have
somewhat flattered results that were still actively improving, and "final" is a more conservative,
defensible choice for a still-converging run.

---

## Parsed values (v2, full)

| Run | Epochs logged | RMSE @ epoch 0 | RMSE @ epoch 299 (final) | Still descending at end? |
|---|---|---|---|---|
| n_obs=100 | 300 | 1.6990 | 0.7294 | Yes (noisy but net downward) |
| n_obs=50 | 300 | 1.6536 | 1.2448 | Ambiguous — flattest of the three |
| n_obs=30 | 300 | 1.7036 | 0.5275 | Yes, clearly |

---

## Ground-truth values used (from sir.json) — unchanged from v1

| Metric | Dataset | Split | Value | is_primary |
|---|---|---|---|---|
| predictive_RMSE | Spiral | test (30/100 obs) | 0.1642 | true |
| predictive_RMSE | Spiral | test (100/100 obs) | 0.1346 | true |

Same SIR gap as v1 remains unaddressed: no ground-truth value for 50/100 obs in `sir.json`
(paper text value 0.1502 used for reference only, not as an official matched metric).

---

## Score computation (v2)

```
Unchanged formula from v1. Per-metric scores:

  ResNet:        1 - min(24.39/50, 1)  = 0.5122   (unchanged, not rerun)
  RK-Net:        1 - min(8.51/50, 1)   = 0.8298   (unchanged, not rerun)
  ODE-Net:       1 - min(21.43/50, 1)  = 0.5714   (unchanged, not rerun)
  Latent ODE 30: 1 - min(221.3/50, 1)  = 0.0000   (still saturated -- deviation still > 50%)
  Latent ODE100: 1 - min(441.9/50, 1)  = 0.0000   (still saturated -- deviation still > 50%)

base_score = mean(above) = 0.3827   (IDENTICAL to v1 -- both latent-ODE terms remain
                                      saturated at the 50%-deviation cap despite large
                                      absolute improvement)

sir_confidence_penalty = 0.0455   (unchanged, SIR not modified between cycles)
unmatched_penalty = 0.0000

reproducibility_score = max(0, 0.3827 - 0.0455) = 0.3372 -> 0.34   (same as v1)
```

**Important**: the score formula's `min(abs(pct)/50, 1)` saturation means any deviation ≥50% scores
identically to a deviation of 1000%. This is a known limitation of the current scoring rubric — it
is well-suited to distinguishing "close" reproductions but has no resolution once a metric is
already badly off. The *raw* metric values (see `benchmark_comparison.md`) are the more informative
signal for tracking progress in this situation, not the composite score.

---

## Manual review flag

**Manual review required: YES** — for a different reason than v1.

v1 reason (bug identification) is now resolved. v2 reason: **the compared runs are not converged**.
Presenting epoch-299 numbers as if they were final results would overstate confidence in the actual
reproducibility gap — the true gap could be smaller (if training continues to improve) or the
current trend could stall. Recommend treating this v2 report as a progress checkpoint and
re-running Stage 6 after longer training and the split-seeding fix (see `hallucination_report.md`
#4).

---

## Traceability

- SIR: `sir-registry/arxiv_1806_07366/sir.json` (v1, unchanged this cycle)
- Architecture plan: `sir-registry/arxiv_1806_07366/architecture_plan.json` (v1, unchanged)
- v1 comparison artifacts: `comparison/v1_archive/`
- This report: `comparison/benchmark_comparison.md`, `comparison/reproducibility_score.json`,
  `comparison/hallucination_report.md`, `comparison/verification_log.md` (v2, this file)
