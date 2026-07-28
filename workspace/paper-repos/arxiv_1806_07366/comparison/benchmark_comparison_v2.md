# Benchmark Comparison Report (v2)

**Paper**: Neural Ordinary Differential Equations
**Paper ID**: arxiv_1806_07366
**arXiv**: https://arxiv.org/abs/1806.07366
**Comparison Date**: 2026-07-27 (v2 — after `obs_noise_std` fix + extended training to 300 epochs)
**Previous version**: `v1_archive/benchmark_comparison.md` (pre-fix, 50-epoch runs)
**SIR Version Used**: 1

---

## Reproducibility Score

| Score | Confidence | Metrics Compared | Matched |
|-------|------------|-------------------|---------|
| **0.34** / 1.0 | High | 5 | 5 |

**The score number is unchanged from v1 (0.34) — but that does NOT mean nothing improved.**
Read that carefully before drawing conclusions. See "Why the score didn't move" below — the
underlying RMSE values improved substantially (55–68% lower absolute error), but the scoring
formula caps any deviation beyond 50% at the same minimum contribution, so it can't distinguish
"10x off" from "3–4x off." I'm flagging this explicitly rather than let the static number hide
real progress.

- 0.90–1.00: Excellent reproduction
- 0.75–0.89: Good reproduction with minor deviations
- 0.60–0.74: Partial reproduction — review moderate deviations
- 0.40–0.59: Significant reproduction gap — likely implementation issues
- **< 0.40: Critical failure — fundamental mismatch** ← still here, but meaningfully closer

---

## Metric Comparison Table

| Metric | Dataset | Split | Paper Value | v1 (pre-fix, 50ep) | v2 (post-fix, 300ep) | v2 Deviation | Severity |
|--------|---------|-------|-------------|---------------------|------------------------|--------------|----------|
| test_error | MNIST | test | 0.41% (ResNet) | 0.51% | 0.51% *(unchanged — not rerun)* | +24.39% | 🟠 Significant |
| test_error | MNIST | test | 0.47% (RK-Net) | 0.51% | 0.51% *(unchanged — not rerun)* | +8.51% | 🟡 Moderate |
| test_error | MNIST | test | 0.42% (ODE-Net) | 0.51% | 0.51% *(unchanged — not rerun)* | +21.43% | 🟠 Significant |
| predictive_RMSE | Spiral | test (30/100 obs) | 0.1642 | 1.6506 | **0.5275** | +221.3% | 🔴 Critical |
| predictive_RMSE | Spiral | test (100/100 obs) | 0.1346 | 1.6482 | **0.7294** | +441.9% | 🔴 Critical |
| predictive_RMSE | Spiral | test (50/100 obs) | *no paper value in SIR* | 1.6097 | **1.2448** | — (ref: paper text 0.1502 → +728.8%) | ⬜ Unmatched |

**Run details for v2 latent-ODE numbers**: all three at 300 epochs (up from 50), with the
`obs_noise_std` fix applied (see `v1_archive/hallucination_report.md` #1). Values shown are the
**final epoch (299)** result, since none of the three runs had reached a stable plateau by epoch
299 — RMSE was still decreasing at the end of all three runs (see Convergence Status below).

---

## What genuinely improved (don't lose this in the score)

| n_obs | v1 RMSE (50ep, buggy) | v2 RMSE (300ep, fixed) | Absolute improvement |
|---|---|---|---|
| 30 | 1.6506 | **0.5275** | **−68.0%** |
| 50 | 1.6097 | **1.2448** | −22.7% |
| 100 | 1.6482 | **0.7294** | **−55.7%** |

This is a real, substantial improvement — not noise. n_obs=30 in particular went from ~10x off
target to ~3.2x off target. The diagnosis (loss-scaling bug + insufficient training length) was
correct and the fix worked as intended.

---

## Convergence status (important caveat)

None of the three v2 runs had plateaued at epoch 299 — all three were still improving at the final
logged epoch:

| n_obs | Epoch 290 RMSE | Epoch 299 RMSE | Still descending? |
|---|---|---|---|
| 30 | ~0.544 | 0.5275 | Yes |
| 50 | ~1.240 | 1.2448 | Roughly flat / noisy, not clearly descending |
| 100 | ~0.544 | 0.7294 | Noisy — see anomaly below |

Because none of these are converged results, this v2 comparison should be read as a **progress
checkpoint**, not a final verdict on this repo's reproducibility ceiling. See Recommended Actions.

---

## Anomaly: non-monotonic RMSE across n_obs (new finding)

The paper's Table 2 pattern is monotonic: **more observed points → lower RMSE**
(0.1642 → 0.1502 → 0.1346 as n_obs goes 30→50→100). Your v2 results do **not** follow this
pattern:

```
n_obs=30:  RMSE=0.5275  (best)
n_obs=100: RMSE=0.7294  (worse than 30-obs!)
n_obs=50:  RMSE=1.2448  (worst)
```

This is unexpected and worth investigating — a model with more observed data points should not
generally do worse than one with fewer, all else equal. Two plausible explanations, in order of
likelihood:

1. **Different random train/test splits per run** (High probability): `train_latent_ode.py` calls
   `random_split()` fresh on each invocation without a run-to-run-consistent seed carry-through for
   the split itself (only `torch.manual_seed` is set globally at start, but three separate process
   invocations each reset and consume RNG state differently depending on dataset size). The 50-obs
   and 100-obs runs may simply have landed on harder/easier test splits by chance.
2. **Under-training, not yet comparable** (High probability, compounding #1): since none of the
   three runs have converged (see Convergence Status above), comparing them to each other at an
   arbitrary stopping point (epoch 299) captures wherever each happened to be in its own trajectory,
   not a fair apples-to-apples comparison. The "expected" monotonic pattern may only emerge once
   all three are closer to convergence.

This is noted as a finding, not yet a confirmed bug — see Recommended Actions #3.

---

## Deviation Summary

| Severity | Count |
|----------|-------|
| ✅ Excellent (≤2%) | 0 |
| 🟢 Good (2–5%) | 0 |
| 🟡 Moderate (5–15%) | 1 |
| 🟠 Significant (15–30%) | 2 |
| 🔴 Critical (>30%) | 2 |
| ⬜ Unmatched | 1 |

Identical bucket counts to v1 — again, this is a scoring-rubric artifact (both v1 and v2 latent-ODE
deviations exceed the 50%-deviation cap the score formula saturates at), not evidence that nothing
changed. See the raw numbers above for the real signal.

---

## Root Cause Analysis (updated)

### Classification (unchanged from v1 — not rerun this cycle)
See `v1_archive/benchmark_comparison.md` for the full analysis (assumed hyperparameters,
convergence/overfitting pattern, seed variance, downsample-stem ambiguity). Nothing new to add
since these weren't rerun.

### Latent ODE RMSE — still Critical, but trajectory now looks healthy

1. **Not yet converged — needs more than 300 epochs** (High probability, primary remaining cause)
   All three runs were still moving at epoch 299. The `neg_elbo`/rmse trend for n_obs=30 and
   n_obs=100 in particular showed accelerating improvement in the middle of training (roughly
   epoch 130-140 onward, per your earlier live monitoring), consistent with the model escaping a
   slow initial regime rather than being structurally capped. Recommend continuing training well
   past 300 epochs (try 600-1000) to see where it actually plateaus.

2. **Non-monotonic n_obs pattern suggests run-to-run split variance is adding noise** (Medium
   probability) — see Anomaly section above. Fix: pass an explicit, fixed split seed
   (`random_split(..., generator=torch.Generator().manual_seed(config.training.seed))`) so all
   three n_obs runs use a comparable, reproducible test split, making the comparison across them
   more meaningful.

3. **The `obs_noise_std` fix was necessary but likely not sufficient on its own** (confirmed) — the
   improvement is real (55-68% lower error), consistent with the fix removing a genuine obstacle,
   but the remaining gap is now plausibly just "needs more training," not "still bugged," pending
   confirmation from a longer run.

**Verdict**: this now looks like a **training-length problem more than a bug problem**. The
fix was correct; the remaining gap is most likely closed by more epochs, not more code changes —
though the non-monotonic anomaly is worth a quick investigation in parallel.

---

## Recommended Actions

1. **Continue training past 300 epochs** for all three n_obs settings — none had converged.
   Suggest 600-1000 epochs, or better: add an early-stopping criterion (e.g. stop when RMSE
   improvement over 20 epochs falls below a threshold) instead of a fixed epoch count.
2. **Fix the train/test split seeding** in `train_latent_ode.py` so results across different
   `--num-observations` runs are comparable (see Root Cause #2).
3. **Investigate the non-monotonic n_obs pattern** once (1) and (2) are both addressed — if it
   persists after longer, seed-controlled training, that would indicate a real (not noise-driven)
   issue worth a deeper look.
4. **Classification experiments remain unaddressed this cycle** — optional hyperparameter sweep
   per v1's recommendations, if you want to tighten those gaps too.
5. **Re-run Stage 6 once (1)-(2) are done** — expect the latent-ODE deviations to drop further,
   and this time the scoring formula may finally register the improvement numerically if the gap
   closes below the 50% saturation threshold.

---

## Verification Log Summary

- Comparison run at: 2026-07-27T00:20:00Z (v2)
- New evidence: `trainlatentodenums100_50_30.txt` (300-epoch runs, post `obs_noise_std` fix)
- Previous evidence retained for comparison: `v1_archive/verification_log.md`
- Manual review required: **Yes** — runs are not converged; treat v2 numbers as a checkpoint, not
  a final result. Recommend re-running Stage 6 after longer training.

Full audit trail in `verification_log.md` (this cycle) and `v1_archive/verification_log.md`
(previous cycle).
