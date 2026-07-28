# Hallucination Report (v2) — arxiv_1806_07366

**Comparison date**: 2026-07-27 (v2 — after `obs_noise_std` fix + 300-epoch retraining)
**Previous version**: `v1_archive/hallucination_report.md`

---

## Status of v1 findings

### 1. `obs_noise_std` not threaded from config into ELBO loss — **RESOLVED**

- **v1 status**: Critical
- **v2 status**: **Fixed and confirmed effective.** User applied the recommended change in
  `trainer_latent_ode.py` (passing `config.data.spiral_observation_noise_std` into
  `latent_ode_negative_elbo`). Post-fix + longer training, RMSE improved 55-68% across n_obs
  settings (see `benchmark_comparison.md`). This confirms the original diagnosis was correct.
- **Residual concern**: none — this specific hallucination is closed out.

### 2. MNIST optimizer/LR/batch-size assumptions — **UNCHANGED (not re-evaluated)**

- **v1 status**: Minor
- **v2 status**: Not re-evaluated — classification experiments were not rerun this cycle.
- Still recommend the hyperparameter sweep from v1 if/when classification is revisited.

### 3. `training.epochs = 50` default not validated — **PARTIALLY RESOLVED, NEW FINDING**

- **v1 status**: Moderate (omission)
- **v2 status**: User retrained at 300 epochs, which is a large improvement over 50, and
  substantial gains resulted. However: **all three v2 runs were still descending at epoch 299** —
  the omission isn't fully resolved so much as reduced. 300 was clearly a better choice than 50,
  but is very likely still short of true convergence.
  Fix (updated): re-run with 600-1000 epochs, or better, implement early stopping based on RMSE
  plateau detection rather than guessing a fixed epoch count.

---

## New finding in v2

### 4. Unseeded / inconsistent train-test split across separate `--num-observations` runs — **NEW, Moderate**

- **Type**: parametric / procedural
- **Severity**: Moderate
- **Location**: `train_latent_ode.py`, the `random_split(full_dataset, [n_train, n_test])` call.
- **Evidence**: while `set_seed(config.training.seed)` is called once at the start of each script
  invocation (seeding Python/NumPy/PyTorch global RNGs), each of the three `--num-observations`
  values was run as a **separate process invocation** (`100`, then `50`, then `30`). Because
  `random_split` consumes RNG state that depends on dataset construction order and size — which
  differs between the three runs (different `n_obs` changes how much of the RNG stream is consumed
  before the split call) — the actual train/test split composition is not guaranteed to be
  consistent or comparable across the three runs, even with the same nominal seed.
- **Symptom this likely explains**: the non-monotonic RMSE pattern across n_obs (30 < 100 < 50 in
  final RMSE, whereas the paper's Table 2 is monotonically decreasing: 30 > 50 > 100). This could
  be split-composition luck rather than a real model behavior difference.
- **Suggested fix**:
  ```python
  gen = torch.Generator().manual_seed(config.training.seed)
  train_set, test_set = random_split(full_dataset, [n_train, n_test], generator=gen)
  ```
  placed identically in all three runs, so the split is deterministic and comparable across
  `--num-observations` invocations (assuming the underlying `full_dataset` construction is also
  deterministic prior to subsampling, which it is, via `SpiralDataset(..., seed=config.training.seed)`).
- **Confidence this is the actual cause**: Medium — plausible and consistent with the data, but not
  yet confirmed by rerunning with the fix. Listed as a recommended action, not a proven root cause.

---

## Summary (v2)

| # | Type | v1 Severity | v2 Status |
|---|------|-------------|-----------|
| 1 | Parametric (obs_noise_std) | Critical | ✅ **Resolved, confirmed effective** |
| 2 | Parametric (MNIST hyperparams) | Minor | Unchanged, not re-evaluated |
| 3 | Omission (epoch count) | Moderate | Partially resolved — needs more epochs still |
| 4 | Parametric (split seeding) | *(new)* | Moderate — recommended fix, not yet applied |

**Bottom line**: the primary Critical hallucination from v1 is confirmed fixed and effective. The
remaining gap to the paper's target values is now most plausibly explained by incomplete
convergence (needs more epochs) plus a newly identified, lower-severity split-seeding issue that
may be adding noise to the n_obs comparison. Neither remaining issue looks like a fundamental
architecture bug at this point.
