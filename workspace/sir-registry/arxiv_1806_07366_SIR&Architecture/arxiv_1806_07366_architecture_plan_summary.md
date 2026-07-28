# Architecture Plan — Neural Ordinary Differential Equations (arxiv_1806_07366)

## Framework
**PyTorch** (3.10+), CUDA optional (11.8+ if used), plain YAML for config.
Reasoning: the paper's own released reference implementation (`torchdiffeq`) is PyTorch-based, and
PyTorch's custom `autograd.Function` support is exactly what the adjoint sensitivity method needs.

## Repo layout
```
src/neural_ode/
├── core/
│   ├── adjoint.py         # custom autograd.Function: Appendix B/C adjoint method
│   ├── ode_solvers.py     # Euler / RK4 (from scratch) + torchdiffeq dopri5 wrapper
│   └── ode_block.py       # ODEBlock: drop-in nn.Module wrapping dynamics+solver+adjoint, tracks NFE
├── models/
│   ├── dynamics_mnist.py  # ConvDownsample + ConvODEFunc
│   ├── odenet_mnist.py    # ResNetClassifier / RKNetClassifier / ODENetClassifier (Table 1)
│   ├── cnf.py             # PlanarCNFDynamics + ContinuousNormalizingFlow (Theorem 1, Figs 4-5)
│   └── latent_ode.py      # RecognitionRNN + LatentODEFunc + Decoder + LatentODEModel (Table 2, Figs 8-10)
├── data/
│   ├── mnist_dataset.py
│   ├── toy_density_dataset.py   # Two Circles / Two Moons targets
│   └── spiral_dataset.py        # synthetic bi-directional spiral generator
├── training/
│   ├── trainer_classification.py
│   ├── trainer_cnf.py
│   ├── trainer_latent_ode.py
│   └── losses.py
├── evaluation/metrics.py     # test error, RMSE, NFE tracking
└── utils/{config.py, nfe_counter.py}
```

## Three reproducible experiment families
1. **MNIST classification** (Table 1, Figure 3) — ResNet vs. RK-Net vs. ODE-Net, all sharing the same
   downsample stem so results are directly comparable. NFE is tracked via a forward-hook counter.
2. **Continuous Normalizing Flows** (Figures 4-5) — planar-flow special case of the instantaneous
   change-of-variables theorem, trained either by density matching (KL loss, Adam, 10k iters) or
   maximum likelihood (reversible sampling).
3. **Latent ODE time series** (Table 2, Figures 8-10) — RNN encoder → ODE-solved latent trajectory →
   decoder, trained as a VAE on the synthetic bi-directional spiral dataset, evaluated at 30/50/100
   observed points for extrapolation RMSE.

## Key config assumptions (flagged `# ASSUMED` in config.yaml)
- MNIST optimizer/LR/batch size: Adam, 1e-3, 128 (**not stated in paper** — biggest reproducibility risk)
- Downsample stem output size: guessed 6×6 spatial (exact conv config unspecified)
- RNN cell type for latent-ODE encoder: GRU (unspecified in paper)
- ODE solver: dopri5 via `torchdiffeq` (paper's original experiments used implicit Adams/LSODE/VODE)
- Spiral dataset observation noise std: 0.1 (paper says "add Gaussian noise" but gives no value)
- Tolerances taken directly from the paper (high confidence): 1e-3 classification, 1e-5 density, 1.5e-8 sequence

## Dependencies
`torch`, `torchvision`, `torchdiffeq` (adaptive solver backend), `numpy`, `scipy` (reference/toy
densities), `matplotlib` (reproducing Figures 3-9), `pyyaml`, `tqdm`. Dev: `pytest`, `pytest-cov`,
`black`, `ruff`, `mypy`.

## Entrypoints
`train_classification.py`, `train_cnf.py`, `train_latent_ode.py`, `evaluate.py`, `inference.py` —
each with its own CLI schema (see `architecture_plan.json`).

## Docker
Base image `pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime`, system deps `git`/`build-essential`,
workdir `/workspace`, default CMD runs the ODE-Net classification training script.

## Risk assessment (highest severity first)
| Severity | Risk | Mitigation |
|---|---|---|
| **High** | MNIST hyperparameters not stated in paper — Table 1's 0.42% test error may not exactly reproduce | Config-driven assumed defaults, clearly marked; suggest small grid search |
| Medium | Downsample stem architecture unspecified | Config-driven kernel/stride list, defaults + shape unit test |
| Medium | Custom adjoint method is easy to get subtly wrong | Default to well-tested `torchdiffeq` backend; validate the from-scratch version with `gradcheck` |
| Low | RNN cell type unspecified | Config flag, default GRU |
| Low | Spiral noise std unspecified | Config flag, default 0.1, documented as assumption |
| Low | Spiral dataset not publicly released | Seeded generator for internal reproducibility; document expected numeric drift from Table 2 |
