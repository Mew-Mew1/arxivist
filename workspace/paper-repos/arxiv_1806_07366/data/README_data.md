# Data Notes

This repo uses three datasets, corresponding to the paper's three experiment
families.

## 1. MNIST (classification, Table 1)
Publicly available. Auto-downloaded by `torchvision.datasets.MNIST` the first
time `train_classification.py` or `data/download.py` is run. No manual setup
needed.

## 2. Two Circles / Two Moons (CNF density experiments, Figures 4-5)
Synthetic target distributions defined analytically in
`src/neural_ode/data/toy_density_dataset.py`. Generated on-the-fly; nothing
to download.

## 3. Bi-directional spiral dataset (latent ODE, Table 2, Figures 8-10)
**Not publicly released by the paper's authors** (Section 5.1 describes the
generation procedure — 1000 trajectories, 100 timesteps, half clockwise/half
counter-clockwise, Gaussian observation noise — but no data file is provided
or linked).

This repo regenerates a statistically equivalent dataset from that
description via `src/neural_ode/data/spiral_dataset.py::SpiralDataset`, using
a fixed seed (`config.training.seed`) for internal run-to-run reproducibility.

**Known limitation**: because the exact original trajectories are not
available, and the paper does not specify the Gaussian observation noise
standard deviation (we assume 0.1, `config.data.spiral_observation_noise_std`
— see `sir.json` implementation_assumptions), the absolute RMSE values in
Table 2 (0.1346–0.1642 for the latent ODE) should be treated as targets to
approach, not numbers this repo is guaranteed to reproduce exactly.

To generate the spiral dataset ahead of time:
```
python data/download.py --generate-spirals
```
