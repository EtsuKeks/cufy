# cufy

A quantitative finance framework for option pricing and model calibration with GPU acceleration via PyTorch.

## Current State

**Backend:** PyTorch only. No JAX, no EvoJAX.

**Device:** Single-device only. The framework assumes exclusive ownership of one GPU (or CPU). No multi-GPU, no distributed / cluster-wide execution.

**Contracts:** European options only. No American, Asian, or exotic contract types.

**Models:** Analytical closed-form models only. No neural network models, no Gaussian process approaches, no Monte Carlo / simulation-based pricing.

## Installation

First, install PyTorch according to your hardware (CPU, CUDA, MPS) from the [official website](https://pytorch.org/get-started/locally/).

For example, for CUDA 12.8:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Then, install cufy:
```bash
pip install cufy
```

## Planned

### Backtesting
- **Speculative backtest engine** — fold-by-fold calibration + pricing loop (`DataSource`, `FoldedDataSource`, `ConsecutiveFolds`); evaluate next-day IV fit across rolling windows
- **MLflow integration** — per-fold param/metric logging, vol surface artifacts, aggregate metrics
- **Hedging backtest engine** — separate `backtest/hedging/` engine; requires fold philosophy change (fit today, evaluate over full TTM horizon)

### Infrastructure
- **Reduce CPU-GPU transfers** — propagate `TorchOptionBatch` through `Runner` so numpy-tensor conversion happens once per fold rather than on every call
- **MPI / multi-GPU support** — distributed scoring and calibration across a GPU cluster via MPI
- **Documentation and API exports** — write detailed docstrings with academic paper references (Hagan, Merton, Hull-White, Jamshidian, etc.) for all models and methods; populate `__init__.py` files across the project to expose clean and convenient public APIs

### Calibration
- **Optuna pruning** — shared explore phase across tuner trials + per-iteration pruning in refine; requires splitting calibration into explore-only / refine-only phases
- **EvoSAX calibration** — port the framework to a JAX backend and add EvoSAX-based calibrators (NES, CMA-ES) for rigorous comparison against the PyTorch grid-search methods
- **Natural NES+EDA hybrid** — run multiple NES trials simultaneously, partitioned into K clusters for population-level diversity
- **CPU-baseline calibrators** — add BS, SABR, Heston calibrators wrapping `fypy` to benchmark against existing CPU-based methods and demonstrate the GPU speedup

### Models
- **Heston model** — stochastic vol with mean reversion; analytical European pricing via characteristic function with logarithmic catch handling
- **SABR model** — NOTE: Currently uses unnormalized prices internally which can cause numerical instability (F*K explosion) for high-priced assets. Needs a robust solution to respect normalized prices while preserving the scale-dependent alpha parameter.
- **Bates model** — Heston + Poisson jump diffusion; adds jump intensity, mean jump size and jump vol to capture short-term smile and crash risk
- **Variance Gamma (VG)** — pure-jump Lévy process; three-parameter closed-form pricing via characteristic function; captures skew and excess kurtosis without stochastic vol
- **Rough Heston / rough Bergomi** — fractional Brownian motion drives instantaneous vol; fits the observed term-structure of ATM skew that classical models cannot reproduce; pricing via Fourier inversion (rough Heston) or Monte Carlo (rough Bergomi)
- **Historical fit for analytical models** — calibrate parameters that are identifiable from underlying price history (e.g. drift, vol-of-vol) directly from discounted underlying price series, reducing the degrees of freedom left to the options calibrator
- **Gaussian process model** — non-parametric vol surface model; variant that takes predictions of analytical models as input features

### Contracts & hedging
- **American and Asian contract support** — add `models/analytical/american/` and `models/analytical/asian/` with numerical pricers (LSM Monte Carlo, PDE, Turnbull-Wakeman approximation); make calibrator `score_fn` injectable so price-RMSE or contract-specific metrics can replace the default IV-RMSE; for contracts where a single forward pass is expensive (Monte Carlo simulation), grid-search calibrators are wasteful — consider sequential Bayesian calibrators (Optuna GPSampler / CMA-ES via ask-tell) that minimise the number of forward evaluations
- **Hedging support** — extend `Model` with optional Greeks (delta, vega, gamma)