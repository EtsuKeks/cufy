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

### Infrastructure
- **MLflow integration** — per-fold param/metric logging, vol surface artifacts, aggregate metrics
- **Structured logging** — replace `print` calls (e.g. `memory_probation`) with a proper logger so log level and output can be controlled at integration time
- **Reduce CPU-GPU transfers** — propagate `TorchOptionBatch` through `Runner` and engine so numpy-tensor conversion happens once per fold rather than on every call
- **MPI / multi-GPU support** — distributed scoring and calibration across a GPU cluster via MPI

### Calibration
- **Optuna pruning** — shared explore phase across tuner trials + per-iteration pruning in refine; requires splitting calibration into explore-only / refine-only phases
- **EvoSAX calibration** — port the framework to a JAX backend and add EvoSAX-based calibrators (NES, CMA-ES) for rigorous comparison against the PyTorch grid-search methods
- **Natural NES+EDA hybrid** — run multiple NES trials simultaneously, partitioned into K clusters for population-level diversity
- **CPU-baseline calibrators** — add BS, SABR, Heston calibrators wrapping `fypy` to benchmark against existing CPU-based methods and demonstrate the GPU speedup

### Models
- **Heston model** — stochastic vol with mean reversion; analytical European pricing via characteristic function with logarithmic catch handling
- **Hull-White analytical model** — interest rate model for discount curve fitting
- **Historical fit for analytical models** — calibrate parameters that are identifiable from underlying price history (e.g. drift, vol-of-vol) directly from discounted underlying price series, reducing the degrees of freedom left to the options calibrator
- **Gaussian process model** — non-parametric vol surface model; variant that takes predictions of analytical models as input features

### Contracts & hedging
- **American and Asian contract support** — add `models/analytical/american/` and `models/analytical/asian/` with numerical pricers (LSM Monte Carlo, PDE, Turnbull-Wakeman approximation); make calibrator `score_fn` injectable so price-RMSE or contract-specific metrics can replace the default IV-RMSE
- **Hedging backtest** — extend `Model` with optional Greeks (delta, vega, gamma); add a separate `backtest/hedging/` engine that recalibrates at each step, rebalances the hedging portfolio, and attributes P&L; requires a different data folding philosophy (fit today, evaluate over full TTM horizon)