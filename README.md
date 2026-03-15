# cufy

A quantitative finance framework for option pricing and model calibration with GPU acceleration via PyTorch.

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

## Features
- Option pricing (Black-Scholes, SABR)
- GPU-accelerated model calibration (Gradient-based, Grid Search with LM and Puzir refinement)
- Hyperparameter tuning via Optuna (TPE, CMA-ES, persistent study storage)
- Backtesting engine

## Planned
- **MLflow integration** — per-fold param/metric logging, vol surface artifacts, cross-fold aggregate metrics
- **Optuna pruning** — shared explore phase across tuner trials (explore result is hyperparameter-independent), per-iteration pruning in refine phase (Puzir, LM, Gradient)
