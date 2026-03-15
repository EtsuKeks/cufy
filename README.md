# cufy

A quantitative finance framework for option pricing and model calibration with GPU acceleration via PyTorch.

## Current State

**Backend:** PyTorch only. No JAX, no EvoJAX.

**Device:** Single-device only. The framework assumes exclusive ownership of one GPU (or CPU). No multi-GPU, no distributed / cluster-wide execution.

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
- **MLflow integration** — per-fold param/metric logging, vol surface artifacts, aggregate metrics
- **Structured logging** — replace `print` calls (e.g. `memory_probation`) with a proper logger so log level and output can be controlled at integration time
- **Optuna pruning** — shared explore phase across tuner trials + per-iteration pruning in refine; requires splitting calibration into explore-only / refine-only phases
- **Reduce CPU↔GPU transfers** — propagate `TorchOptionBatch` through `Runner` and engine so numpy↔tensor conversion happens once per fold rather than on every call