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
- GPU-accelerated model calibration (Gradient-based, Grid Search)
- Backtesting engine
