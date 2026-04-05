# cufy

A quantitative finance framework for option pricing and model calibration with GPU acceleration via PyTorch.

## Current State

**Calibration Philosophy:** Currently, the framework is strictly focused on **time-series calibration** for predictive tasks (Alpha Signal Evaluation). To ensure stability, prevent overfitting, and maintain the physical meaning of parameters across time, we avoid fitting state variables (like $v_0$ in Heston/Bates) as free parameters. Instead, we derive them directly from market observables (e.g., using $\sigma_{ATM}$ as a deterministic proxy).

**Backend:** PyTorch only for modeling and pricing, JAX for evolutionary calibration (EvoSAX).

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
- **Speculative backtest engine** — fold-by-fold calibration + pricing loop; possibly synthetic dealing
- **MLflow integration** — param/metric logging, vol surface artifacts, aggregate metrics
- **Hedging backtest engine** — separate hedging engine; requires fold philosophy change (fit today, evaluate over full TTM horizon), new models added (convenient for hedging purposes)

### Infrastructure
- **Documentation and API exports** — write detailed docstrings with academic paper references (Hagan, Merton Bates, etc.) for all models and methods; populate init files across the project to expose clean and convenient public APIs
- **Reduce CPU-GPU transfers** — propagate `TorchOptionBatch` through `Runner` so numpy-tensor conversion happens once per fold rather than on every call
- **MPI / multi-GPU support** — distributed scoring and calibration across a GPU cluster via MPI

### Calibration
- **Target Abstraction** — currently, calibrators hardcode `weighted_iv` as the loss function. This needs to be abstracted so models can be calibrated directly on prices or other metrics. Crucially, the target must support both scalar scoring (for black-box optimizers like Puzir/EvoSax) and residual vector generation (for gradient-based methods like Levenberg-Marquardt)
- **Optuna pruning** — shared explore phase across tuner trials + per-iteration pruning in refine; requires splitting calibration into explore-only / refine-only phases
- **JAX Migration** — port the core framework and pricing models to a JAX backend to fully utilize XLA compilation and unify the codebase. This will eliminate the `backends/torch` abstraction layer entirely, promoting JAX arrays to the `core` level. Consequently, the "CPU-GPU transfers" bottleneck will disappear, as data will reside natively on the device from the start. That is, core abstractions will utilize JAX arrays instead of numpy, with only weight functionality left, filtering will be removed and **become a part of DataSource contract**.

### Models
- **Gaussian process model** — non-parametric vol surface model; variant that takes predictions of analytical models as input features
- **Variance Gamma (VG)** — pure-jump Lévy process; three-parameter closed-form pricing via characteristic function; captures skew and excess kurtosis without stochastic vol
- **Rough Volatility (Rough Heston / rBergomi)** — cutting-edge models driven by fractional Brownian motion. Perfectly captures the power-law explosion of ATM skew at short maturities, which is critical for highly volatile crypto markets
- **Path-Dependent Volatility (PDV / Guyon-Lipton)** — advanced frontier models that construct the instantaneous volatility surface directly from the historical path of the underlying asset, completely eliminating the need for unobservable state variables
- **Historical fit for analytical models** — calibrate parameters that are identifiable from underlying price history (e.g. drift, vol-of-vol) directly from discounted underlying price series, reducing the degrees of freedom left to the options calibrator
- **Advanced Quadrature Methods** — implement Filon's Quadrature (for deep OTM/ITM options with highly oscillatory integrands) and Double-Exponential (Tanh-Sinh) Quadrature (for robust, cryptographically high-precision ground truth testing of other algorithms). Note: the current architecture hardcodes both the quadrature rule (Gauss-Legendre / Simpson) and the pricing formula (Attari). Supporting alternative quadratures and alternative Fourier pricing methods (COS, Carr-Madan FFT, frame projection) will require a deeper architectural change — decoupling quadrature strategy, pricing formula, and model characteristic function into separate, independently swappable components.
- **Integration Optimization (Heston/Bates)** — currently, the characteristic function is evaluated for every option individually ($N$ times). This, in theory, avoids JAX recompilation issues and OOM risks, but leaves a huge speedup on the table for datasets with few unique expirations ($U \ll N$). Future versions should group evaluations by unique TTM ($U$ times) while safely handling memory probation and dynamic shapes.
- **Analytical Jacobians & Gradients** — currently Jacobian (and potentially gradients) are vomputed via `torch.func.jacfwd` (forward-mode AD), which requires one forward pass per parameter. For models with known closed-form derivatives this is unnecessary overhead which should be reduced
- **Neural SDEs / Neural Stochastic Volatility** — hybrid frontier models that use neural networks to learn the drift and diffusion coefficients of the volatility process directly from panel data, bridging the gap between rigorous analytical SDEs and deep learning

### Contracts & hedging
- **American and Asian contract support** — add models variations with numerical pricers (LSM Monte Carlo, PDE, Turnbull-Wakeman approximation); will need loss function ot be injectable so price-RMSE or contract-specific metrics can replace the default IV-RMSE; for contracts where a single forward pass is expensive (Monte Carlo simulation), grid-search exhaustive calibrators are wasteful — consider other approaches like sequential Bayesian calibrators (Optuna GPSampler) that minimise the number of forward evaluations
- **Hedging support** — extend `Model` with optional Greeks (delta, vega, gamma)