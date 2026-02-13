# CRSS: Causal Robust Soft Sensor

Official implementation of the CRSS framework proposed in:

> **A Novel Causal Robust Soft Sensor for Industrial Processes with Data Distribution Drift**
>
> Liang Cao, Yan Qin, Dong Zhao, Youqing Wang
>
> *IEEE Transactions on Industrial Electronics*, 2026.

## Overview

CRSS is a spatio-temporal soft sensor designed for industrial processes where the data distribution may shift over time (e.g., regime changes, sensor degradation, process drift). It extracts **causal features** by jointly optimising spatial projections and temporal FIR filters, with built-in robustness via **Wasserstein Distributionally Robust Optimisation (DRO)** and **Huber-weighted covariance estimation**.

### Key Features

- **Spatio-temporal causal feature extraction** -- iterative deflation with alternating maximisation
- **Wasserstein DRO regularisation** -- mixed L1/L2 penalty on temporal filters for distributional robustness
- **Huber-weighted covariance** -- outlier-robust estimation of cross-covariance matrices
- **PLS-guided initialisation + multi-restart** -- escapes local optima efficiently
- **Theoretical guarantees** -- prediction error decomposition (Theorem 1), robustness bounds (Theorem 2), and convergence analysis (Theorem 3)

## Installation

```bash
git clone https://github.com/<your-username>/CRSS.git
cd CRSS
pip install -r requirements.txt
```

## Quick Start

```python
import numpy as np
from crss import CRSS

# Load your data: Y (N x m sensor matrix), tau (N, quality variable)
# Y, tau = ...

model = CRSS(
    n_features=8,        # Number of causal features to extract
    filter_length=12,    # Temporal FIR filter length (lags)
    lambda_r=0.3,        # DRO regularisation strength
    epsilon=0.2,         # Wasserstein ball radius
    max_iter=300,        # Max alternating-optimisation iterations
    n_restarts=5,        # Multi-restart for global search
    huber_delta=1.35,    # Huber robustness threshold
)

model.fit(Y_train, tau_train)
tau_pred = model.predict(Y_test)

# Theoretical robustness bound
bound = model.get_robustness_bound()

# Interpret temporal dynamics of each feature
for j in range(model.n_features):
    print(f"Feature {j+1}: {model.interpret_pattern(j)}")
```

## Running the Demo

```bash
python demo.py
```

This generates a synthetic industrial-process dataset with FIR-type temporal causal structure and demonstrates training, prediction, robustness testing, and convergence tracking.

## Project Structure

```
CRSS/
├── crss/
│   ├── __init__.py          # Package exports
│   ├── model.py             # CRSS & CRSSWithConvergence
│   ├── baselines.py         # 28 comparison methods (see paper Table III)
│   └── evaluation.py        # Evaluation, robustness, sensitivity, statistical tests
├── demo.py                  # Quick-start example
├── README.md
├── requirements.txt
└── LICENSE
```

### Module Details

| File | Description |
|------|-------------|
| `crss/model.py` | Core CRSS algorithm: alternating maximisation, Wasserstein DRO, Huber weights, iterative deflation. Also includes `CRSSWithConvergence` for tracking optimisation trajectories. |
| `crss/baselines.py` | 28 baseline models including OLS, Ridge, Lasso, PLS, SVR, GPR, Random Forest, XGBoost, MLP, LSTM, TCN, Transformer, IRM, CDA, and more. Factory functions `create_all_models()` and `create_ablation_models()`. |
| `crss/evaluation.py` | `evaluate_model()` with bootstrap CIs, extended robustness testing (drift, dropout, outage, saturation, dead-band), ablation studies, hyperparameter sensitivity sweeps, scalability benchmarks, orthogonality verification, conditional Granger causality, ADF stationarity test, and MMD distribution test. |

## Algorithm

CRSS extracts $\ell$ causal features $\{\varphi_j\}_{j=1}^{\ell}$ via iterative deflation. Each feature is computed as:

$$\varphi_j[k] = \sum_{i=0}^{s-1} \beta_j[i] \cdot \mathbf{Y}[k-i]^\top \mathbf{w}_j$$

where $\mathbf{w}_j \in \mathbb{R}^m$ is the spatial projection and $\boldsymbol{\beta}_j \in \mathbb{R}^s$ is the temporal FIR filter. The parameters are optimised by alternating maximisation of a Huber-weighted cross-covariance objective subject to Wasserstein DRO constraints.

The final prediction is:

$$\hat{\tau}[k] = \sum_{j=1}^{\ell} b_j \cdot \varphi_j[k]$$

## Hyperparameters

| Parameter | Symbol | Recommended | Description |
|-----------|--------|-------------|-------------|
| `n_features` | $\ell$ | 5--10 | Number of causal features |
| `filter_length` | $s$ | 10--15 | FIR filter length (max lag) |
| `lambda_r` | $\lambda_r$ | 0.1--0.5 | DRO regularisation strength |
| `epsilon` | $\varepsilon$ | 0.1--0.3 | Wasserstein ball radius |
| `n_restarts` | -- | 3--5 | Multi-restart count |
| `huber_delta` | $\delta$ | 1.35 | Huber threshold |

## Citation

```bibtex
@article{cao2026crss,
  author  = {Cao, Liang and Qin, Yan and Zhao, Dong and Wang, Youqing},
  title   = {A Novel Causal Robust Soft Sensor for Industrial Processes with Data Distribution Drift},
  journal = {IEEE Transactions on Industrial Electronics},
  year    = {2026},
}
```

## License

This project is licensed under the MIT License -- see the [LICENSE](LICENSE) file for details.
