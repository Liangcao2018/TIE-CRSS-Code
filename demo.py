"""
CRSS Demo -- Quick-start example
=================================

Demonstrates how to use the CRSS (Causal Robust Soft Sensor) model
on a synthetic industrial-process dataset with temporal causal structure.

Usage::

    python demo.py
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from crss import CRSS, CRSSWithConvergence


# ======================================================================
# 1. Generate a simple synthetic dataset
# ======================================================================
def generate_demo_data(n_samples=3000, n_sensors=20, noise_level=0.05,
                       seed=42):
    """Create a synthetic industrial-process dataset.

    The quality variable depends on *lagged* sensor projections (FIR
    structure), so methods that exploit temporal causality have an
    inherent advantage.

    Parameters
    ----------
    n_samples : int
    n_sensors : int
    noise_level : float
    seed : int

    Returns
    -------
    Y : ndarray (n_samples, n_sensors)  -- sensor measurements
    tau : ndarray (n_samples,)           -- quality variable
    """
    rng = np.random.RandomState(seed)
    t = np.arange(n_samples) / 100.0

    # Two latent process states with AR(1) dynamics
    state1 = np.zeros(n_samples)  # "temperature"
    state2 = np.zeros(n_samples)  # "pressure"
    state1[0], state2[0] = 50.0, 100.0
    for k in range(1, n_samples):
        state1[k] = (0.7 * state1[k - 1] + 0.3 * 50.0
                      + 3.5 * np.sin(0.5 * t[k]) + rng.randn() * 2.5)
        state2[k] = (0.72 * state2[k - 1] + 0.28 * 100.0
                      + 4.0 * np.cos(0.3 * t[k]) + rng.randn() * 3.0)

    # Sensor measurements (some causal, some spurious)
    Y = np.zeros((n_samples, n_sensors))
    Y[:, 0] = state1 + rng.randn(n_samples) * noise_level * 1.5
    Y[:, 1] = state1 * 0.9 + 5 + rng.randn(n_samples) * noise_level * 1.5
    Y[:, 2] = state2 + rng.randn(n_samples) * noise_level * 2.0
    Y[:, 3] = state2 * 0.95 + 3 + rng.randn(n_samples) * noise_level * 2.0
    Y[:, 4] = np.sqrt(np.abs(state2)) * 3 + rng.randn(n_samples) * noise_level
    Y[:, 5] = state1 * 0.8 + rng.randn(n_samples) * noise_level

    for i in range(6, 15):
        Y[:, i] = (state1 * (0.2 + 0.04 * i) + state2 * 0.05
                    + rng.randn(n_samples) * noise_level * (4 + i * 0.4))

    # Spurious sensors (no real causal link)
    for i in range(15, n_sensors):
        Y[:, i] = rng.randn(n_samples) * 12

    # Quality variable with FIR-type temporal causal structure
    tau = np.zeros(n_samples)
    for k in range(10, n_samples):
        tau[k] = (0.35 * (0.25 * state1[k - 4] + 0.50 * state1[k - 5]
                           + 0.25 * state1[k - 6])
                  + 0.004 * (0.20 * state2[k - 2] + 0.60 * state2[k - 3]
                              + 0.20 * state2[k - 4]))
    tau += rng.randn(n_samples) * np.std(tau[10:]) * noise_level * 0.15

    return Y, tau


# ======================================================================
# 2. Main demo
# ======================================================================
def main():
    print("=" * 60)
    print("CRSS -- Causal Robust Soft Sensor Demo")
    print("=" * 60)

    # --- Data ---
    Y, tau = generate_demo_data()
    scaler = StandardScaler()
    Y = scaler.fit_transform(Y)
    tau = (tau - tau.mean()) / (tau.std() + 1e-10)

    n_train = int(len(Y) * 0.7)
    Y_train, Y_test = Y[:n_train], Y[n_train:]
    tau_train, tau_test = tau[:n_train], tau[n_train:]

    print(f"\nDataset: {Y.shape[0]} samples, {Y.shape[1]} sensors")
    print(f"Train: {n_train}  |  Test: {len(Y) - n_train}")

    # --- Train CRSS ---
    print("\n--- Training CRSS ---")
    model = CRSS(
        n_features=8,
        filter_length=12,
        lambda_r=0.3,
        epsilon=0.2,
        max_iter=300,
        tol=1e-6,
        n_restarts=5,
        huber_delta=1.35,
    )
    model.fit(Y_train, tau_train)

    # --- Evaluate ---
    tau_pred = model.predict(Y_test)
    rmse = np.sqrt(mean_squared_error(tau_test, tau_pred))
    r2 = r2_score(tau_test, tau_pred)

    print(f"\nTest RMSE : {rmse:.4f}")
    print(f"Test R2   : {r2:.4f}")
    print(f"Robustness bound: {model.get_robustness_bound():.4f}")

    # --- Temporal pattern interpretation ---
    print("\n--- Causal Feature Interpretation ---")
    for j in range(model.n_features):
        dominant_lag = int(np.argmax(np.abs(model.Beta[j])))
        pattern = model.interpret_pattern(j)
        print(f"  Feature {j+1}: dominant lag = {dominant_lag}, "
              f"regression coeff = {model.b[j]:.4f}  ({pattern})")

    # --- Robustness test ---
    print("\n--- Robustness Under Additive Noise ---")
    for noise_pct in [0, 5, 10, 15, 20]:
        noise_level = noise_pct / 100.0
        rng = np.random.RandomState(42)
        Y_noisy = Y_test + rng.randn(*Y_test.shape) * np.std(Y_test, axis=0) * noise_level
        tau_noisy = model.predict(Y_noisy)
        rmse_n = np.sqrt(mean_squared_error(tau_test, tau_noisy))
        print(f"  Noise {noise_pct:>2d}%: RMSE = {rmse_n:.4f}")

    # --- Convergence tracking ---
    print("\n--- Convergence Tracking ---")
    conv_model = CRSSWithConvergence(
        n_features=8, filter_length=12, lambda_r=0.3, epsilon=0.2,
        max_iter=300, tol=1e-6, n_restarts=5, huber_delta=1.35,
    )
    conv_model.fit(Y_train, tau_train)
    for j, hist in enumerate(conv_model.feature_convergence):
        iters = len(hist)
        final_change = hist[-1]['w_change'] if hist else float('nan')
        print(f"  Feature {j+1}: converged in {iters} iterations "
              f"(final |dw| = {final_change:.2e})")

    print("\nDone.")


if __name__ == '__main__':
    main()
