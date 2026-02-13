"""
Evaluation utilities for the CRSS framework.

Provides model evaluation with bootstrap confidence intervals,
robustness testing under various perturbation scenarios, ablation
studies, hyperparameter sensitivity analysis, scalability benchmarks,
and statistical validation tests (Granger causality, ADF, MMD).
"""

import time
import numpy as np
from scipy import signal, stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# ======================================================================
# Core evaluation
# ======================================================================

def evaluate_model(model, model_name, X_train, y_train, X_test, y_test,
                   X_regime, y_regime, noise_levels, n_bootstrap=20):
    """Evaluate a single model with bootstrap CIs and robustness testing.

    Parameters
    ----------
    model : estimator or ``'custom'``
        A fitted/fittable model with ``fit`` and ``predict`` methods.
        Pass ``'custom'`` for the PCA+MLR pipeline (handled internally).
    model_name : str
        Human-readable model name (used for special-case handling).
    X_train, y_train : ndarray
        Training data.
    X_test, y_test : ndarray
        Test data.
    X_regime : ndarray
        Sensor data from a different operating regime.
    y_regime : ndarray
        Quality variable for the regime-change data.
    noise_levels : array-like of float
        Sequence of noise standard-deviation multipliers (e.g. 0, 0.05, ...).
    n_bootstrap : int, default=20
        Number of bootstrap resamples for CI estimation.

    Returns
    -------
    results : dict
        Keys include ``rmse_test``, ``r2_test``, ``mape_test``,
        ``rmse_regime``, ``robustness`` (list of RMSE at each noise level),
        ``train_time``, ``pred_time``, ``predictions``, and bootstrap CIs.
    """
    results = {
        'name': model_name,
        'train_time': 0.0,
        'pred_time': 0.0,
        'rmse_test': 0.0,
        'rmse_ci': [0.0, 0.0],
        'r2_test': 0.0,
        'r2_ci': [0.0, 0.0],
        'mape_test': 0.0,
        'rmse_regime': 0.0,
        'robustness': [],
        'predictions': None,
    }

    # --- Special handling for PCA+MLR ---
    if model_name == 'PCA+MLR':
        pca = PCA(n_components=5)
        lr = LinearRegression()
        t0 = time.time()
        X_train_pca = pca.fit_transform(X_train)
        lr.fit(X_train_pca, y_train)
        results['train_time'] = time.time() - t0

        t0 = time.time()
        y_pred = lr.predict(pca.transform(X_test))
        results['pred_time'] = time.time() - t0

        y_pred_regime = lr.predict(pca.transform(X_regime))
        model_combo = (pca, lr)
    else:
        t0 = time.time()
        model.fit(X_train, y_train)
        results['train_time'] = time.time() - t0

        t0 = time.time()
        y_pred = np.atleast_1d(model.predict(X_test)).flatten()
        results['pred_time'] = time.time() - t0

        y_pred_regime = np.atleast_1d(model.predict(X_regime)).flatten()
        model_combo = None

    results['predictions'] = y_pred

    # --- Handle NaN from time-series models ---
    ts_models = {'LSTM', 'Causal NN', 'Elastic Net TS', 'Transformer'}
    if model_name in ts_models:
        valid = ~np.isnan(y_pred)
        y_pred_clean = y_pred[valid]
        y_test_clean = y_test[valid]
    else:
        y_pred_clean = y_pred
        y_test_clean = y_test

    # --- Bootstrap confidence intervals ---
    rmse_boots, r2_boots = [], []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(y_test_clean), len(y_test_clean), replace=True)
        rmse_boots.append(np.sqrt(mean_squared_error(y_test_clean[idx], y_pred_clean[idx])))
        r2_boots.append(r2_score(y_test_clean[idx], y_pred_clean[idx]))

    results['rmse_test'] = float(np.sqrt(mean_squared_error(y_test_clean, y_pred_clean)))
    results['rmse_ci'] = [float(np.percentile(rmse_boots, 2.5)),
                          float(np.percentile(rmse_boots, 97.5))]
    results['r2_test'] = float(r2_score(y_test_clean, y_pred_clean))
    results['r2_ci'] = [float(np.percentile(r2_boots, 2.5)),
                        float(np.percentile(r2_boots, 97.5))]

    # --- MAPE ---
    mask = np.abs(y_test_clean) > 0.1
    if np.sum(mask) > 0:
        results['mape_test'] = float(
            np.mean(np.abs((y_test_clean[mask] - y_pred_clean[mask]) / y_test_clean[mask])) * 100
        )
    else:
        results['mape_test'] = float('nan')

    # --- Regime-change performance ---
    valid_r = ~np.isnan(y_pred_regime)
    results['rmse_regime'] = float(np.sqrt(
        mean_squared_error(y_regime[valid_r], y_pred_regime[valid_r])
    ))

    # --- Robustness under additive noise ---
    base_std = np.std(X_test, axis=0)
    for noise_idx, noise_level in enumerate(noise_levels):
        rng = np.random.RandomState(42 + noise_idx * 1000)
        noise = rng.randn(*X_test.shape)
        for si in range(X_test.shape[1]):
            amp = 1.0 + 0.5 * np.sin(si)
            if noise_idx > 0:
                noise[:, si] = signal.lfilter([1], [1, -0.3], noise[:, si])
            noise[:, si] *= amp
        X_noisy = X_test + noise * base_std[None, :] * noise_level

        if model_name == 'PCA+MLR' and model_combo is not None:
            y_noisy = model_combo[1].predict(model_combo[0].transform(X_noisy))
        else:
            y_noisy = np.atleast_1d(model.predict(X_noisy)).flatten()

        if model_name in ts_models:
            v = ~np.isnan(y_noisy)
            y_noisy, y_t = y_noisy[v], y_test[v]
        else:
            y_t = y_test

        results['robustness'].append(float(np.sqrt(mean_squared_error(y_t, y_noisy))))

    return results


# ======================================================================
# Extended robustness (realistic perturbation scenarios)
# ======================================================================

def run_extended_robustness(model, model_name, X_train, y_train, X_test, y_test):
    """Test model resilience under realistic perturbation scenarios.

    Scenarios tested:
      1. Abrupt sensor drift (step changes)
      2. Intermittent sensor dropout
      3. Heavy-tailed noise (Student-t, df=3)
      4. Complete sensor outages
      5. Saturation clipping
      6. Dead-band quantisation
      7. Combined "bad day" scenario
      8. Graceful degradation curve (0-50 % Gaussian noise)

    Parameters
    ----------
    model : estimator
    model_name : str
    X_train, y_train, X_test, y_test : ndarray

    Returns
    -------
    results : dict
    """
    results = {}

    # Train & build predict function
    if model_name == 'PCA+MLR':
        pca = PCA(n_components=5)
        lr = LinearRegression()
        lr.fit(pca.fit_transform(X_train), y_train)
        predict_fn = lambda X: lr.predict(pca.transform(X))
    elif model_name == 'Standard PLS':
        model.fit(X_train, y_train)
        predict_fn = lambda X: np.atleast_1d(model.predict(X)).flatten()
    else:
        model.fit(X_train, y_train)
        predict_fn = lambda X: np.atleast_1d(model.predict(X)).flatten()

    def _rmse_safe(y_true, y_pred):
        v = ~np.isnan(y_pred)
        return float(np.sqrt(mean_squared_error(y_true[v], y_pred[v])))

    y_base = predict_fn(X_test)
    results['rmse_baseline'] = _rmse_safe(y_test, y_base)

    # 1. Abrupt drift
    drift_res = {}
    for pct in [0.10, 0.20]:
        Xd = X_test.copy(); Xd[:, :10] *= (1.0 + pct)
        drift_res[f'{int(pct*100)}%'] = _rmse_safe(y_test, predict_fn(Xd))
    results['abrupt_drift'] = drift_res

    # 2. Dropout
    drop_res = {}
    for dp in [0.05, 0.10]:
        rng = np.random.RandomState(42)
        Xdr = X_test.copy(); Xdr[rng.random(Xdr.shape) < dp] = 0.0
        drop_res[f'{int(dp*100)}%'] = _rmse_safe(y_test, predict_fn(Xdr))
    results['dropout'] = drop_res

    # 3. Heavy-tailed noise
    rng = np.random.RandomState(42)
    Xh = X_test + stats.t.rvs(df=3, size=X_test.shape, random_state=rng) * np.std(X_test, axis=0) * 0.1
    results['heavy_tail_noise'] = _rmse_safe(y_test, predict_fn(Xh))

    # 4. Sensor outage
    Xo = X_test.copy(); Xo[:, [0, 2, 5]] = 0.0
    results['sensor_outage'] = _rmse_safe(y_test, predict_fn(Xo))

    # 5. Saturation
    Xs = X_test.copy()
    for j in range(Xs.shape[1]):
        Xs[:, j] = np.clip(Xs[:, j],
                           np.percentile(X_train[:, j], 5),
                           np.percentile(X_train[:, j], 95))
    results['saturation'] = _rmse_safe(y_test, predict_fn(Xs))

    # 6. Dead-band
    Xdb = X_test.copy()
    for j in range(Xdb.shape[1]):
        th = np.std(X_train[:, j]) * 0.1
        diff = np.diff(Xdb[:, j], prepend=Xdb[0, j])
        Xdb[np.abs(diff) < th, j] = np.mean(X_train[:, j])
    results['dead_band'] = _rmse_safe(y_test, predict_fn(Xdb))

    # 7. Combined
    rng = np.random.RandomState(42)
    Xb = X_test.copy()
    Xb[:, :5] *= 1.15
    Xb[rng.random(Xb.shape) < 0.03] = 0.0
    Xb += stats.t.rvs(df=5, size=Xb.shape, random_state=rng) * np.std(X_test, axis=0) * 0.05
    results['combined'] = _rmse_safe(y_test, predict_fn(Xb))

    # 8. Degradation curve
    levels = np.arange(0, 0.55, 0.05)
    curve = []
    for nl in levels:
        rng = np.random.RandomState(42)
        Xn = X_test + rng.randn(*X_test.shape) * np.std(X_test, axis=0) * nl
        curve.append(_rmse_safe(y_test, predict_fn(Xn)))
    results['degradation_curve'] = curve
    results['degradation_levels'] = levels

    return results


# ======================================================================
# Ablation study
# ======================================================================

def run_ablation_study(ablation_models, X_train, y_train, X_test, y_test,
                       X_regime, y_regime, noise_levels):
    """Run ablation study across model variants.

    Parameters
    ----------
    ablation_models : dict[str, estimator]
        Output of ``baselines.create_ablation_models()``.

    Returns
    -------
    ablation_results : dict[str, dict]
    """
    out = {}
    for name, model in ablation_models.items():
        try:
            res = evaluate_model(model, name, X_train, y_train, X_test, y_test,
                                 X_regime, y_regime, noise_levels, n_bootstrap=200)
            out[name] = res
        except Exception as e:
            print(f"  {name:<22s}  ERROR: {e}")
    return out


# ======================================================================
# Hyperparameter sensitivity
# ======================================================================

def run_hyperparameter_sensitivity(X_train, y_train, X_test, y_test):
    """Sweep four key CRSS hyperparameters and record RMSE / R2.

    Returns
    -------
    results : dict  with keys ``lambda_r``, ``n_features``,
              ``filter_length``, ``epsilon``, each containing
              ``values``, ``rmse``, ``r2``.
    """
    from .model import CRSS

    results = {}

    sweeps = {
        'lambda_r':      {'param': 'lambda_r',
                          'values': [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]},
        'n_features':    {'param': 'n_features',
                          'values': [1, 2, 3, 4, 5, 6, 7, 8, 10, 12]},
        'filter_length': {'param': 'filter_length',
                          'values': [1, 3, 5, 8, 10, 12, 15, 20]},
        'epsilon':       {'param': 'epsilon',
                          'values': [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]},
    }

    defaults = dict(n_features=8, filter_length=12, lambda_r=0.3,
                    epsilon=0.2, max_iter=300, tol=1e-6, n_restarts=3,
                    huber_delta=1.35)

    for key, cfg in sweeps.items():
        rmses, r2s = [], []
        for val in cfg['values']:
            kw = {**defaults, cfg['param']: val}
            m = CRSS(**kw)
            m.fit(X_train, y_train)
            yp = m.predict(X_test)
            rmses.append(float(np.sqrt(mean_squared_error(y_test, yp))))
            r2s.append(float(r2_score(y_test, yp)))
        results[key] = {'values': cfg['values'], 'rmse': rmses, 'r2': r2s}

    return results


# ======================================================================
# Scalability benchmarks
# ======================================================================

def run_scalability_experiments(Y_full, tau_full):
    """Measure CRSS training time vs. number of sensors and sample size.

    Returns
    -------
    results : dict  with keys ``sensors`` and ``samples``.
    """
    from .model import CRSS

    results = {}

    # Sensors sweep
    sensor_counts = [10, 20, 50, 100, 200]
    sensor_times = []
    for ns in sensor_counts:
        if ns <= Y_full.shape[1]:
            X = Y_full[:3000, :ns]
        else:
            reps = ns // Y_full.shape[1] + 1
            X = np.tile(Y_full[:3000], (1, reps))[:, :ns]
            X += np.random.randn(*X.shape) * 0.01
        t0 = time.time()
        CRSS(n_features=5, filter_length=10, lambda_r=0.3, epsilon=0.2,
             max_iter=100, tol=1e-5, n_restarts=2).fit(X, tau_full[:3000])
        sensor_times.append(time.time() - t0)
    results['sensors'] = {'counts': sensor_counts, 'times': sensor_times}

    # Samples sweep
    sample_sizes = [1000, 2000, 5000, 10000, 20000]
    sample_times = []
    for ns in sample_sizes:
        if ns <= len(Y_full):
            X, y = Y_full[:ns], tau_full[:ns]
        else:
            reps = ns // len(Y_full) + 1
            X = np.tile(Y_full, (reps, 1))[:ns]
            y = np.tile(tau_full, reps)[:ns]
            X += np.random.randn(*X.shape) * 0.01
        t0 = time.time()
        CRSS(n_features=5, filter_length=10, lambda_r=0.3, epsilon=0.2,
             max_iter=100, tol=1e-5, n_restarts=2).fit(X, y)
        sample_times.append(time.time() - t0)
    results['samples'] = {'sizes': sample_sizes, 'times': sample_times}

    return results


# ======================================================================
# Orthogonality verification
# ======================================================================

def run_orthogonality_verification(crss_model, Y_train, tau_train):
    """Verify near-orthogonality of CRSS features and compare
    stagewise vs. joint OLS refitting.

    Returns
    -------
    results : dict
    """
    n_f = len(crss_model.W)
    features = np.zeros((len(Y_train), n_f))
    Y_def = Y_train.copy()

    for j in range(n_f):
        features[:, j] = crss_model._compute_feature(Y_def, crss_model.W[j], crss_model.Beta[j])
        phi = features[:, j]
        p = Y_def.T @ phi / (phi @ phi + 1e-10)
        Y_def = Y_def - np.outer(phi, p)

    corr = np.corrcoef(features.T)
    max_pw = max(abs(corr[i, j])
                 for i in range(n_f) for j in range(i + 1, n_f))

    rmse_stage = float(np.sqrt(mean_squared_error(tau_train, crss_model.predict(Y_train))))

    lr = LinearRegression().fit(features, tau_train)
    rmse_ols = float(np.sqrt(mean_squared_error(tau_train, lr.predict(features))))

    return {
        'correlation_matrix': corr,
        'max_pairwise_corr': max_pw,
        'rmse_stagewise': rmse_stage,
        'rmse_joint_ols': rmse_ols,
        'ols_coefficients': lr.coef_,
    }


# ======================================================================
# Conditional Granger Causality
# ======================================================================

def run_conditional_granger_causality(Y_train, tau_train, max_lag=12):
    """Conditional Granger causality F-test, transfer entropy, and
    Spearman correlation with CRSS sensor importance.

    Returns
    -------
    results : dict
    """
    from .model import CRSS

    n_sensors = Y_train.shape[1]
    f_stats = np.zeros(n_sensors)
    p_vals = np.ones(n_sensors)

    # Restricted model (AR on tau only)
    tau_lag = np.column_stack(
        [tau_train[max_lag - i - 1 : -i - 1] if i > 0 else tau_train[max_lag:]
         for i in range(max_lag)]
    )
    tau_tgt = tau_train[max_lag:]
    n_eff = len(tau_tgt)

    lr_r = LinearRegression().fit(tau_lag, tau_tgt)
    rss_r = float(np.sum((tau_tgt - lr_r.predict(tau_lag)) ** 2))

    for s in range(n_sensors):
        s_lag = np.column_stack(
            [Y_train[max_lag - i - 1 : -i - 1, s] if i > 0 else Y_train[max_lag:, s]
             for i in range(max_lag)]
        )
        Xu = np.hstack([tau_lag, s_lag])
        lr_u = LinearRegression().fit(Xu, tau_tgt)
        rss_u = float(np.sum((tau_tgt - lr_u.predict(Xu)) ** 2))
        df1, df2 = max_lag, n_eff - 2 * max_lag - 1
        if df2 > 0 and rss_u > 0:
            f_stats[s] = ((rss_r - rss_u) / df1) / (rss_u / df2)
            p_vals[s] = 1.0 - stats.f.cdf(f_stats[s], df1, df2)

    # Transfer entropy approximation
    te = np.zeros(n_sensors)
    for s in range(n_sensors):
        for lag in range(1, min(max_lag, 8)):
            if lag < len(tau_train):
                sl = Y_train[:-lag, s]
                tf = tau_train[lag:]
                n_te = min(len(sl), len(tf))
                if np.std(sl[:n_te]) > 1e-10:
                    c = np.corrcoef(sl[:n_te], tf[:n_te])[0, 1]
                    if abs(c) < 1:
                        te[s] += -0.5 * np.log(1 - c ** 2)

    # CRSS importance for Spearman correlation
    crss = CRSS(n_features=8, filter_length=12, lambda_r=0.3, epsilon=0.2,
                max_iter=100, tol=1e-5, n_restarts=2).fit(Y_train, tau_train)
    imp = np.zeros(n_sensors)
    for j in range(len(crss.W)):
        imp += np.abs(crss.W[j]) * abs(crss.b[j])

    rho, rho_p = stats.spearmanr(te, imp)

    return {
        'f_statistics': f_stats,
        'p_values': p_vals,
        'significant_sensors': np.where(p_vals < 0.05)[0],
        'removed_sensors': np.where(p_vals >= 0.05)[0],
        'transfer_entropy': te,
        'sensor_importance': imp,
        'spearman_corr': float(rho),
        'spearman_p': float(rho_p),
    }


# ======================================================================
# ADF Stationarity test
# ======================================================================

def run_adf_stationarity_test(Y_train, tau_train):
    """Augmented Dickey-Fuller stationarity test (statsmodels-free).

    Returns
    -------
    results : dict
    """

    def _adf(series, p=12):
        dy = np.diff(series)
        y_lag = series[p:-1]
        dy_tgt = dy[p:]
        X = np.column_stack(
            [y_lag] +
            [dy[p - i - 1 : -i - 1] if i > 0 else dy[p:] for i in range(p)]
        )
        X = np.column_stack([np.ones(len(dy_tgt)), X])
        try:
            beta = np.linalg.lstsq(X, dy_tgt, rcond=None)[0]
            res = dy_tgt - X @ beta
            s2 = np.sum(res ** 2) / (len(dy_tgt) - X.shape[1])
            cov = s2 * np.linalg.inv(X.T @ X)
            t = beta[1] / np.sqrt(cov[1, 1])
            return {'t_stat': float(t), 'is_stationary': t < -2.86}
        except Exception:
            return {'t_stat': 0.0, 'is_stationary': True}

    sensor_res = [_adf(Y_train[:, s]) for s in range(Y_train.shape[1])]
    return {
        'quality_variable': _adf(tau_train),
        'sensors': sensor_res,
        'n_stationary': sum(r['is_stationary'] for r in sensor_res),
        'n_nonstationary': sum(not r['is_stationary'] for r in sensor_res),
    }


# ======================================================================
# MMD test
# ======================================================================

def run_mmd_test(X_train, X_test, n_perm=200):
    """Maximum Mean Discrepancy between train and test distributions
    with per-sensor breakdown and permutation p-value.

    Returns
    -------
    results : dict
    """

    def _gauss_kernel(X1, X2, sigma):
        d = np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=2)
        return np.exp(-d / (2 * sigma ** 2))

    n_sub = min(500, len(X_train), len(X_test))
    rng = np.random.RandomState(42)
    Xtr = X_train[rng.choice(len(X_train), n_sub, replace=False)]
    Xte = X_test[rng.choice(len(X_test), n_sub, replace=False)]

    # Median heuristic bandwidth
    combined = np.vstack([Xtr[:100], Xte[:100]])
    pw = np.sqrt(np.sum((combined[:, None, :] - combined[None, :, :]) ** 2, axis=2))
    sigma = float(np.median(pw[np.triu_indices(len(combined), k=1)]))
    if sigma < 1e-10:
        sigma = 1.0

    Ktt = _gauss_kernel(Xtr, Xtr, sigma)
    Kss = _gauss_kernel(Xte, Xte, sigma)
    Kts = _gauss_kernel(Xtr, Xte, sigma)
    mmd = float(np.sqrt(max(np.mean(Ktt) - 2 * np.mean(Kts) + np.mean(Kss), 0)))

    # Per-sensor MMD
    pf_mmd = []
    for j in range(X_train.shape[1]):
        x1, x2 = Xtr[:, j:j+1], Xte[:, j:j+1]
        K1 = _gauss_kernel(x1, x1, sigma)
        K2 = _gauss_kernel(x2, x2, sigma)
        K12 = _gauss_kernel(x1, x2, sigma)
        pf_mmd.append(float(np.sqrt(max(np.mean(K1) - 2*np.mean(K12) + np.mean(K2), 0))))

    # Permutation test
    comb = np.vstack([Xtr, Xte])
    perm_mmds = []
    for _ in range(n_perm):
        perm = rng.permutation(len(comb))
        A, B = comb[perm[:n_sub]], comb[perm[n_sub:2*n_sub]]
        Ka = _gauss_kernel(A, A, sigma)
        Kb = _gauss_kernel(B, B, sigma)
        Kab = _gauss_kernel(A, B, sigma)
        perm_mmds.append(float(np.sqrt(max(np.mean(Ka) - 2*np.mean(Kab) + np.mean(Kb), 0))))

    return {
        'mmd': mmd,
        'sigma': sigma,
        'per_feature_mmd': pf_mmd,
        'p_value': float(np.mean(np.array(perm_mmds) >= mmd)),
        'perm_mmds': perm_mmds,
    }
