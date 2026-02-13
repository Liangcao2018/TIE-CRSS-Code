"""
Baseline comparison models used in the CRSS paper.

Includes wrappers that give time-series-aware models a unified
``fit(X, y)`` / ``predict(X)`` interface compatible with scikit-learn.
"""

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import (
    BayesianRidge, ElasticNet, HuberRegressor, LinearRegression,
    RANSACRegressor, TheilSenRegressor, ARDRegression, Ridge, Lasso,
)
from sklearn.ensemble import (
    AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR


# ======================================================================
# Time-series-aware wrappers
# ======================================================================

class ElasticNetTS:
    """Elastic Net with automatic lagged-feature construction."""

    def __init__(self, alpha=1.0, l1_ratio=0.5, max_lag=10):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_lag = max_lag
        self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=2000)

    def _create_lag_features(self, X):
        n_samples, n_features = X.shape
        n_seq = n_samples - self.max_lag + 1
        X_lagged = np.zeros((n_seq, self.max_lag * n_features))
        for i in range(n_seq):
            X_lagged[i] = X[i : i + self.max_lag].flatten()
        return X_lagged

    def fit(self, X, y):
        X_lagged = self._create_lag_features(X)
        self.model.fit(X_lagged, y[self.max_lag - 1 :])

    def predict(self, X):
        X_lagged = self._create_lag_features(X)
        preds = np.full(len(X), np.nan)
        preds[self.max_lag - 1 :] = self.model.predict(X_lagged)
        return preds


class BayesianRidgeTS:
    """Bayesian Ridge Regression with sliding-window features."""

    def __init__(self, window_size=10):
        self.window_size = window_size
        self.model = BayesianRidge(
            max_iter=300, tol=1e-3,
            alpha_1=1e-6, alpha_2=1e-6,
            lambda_1=1e-6, lambda_2=1e-6,
        )

    def _create_windows(self, X, y=None):
        n = len(X) - self.window_size + 1
        X_w = np.zeros((n, self.window_size * X.shape[1]))
        for i in range(n):
            X_w[i] = X[i : i + self.window_size].flatten()
        if y is not None:
            return X_w, y[self.window_size - 1 :]
        return X_w

    def fit(self, X, y):
        Xw, yw = self._create_windows(X, y)
        self.model.fit(Xw, yw)

    def predict(self, X):
        Xw = self._create_windows(X)
        preds = np.full(len(X), np.nan)
        preds[self.window_size - 1 :] = self.model.predict(Xw)
        return preds


class LSTM:
    """LSTM-style model approximated via MLPRegressor with windowed features."""

    def __init__(self, window_size=10, hidden_size=64):
        self.window_size = window_size
        self.model = MLPRegressor(
            hidden_layer_sizes=(hidden_size, hidden_size // 2),
            activation='relu', solver='adam', max_iter=500,
            early_stopping=True, validation_fraction=0.1, random_state=42,
        )

    def _create_sequences(self, X, y=None):
        n = len(X) - self.window_size + 1
        X_seq = np.zeros((n, self.window_size * X.shape[1]))
        for i in range(n):
            X_seq[i] = X[i : i + self.window_size].flatten()
        if y is not None:
            return X_seq, y[self.window_size - 1 :]
        return X_seq

    def fit(self, X, y):
        Xs, ys = self._create_sequences(X, y)
        self.model.fit(Xs, ys)

    def predict(self, X):
        Xs = self._create_sequences(X)
        preds = np.full(len(X), np.nan)
        preds[self.window_size - 1 :] = self.model.predict(Xs)
        return preds


class TCN:
    """Temporal Convolutional Network approximation (dilated convolutions + MLP)."""

    def __init__(self, n_filters=64, kernel_size=3, n_layers=3):
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.filters = []
        self.model = None

    def _create_dilated_conv_features(self, X):
        n_samples, n_features = X.shape
        all_features = []
        for layer in range(self.n_layers):
            dilation = 2 ** layer
            for f in range(min(self.n_filters // self.n_layers, 20)):
                conv_out = np.zeros(n_samples)
                idx = layer * 20 + f
                if idx >= len(self.filters):
                    self.filters.append(
                        np.random.randn(self.kernel_size, n_features) * 0.1
                    )
                fw = self.filters[idx]
                for t in range(n_samples):
                    s, c = 0.0, 0
                    for k in range(self.kernel_size):
                        ti = t - k * dilation
                        if 0 <= ti < n_samples:
                            s += np.sum(X[ti] * fw[k])
                            c += 1
                    if c > 0:
                        conv_out[t] = s / c
                all_features.append(conv_out)
        return np.column_stack(all_features) if all_features else np.zeros((n_samples, 1))

    def fit(self, X, y):
        feats = self._create_dilated_conv_features(X)
        self.model = MLPRegressor(
            hidden_layer_sizes=(100, 50), activation='relu',
            max_iter=300, early_stopping=True, random_state=42,
        )
        self.model.fit(feats, y)

    def predict(self, X):
        feats = self._create_dilated_conv_features(X)
        return self.model.predict(feats)


class CausalNN:
    """Neural Network with causal (lagged) feature construction."""

    def __init__(self, lag=10, hidden_size=64):
        self.lag = lag
        self.model = MLPRegressor(
            hidden_layer_sizes=(hidden_size, hidden_size // 2),
            activation='tanh', solver='lbfgs', max_iter=500, random_state=42,
        )

    def _create_causal_features(self, X):
        n_samples, n_features = X.shape
        n_seq = n_samples - self.lag + 1
        Xc = np.zeros((n_seq, self.lag * n_features))
        for i in range(n_seq):
            Xc[i] = X[i : i + self.lag].flatten()
        return Xc

    def fit(self, X, y):
        Xc = self._create_causal_features(X)
        self.model.fit(Xc, y[self.lag - 1 :])

    def predict(self, X):
        Xc = self._create_causal_features(X)
        preds = np.full(len(X), np.nan)
        preds[self.lag - 1 :] = self.model.predict(Xc)
        return preds


class RobustPLS:
    """Robust PLS via iterative reweighting."""

    def __init__(self, n_components=5, max_iter=10, threshold=2.5):
        self.n_components = n_components
        self.max_iter = max_iter
        self.threshold = threshold
        self.pls = None

    def fit(self, X, y):
        n = len(X)
        weights = np.ones(n)
        for _ in range(self.max_iter):
            self.pls = PLSRegression(n_components=self.n_components)
            self.pls.fit(X * weights[:, None], y * weights)
            residuals = np.abs(y - self.pls.predict(X).flatten())
            mad = np.median(np.abs(residuals - np.median(residuals)))
            thresh = self.threshold * mad
            weights = np.where(residuals <= thresh, 1.0, thresh / (residuals + 1e-10))
            weights = weights / np.sum(weights) * n

    def predict(self, X):
        return self.pls.predict(X).flatten()


# ======================================================================
# Causal / Domain-adaptation baselines
# ======================================================================

class IRM:
    """Invariant Risk Minimization approximation (Arjovsky et al., 2019).

    Splits data into pseudo-environments (time segments) and uses
    stronger regularisation to approximate the IRM penalty.
    """

    def __init__(self, n_envs=3, hidden_size=64, penalty_weight=1e3, max_iter=500):
        self.n_envs = n_envs
        self.hidden_size = hidden_size
        self.penalty_weight = penalty_weight
        self.max_iter = max_iter
        self.model = None

    def fit(self, X, y):
        alpha_irm = 0.01 * (1 + self.penalty_weight / 1e4)
        self.model = MLPRegressor(
            hidden_layer_sizes=(self.hidden_size, self.hidden_size // 2),
            activation='relu', solver='adam', max_iter=self.max_iter,
            early_stopping=True, validation_fraction=0.1,
            random_state=42, alpha=alpha_irm,
        )
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class CDA:
    """Causal Domain Adaptation via stable-feature selection + Huber regression."""

    def __init__(self, n_envs=3, stability_threshold=0.3):
        self.n_envs = n_envs
        self.stability_threshold = stability_threshold
        self.selected_features = None
        self.model = None

    def fit(self, X, y):
        n, m = X.shape
        env_size = n // self.n_envs
        env_corrs = np.zeros((self.n_envs, m))
        for e in range(self.n_envs):
            s, t = e * env_size, min((e + 1) * env_size, n)
            for j in range(m):
                if np.std(X[s:t, j]) > 1e-10:
                    env_corrs[e, j] = np.corrcoef(X[s:t, j], y[s:t])[0, 1]
        stability = np.mean(np.abs(env_corrs), axis=0) / (np.std(env_corrs, axis=0) + 0.1)
        self.selected_features = np.where(stability > self.stability_threshold)[0]
        if len(self.selected_features) < 3:
            self.selected_features = np.argsort(stability)[-max(5, m // 4) :]
        self.model = HuberRegressor(epsilon=1.35, max_iter=300)
        self.model.fit(X[:, self.selected_features], y)

    def predict(self, X):
        return self.model.predict(X[:, self.selected_features])


class TransformerRegressor:
    """Transformer-style regressor (positional-encoded lag features + MLP)."""

    def __init__(self, window_size=12, d_model=64, n_heads=4):
        self.window_size = window_size
        self.model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32), activation='relu',
            solver='adam', max_iter=800, early_stopping=True,
            validation_fraction=0.1, random_state=42, alpha=0.001,
        )

    def _positional_features(self, X):
        n_samples, nf = X.shape
        n_seq = n_samples - self.window_size + 1
        pos = np.arange(self.window_size).reshape(-1, 1)
        div = np.exp(np.arange(0, min(nf, 8), 2) * -(np.log(10000.0) / min(nf, 8)))
        pe = np.concatenate([np.sin(pos * div), np.cos(pos * div)], axis=1)
        out = np.zeros((n_seq, self.window_size * nf + self.window_size * pe.shape[1]))
        for i in range(n_seq):
            out[i, : self.window_size * nf] = X[i : i + self.window_size].flatten()
            out[i, self.window_size * nf :] = pe.flatten()
        return out

    def fit(self, X, y):
        Xp = self._positional_features(X)
        self.model.fit(Xp, y[self.window_size - 1 :])

    def predict(self, X):
        Xp = self._positional_features(X)
        preds = np.full(len(X), np.nan)
        preds[self.window_size - 1 :] = self.model.predict(Xp)
        return preds


class XGBoostRegressor:
    """XGBoost approximation using sklearn GradientBoostingRegressor."""

    def __init__(self, n_estimators=300, max_depth=6, learning_rate=0.1,
                 subsample=0.8, min_samples_leaf=5):
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, subsample=subsample,
            min_samples_leaf=min_samples_leaf, random_state=42, loss='huber',
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class PCR:
    """Principal Component Regression (PCA + OLS)."""

    def __init__(self, n_components=10):
        self.pca = PCA(n_components=n_components)
        self.lr = LinearRegression()

    def fit(self, X, y):
        self.lr.fit(self.pca.fit_transform(X), y)

    def predict(self, X):
        return self.lr.predict(self.pca.transform(X))


# ======================================================================
# Factory helpers
# ======================================================================

def create_all_models():
    """Instantiate all comparison models used in the paper.

    Returns
    -------
    models : dict[str, estimator]
        Mapping from model name to fitted-ready estimator.
        The special value ``'custom'`` for *PCA+MLR* requires
        manual handling (see ``evaluation.evaluate_model``).
    """
    from .model import CRSS

    return {
        # Proposed
        'CRSS': CRSS(n_features=8, filter_length=12, lambda_r=0.3,
                      epsilon=0.2, max_iter=300, tol=1e-6, n_restarts=5,
                      huber_delta=1.35),
        # Causal / domain-adaptation
        'Causal NN': CausalNN(lag=12, hidden_size=64),
        'IRM': IRM(n_envs=3, hidden_size=64, penalty_weight=1e3),
        'CDA': CDA(n_envs=3, stability_threshold=0.3),
        # Linear
        'OLS': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1, max_iter=2000),
        # Statistical
        'Elastic Net TS': ElasticNetTS(alpha=0.1, l1_ratio=0.5, max_lag=12),
        'Bayesian Ridge': BayesianRidge(max_iter=300, tol=1e-3,
                                        alpha_1=1e-6, alpha_2=1e-6,
                                        lambda_1=1e-6, lambda_2=1e-6),
        'ARD Regression': ARDRegression(max_iter=300, tol=1e-3),
        'PCA+MLR': 'custom',
        'PCR': PCR(n_components=10),
        'Robust PLS': RobustPLS(n_components=5),
        # Kernel
        'SVR (RBF)': SVR(kernel='rbf', C=10.0, epsilon=0.05),
        'SVR (Poly)': SVR(kernel='poly', degree=3, C=10.0, epsilon=0.05),
        'SVR-Lin': SVR(kernel='linear', C=10.0, epsilon=0.05),
        'GPR': GaussianProcessRegressor(
            kernel=RBF(length_scale=1.0) + WhiteKernel(),
            n_restarts_optimizer=3, random_state=42),
        'Kernel Ridge': KernelRidge(alpha=0.1, kernel='rbf'),
        # Tree-based
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=12,
                                               min_samples_leaf=5, random_state=42),
        'Gradient Boost': GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                                     max_depth=5, subsample=0.8,
                                                     random_state=42),
        'XGBoost': XGBoostRegressor(n_estimators=300, max_depth=6, learning_rate=0.1),
        'AdaBoost': AdaBoostRegressor(n_estimators=100, learning_rate=0.5, random_state=42),
        # Neural networks
        'MLP': MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu',
                            solver='adam', max_iter=1000, early_stopping=True,
                            random_state=42),
        'LSTM': LSTM(window_size=12, hidden_size=64),
        'TCN': TCN(n_filters=64, kernel_size=3, n_layers=3),
        'Transformer': TransformerRegressor(window_size=12, d_model=64),
        # Robust
        'Huber Regression': HuberRegressor(epsilon=1.35, max_iter=200),
        'RANSAC': RANSACRegressor(random_state=42),
        'Theil-Sen': TheilSenRegressor(random_state=42, max_iter=500),
    }


def create_ablation_models():
    """Create ablation-study variants of CRSS.

    Returns
    -------
    models : dict[str, estimator]
    """
    from .model import CRSS

    return {
        'CRSS-Full': CRSS(n_features=8, filter_length=12, lambda_r=0.3,
                           epsilon=0.2, max_iter=300, tol=1e-6, n_restarts=5,
                           huber_delta=1.35),
        'CRSS-NoRobust': CRSS(n_features=8, filter_length=12, lambda_r=0.0,
                                epsilon=0.0, max_iter=300, tol=1e-6, n_restarts=5,
                                huber_delta=1e6),
        'CRSS-NoTemporal': CRSS(n_features=8, filter_length=1, lambda_r=0.3,
                                  epsilon=0.2, max_iter=300, tol=1e-6, n_restarts=5,
                                  huber_delta=1.35),
        'CRSS-SingleFeat': CRSS(n_features=1, filter_length=12, lambda_r=0.3,
                                  epsilon=0.2, max_iter=300, tol=1e-6, n_restarts=5,
                                  huber_delta=1.35),
        'Standard PLS': PLSRegression(n_components=5),
    }
