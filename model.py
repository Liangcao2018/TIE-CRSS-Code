"""
CRSS: Causal Robust Soft Sensor

Core algorithm implementation for the paper:
    "A Novel Causal Robust Soft Sensor for Industrial Processes
     with Data Distribution Drift"
    L. Cao, Y. Qin, D. Zhao, Y. Wang
    IEEE Transactions on Industrial Electronics, 2026.

The CRSS framework extracts spatio-temporal causal features from
multivariate sensor data via alternating maximisation with:
  1. PLS-guided initialisation of spatial projection w
  2. Multiple restarts to escape local optima
  3. Huber-weighted covariance for outlier-robust estimation
  4. Wasserstein DRO-based L1 + L2 regularisation on temporal filter beta
  5. Iterative deflation for sequential feature extraction
"""

import numpy as np


class CRSS:
    """Causal Robust Soft Sensor.

    Parameters
    ----------
    n_features : int, default=5
        Number of causal features to extract.
    filter_length : int, default=12
        Length of the temporal FIR filter (number of lags).
    lambda_r : float, default=0.5
        DRO regularisation strength (controls robustness).
    epsilon : float, default=0.3
        Wasserstein ball radius.
    max_iter : int, default=200
        Maximum alternating-optimisation iterations per feature.
    tol : float, default=1e-5
        Convergence tolerance on spatial projection change.
    n_restarts : int, default=3
        Number of random restarts for global search.
    huber_delta : float, default=1.35
        Huber robustness threshold (in units of median |residual|).
    """

    def __init__(self, n_features=5, filter_length=12, lambda_r=0.5,
                 epsilon=0.3, max_iter=200, tol=1e-5, n_restarts=3,
                 huber_delta=1.35):
        self.n_features = n_features
        self.filter_length = filter_length
        self.lambda_r = lambda_r
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol
        self.n_restarts = n_restarts
        self.huber_delta = huber_delta

        # Learned parameters
        self.W = []       # Spatial projections  (list of 1-D arrays)
        self.Beta = []    # Temporal filters      (list of 1-D arrays)
        self.b = []       # Regression coefficients (list of scalars)
        self.P = []       # Loadings for deflation  (list of 1-D arrays)

        # Training statistics
        self.train_mean = None
        self.train_std = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, Y, tau):
        """Train the CRSS model.

        Parameters
        ----------
        Y : ndarray of shape (N, m)
            Sensor measurement matrix (N time steps, m sensors).
        tau : ndarray of shape (N,)
            Quality variable vector.

        Returns
        -------
        self
        """
        N, m = Y.shape

        self.train_mean = np.mean(Y, axis=0)
        self.train_std = np.std(Y, axis=0) + 1e-10

        # Reset learned parameters
        self.W, self.Beta, self.b, self.P = [], [], [], []

        Y_deflated = Y.copy()
        tau_deflated = tau.copy()

        for j in range(self.n_features):
            # Step 1: extract j-th causal feature (multi-restart)
            w_j, beta_j = self._extract_causal_feature(Y_deflated, tau_deflated)

            # Step 2: compute feature and regression coefficient
            phi_j = self._compute_feature(Y_deflated, w_j, beta_j)
            phi_sq = phi_j @ phi_j + 1e-10
            p_j = Y_deflated.T @ phi_j / phi_sq
            b_j = tau_deflated @ phi_j / phi_sq

            self.W.append(w_j)
            self.Beta.append(beta_j)
            self.b.append(b_j)
            self.P.append(p_j)

            # Step 3: deflation
            Y_deflated = Y_deflated - np.outer(phi_j, p_j)
            tau_deflated = tau_deflated - b_j * phi_j

        return self

    def predict(self, Y_test):
        """Predict quality variable for new sensor data.

        Parameters
        ----------
        Y_test : ndarray of shape (N_test, m)
            Test sensor measurement matrix.

        Returns
        -------
        predictions : ndarray of shape (N_test,)
        """
        predictions = np.zeros(len(Y_test))
        for j in range(self.n_features):
            phi_j = self._compute_feature(Y_test, self.W[j], self.Beta[j])
            predictions += self.b[j] * phi_j
        return predictions

    def get_robustness_bound(self):
        """Compute the theoretical robustness bound (Theorem 2 in the paper).

        Returns
        -------
        bound : float
            Upper bound on prediction degradation under Wasserstein
            perturbation of radius ``epsilon``.
        """
        bound = 0.0
        for j in range(self.n_features):
            bound += abs(self.b[j]) * np.sum(np.abs(self.Beta[j]))
        return self.epsilon * bound

    def interpret_pattern(self, feature_idx):
        """Return a human-readable description of the temporal dynamics
        captured by the *feature_idx*-th causal feature.

        Parameters
        ----------
        feature_idx : int

        Returns
        -------
        description : str
        """
        beta = self.Beta[feature_idx]
        dominant_lag = int(np.argmax(np.abs(beta)))
        if dominant_lag == 0:
            return "Immediate effect"
        elif dominant_lag < 3:
            return "Short-term dynamics"
        elif dominant_lag < 7:
            return "Medium-term effect"
        else:
            return "Long-term dependency"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _huber_weights(self, residuals):
        """Compute Huber weights for robust estimation."""
        abs_r = np.abs(residuals)
        delta = self.huber_delta * np.median(abs_r + 1e-10)
        return np.where(abs_r <= delta, 1.0, delta / (abs_r + 1e-10))

    def _pls_init(self, Y, tau):
        """Initialise w using cross-covariance (PLS1 first step)."""
        cov = Y.T @ tau / len(tau)
        norm = np.linalg.norm(cov)
        if norm > 1e-10:
            return cov / norm
        return np.random.randn(Y.shape[1])

    def _extract_causal_feature(self, Y, tau):
        """Extract a single causal feature with multi-restart + DRO."""
        best_w, best_beta, best_obj = None, None, -np.inf

        for restart in range(self.n_restarts):
            if restart == 0:
                w_init = self._pls_init(Y, tau)
            else:
                rng = np.random.RandomState(42 + restart * 7)
                w_init = rng.randn(Y.shape[1])
                w_init /= np.linalg.norm(w_init)

            w, beta, obj = self._single_restart_extract(Y, tau, w_init)
            if obj > best_obj:
                best_w, best_beta, best_obj = w, beta, obj

        return best_w, best_beta

    def _single_restart_extract(self, Y, tau, w_init):
        """Single alternating-optimisation run."""
        N, m = Y.shape
        s = self.filter_length
        n_eff = N - s + 1

        w = w_init.copy()

        for _ in range(self.max_iter):
            w_old = w.copy()

            # --- Update temporal filter beta (fixed w) ---
            nu = Y @ w                                              # (N,)
            Lambda = np.column_stack(
                [nu[s - 1 - i : N - i] for i in range(s)]
            )                                                       # (n_eff, s)
            tau_trunc = tau[s - 1:]

            # Huber-weighted covariance
            residuals = tau_trunc - Lambda @ (Lambda.T @ tau_trunc / n_eff)
            hw = self._huber_weights(residuals)
            hw_norm = hw / (np.sum(hw) + 1e-10) * n_eff

            c = (Lambda * hw_norm[:, None]).T @ tau_trunc / n_eff

            # Wasserstein DRO regularisation: mixed L1/L2 penalty
            l1_threshold = self.lambda_r * self.epsilon
            l2_penalty = self.lambda_r * self.epsilon * 0.5
            c_reg = np.sign(c) * np.maximum(np.abs(c) - l1_threshold, 0)
            c_reg = c_reg / (1.0 + l2_penalty)

            norm_c = np.linalg.norm(c_reg)
            beta = c_reg / norm_c if norm_c > 1e-10 else c / (np.linalg.norm(c) + 1e-10)

            # --- Update spatial projection w (fixed beta) ---
            g = np.zeros(m)
            tau_lagged = tau[s - 1:]
            for i in range(s):
                Y_lagged = Y[s - 1 - i : N - i, :]               # (n_eff, m)
                cov_term = (Y_lagged * hw_norm[:, None]).T @ tau_lagged / n_eff
                g += beta[i] * cov_term

            # L2 regularisation on w
            g -= self.lambda_r * self.epsilon * 0.1 * w

            norm_g = np.linalg.norm(g)
            if norm_g > 1e-10:
                w = g / norm_g
            else:
                break

            if np.linalg.norm(w - w_old) < self.tol:
                break

        # Objective: absolute Pearson correlation between feature and target
        phi = self._compute_feature(Y, w, beta)
        if np.std(phi[s - 1:]) > 1e-10:
            obj = float(np.abs(np.corrcoef(phi[s - 1:], tau[s - 1:])[0, 1]))
        else:
            obj = 0.0

        return w, beta, obj

    @staticmethod
    def _compute_feature(Y, w, beta):
        """Compute causal feature phi = temporal_filter(Y @ w).

        Parameters
        ----------
        Y : ndarray (N, m)
        w : ndarray (m,)
        beta : ndarray (s,)

        Returns
        -------
        phi : ndarray (N,)
        """
        N = Y.shape[0]
        s = len(beta)
        nu = Y @ w
        phi = np.convolve(nu, beta, mode='full')[:N]
        phi[:s - 1] = 0.0
        return phi


class CRSSWithConvergence(CRSS):
    """CRSS variant that records per-iteration convergence history.

    Useful for analysing and visualising the optimisation trajectory.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.overall_convergence = []    # residual RMSE after each feature
        self.feature_convergence = []    # per-feature iteration histories

    def fit(self, Y, tau):
        N, m = Y.shape
        self.train_mean = np.mean(Y, axis=0)
        self.train_std = np.std(Y, axis=0) + 1e-10

        self.W, self.Beta, self.b, self.P = [], [], [], []
        self.overall_convergence, self.feature_convergence = [], []

        Y_deflated = Y.copy()
        tau_deflated = tau.copy()

        for j in range(self.n_features):
            w_j, beta_j, conv = self._extract_tracked(Y_deflated, tau_deflated)
            self.feature_convergence.append(conv)

            phi_j = self._compute_feature(Y_deflated, w_j, beta_j)
            phi_sq = phi_j @ phi_j + 1e-10
            p_j = Y_deflated.T @ phi_j / phi_sq
            b_j = tau_deflated @ phi_j / phi_sq

            self.W.append(w_j)
            self.Beta.append(beta_j)
            self.b.append(b_j)
            self.P.append(p_j)

            Y_deflated = Y_deflated - np.outer(phi_j, p_j)
            tau_deflated = tau_deflated - b_j * phi_j

            self.overall_convergence.append(float(np.sqrt(np.mean(tau_deflated ** 2))))

        return self

    def _extract_tracked(self, Y, tau):
        """Extract a causal feature while recording convergence metrics."""
        N, m = Y.shape
        s = self.filter_length
        n_eff = N - s + 1
        history = []

        w = self._pls_init(Y, tau)

        for it in range(self.max_iter):
            w_old = w.copy()

            nu = Y @ w
            Lambda = np.column_stack([nu[s - 1 - i : N - i] for i in range(s)])
            tau_trunc = tau[s - 1:]

            residuals = tau_trunc - Lambda @ (Lambda.T @ tau_trunc / n_eff)
            hw = self._huber_weights(residuals)
            hw_norm = hw / (np.sum(hw) + 1e-10) * n_eff

            c = (Lambda * hw_norm[:, None]).T @ tau_trunc / n_eff
            l1_threshold = self.lambda_r * self.epsilon
            l2_penalty = self.lambda_r * self.epsilon * 0.5
            c_reg = np.sign(c) * np.maximum(np.abs(c) - l1_threshold, 0)
            c_reg /= (1.0 + l2_penalty)

            norm_c = np.linalg.norm(c_reg)
            beta = c_reg / norm_c if norm_c > 1e-10 else c / (np.linalg.norm(c) + 1e-10)

            g = np.zeros(m)
            tau_lagged = tau[s - 1:]
            for i in range(s):
                Y_lagged = Y[s - 1 - i : N - i, :]
                cov_term = (Y_lagged * hw_norm[:, None]).T @ tau_lagged / n_eff
                g += beta[i] * cov_term
            g -= self.lambda_r * self.epsilon * 0.1 * w

            norm_g = np.linalg.norm(g)
            if norm_g > 1e-10:
                w = g / norm_g
            else:
                break

            w_change = float(np.linalg.norm(w - w_old))
            history.append({
                'iteration': it + 1,
                'w_change': w_change,
                'w_norm': float(np.linalg.norm(w)),
            })

            if w_change < self.tol:
                break

        return w, beta, history
