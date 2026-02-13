"""
CRSS -- Causal Robust Soft Sensor
==================================

A spatio-temporal soft sensor for industrial processes under
data distribution drift, based on Wasserstein DRO and Huber-weighted
covariance estimation.

Reference
---------
L. Cao, Y. Qin, D. Zhao, Y. Wang,
"A Novel Causal Robust Soft Sensor for Industrial Processes
 with Data Distribution Drift,"
IEEE Transactions on Industrial Electronics, 2026.
"""

from .model import CRSS, CRSSWithConvergence
from .baselines import (
    create_all_models,
    create_ablation_models,
    # Individual baselines
    ElasticNetTS,
    BayesianRidgeTS,
    LSTM,
    TCN,
    CausalNN,
    RobustPLS,
    IRM,
    CDA,
    TransformerRegressor,
    XGBoostRegressor,
    PCR,
)
from .evaluation import (
    evaluate_model,
    run_extended_robustness,
    run_ablation_study,
    run_hyperparameter_sensitivity,
    run_scalability_experiments,
    run_orthogonality_verification,
    run_conditional_granger_causality,
    run_adf_stationarity_test,
    run_mmd_test,
)

__all__ = [
    # Core
    'CRSS',
    'CRSSWithConvergence',
    # Baselines
    'create_all_models',
    'create_ablation_models',
    'ElasticNetTS',
    'BayesianRidgeTS',
    'LSTM',
    'TCN',
    'CausalNN',
    'RobustPLS',
    'IRM',
    'CDA',
    'TransformerRegressor',
    'XGBoostRegressor',
    'PCR',
    # Evaluation
    'evaluate_model',
    'run_extended_robustness',
    'run_ablation_study',
    'run_hyperparameter_sensitivity',
    'run_scalability_experiments',
    'run_orthogonality_verification',
    'run_conditional_granger_causality',
    'run_adf_stationarity_test',
    'run_mmd_test',
]

__version__ = '1.0.0'
