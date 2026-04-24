"""
kriging.py - Ordinary Kriging with Optuna-based anisotropic parameter optimization
"""
from typing import Optional, Dict, Any
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cluster import KMeans
import optuna

def stable_variogram_model(params, dists):
    psill, r, nugget, alpha = params
    return psill * (1.0 - np.exp(-(dists / r) ** alpha)) + nugget

def circular_variogram_model(params, dists):
    psill, r, nugget = params
    mask = dists <= r
    val = np.full_like(dists, psill + nugget)
    h_r = dists[mask] / r
    term = (2.0 / np.pi) * (np.arccos(h_r) - h_r * np.sqrt(1.0 - h_r**2))
    val[mask] = psill * (1.0 - term) + nugget
    return val

def rational_quadratic_variogram_model(params, dists):
    psill, r, nugget, alpha = params
    return psill * (1.0 - (1.0 + dists**2 / (2.0 * alpha * r**2)) ** (-alpha)) + nugget


class AnisotropicKriging(BaseEstimator, RegressorMixin):
    """
    Ordinary Kriging with Optuna-based anisotropic parameter optimization.
    Supports all native PyKrige models plus custom Stable, Circular, and Rational Quadratic models.
    """
    
    NATIVE_MODELS = ['linear', 'power', 'gaussian', 'spherical', 'exponential', 'hole-effect']
    CUSTOM_MODELS = {
        'stable': stable_variogram_model,
        'circular': circular_variogram_model,
        'rational-quadratic': rational_quadratic_variogram_model
    }

    # Minimum recommended trials: TPE needs ~25 random startup trials before it
    # starts making informed suggestions.  With 9 categorical model choices, the
    # effective search space is large enough that fewer than 100 trials gives
    # unreliable results.
    _MIN_TRIALS_RECOMMENDED = 100

    def __init__(
        self,
        n_trials: int = 150,
        n_splits: int = 5,
        verbose: bool = False,
        random_state: Optional[int] = None,
        max_anisotropy: float = 3.0
    ):
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.verbose = verbose
        self.random_state = random_state
        self.max_anisotropy = max_anisotropy

        self.model_ = None
        self.best_params_ = None
        self.best_model_name_ = None
        self.study_ = None

    def _get_ok_instance(self, X, y, params, model_name):
        from pykrige.ok import OrdinaryKriging
        
        psill = params['psill']
        v_range = params['range']
        nugget = params['nugget']
        angle = params['angle']
        scaling = params['scaling']
        
        kwargs = {
            'variogram_model': model_name if model_name in self.NATIVE_MODELS else 'custom',
            'anisotropy_angle': angle,
            'anisotropy_scaling': scaling,
            'verbose': False,
            'enable_plotting': False
        }
        
        if model_name in self.NATIVE_MODELS:
            kwargs['variogram_parameters'] = [psill, v_range, nugget]
        else:
            kwargs['variogram_function'] = self.CUSTOM_MODELS[model_name]
            v_params = [psill, v_range, nugget]
            if model_name in ['stable', 'rational-quadratic']:
                v_params.append(params['alpha'])
            kwargs['variogram_parameters'] = v_params
            
        return OrdinaryKriging(X[:, 0], X[:, 1], y, **kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AnisotropicKriging':
        if not self.verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Guard: warn the user if n_trials is too low for reliable optimisation.
        # The search space has 9 categorical model types + ~6 continuous params;
        # TPE requires ~25 warm-up (random) trials before it can guide search.
        if self.n_trials < self._MIN_TRIALS_RECOMMENDED:
            import warnings
            warnings.warn(
                f"AnisotropicKriging: n_trials={self.n_trials} is below the "
                f"recommended minimum of {self._MIN_TRIALS_RECOMMENDED}. "
                f"The optimiser may not converge to a good solution. "
                f"Consider setting n_trials >= {self._MIN_TRIALS_RECOMMENDED} "
                f"in your config.yaml.",
                UserWarning,
                stacklevel=2,
            )

        X = np.asarray(X)
        y = np.asarray(y)
        
        clusters = KMeans(
            n_clusters=self.n_splits, random_state=self.random_state, n_init=10
        ).fit_predict(X)
        
        data_var   = float(np.var(y))
        max_dist   = float(np.sqrt((X[:, 0].max() - X[:, 0].min())**2 +
                                   (X[:, 1].max() - X[:, 1].min())**2))
        
        def objective(trial):
            model_name = trial.suggest_categorical(
                'model', self.NATIVE_MODELS + list(self.CUSTOM_MODELS.keys())
            )
            
            psill   = trial.suggest_float('psill',   data_var * 0.05, data_var * 3.0)
            v_range = trial.suggest_float('range',   max_dist * 0.02, max_dist * 1.5)
            nugget  = trial.suggest_float('nugget',  0.0, data_var * 0.8)
            angle   = trial.suggest_float('angle',   0.0, 180.0)
            scaling = trial.suggest_float('scaling', 1.0, self.max_anisotropy)
            
            params = {
                'psill': psill, 'range': v_range, 'nugget': nugget,
                'angle': angle, 'scaling': scaling
            }
            
            if model_name in ['stable', 'rational-quadratic']:
                params['alpha'] = trial.suggest_float('alpha', 0.1, 2.0)
            
            scores = []
            for fold in range(self.n_splits):
                train_idx = np.where(clusters != fold)[0]
                val_idx   = np.where(clusters == fold)[0]
                
                if len(train_idx) < 5 or len(val_idx) < 1:
                    continue
                
                try:
                    ok = self._get_ok_instance(
                        X[train_idx], y[train_idx], params, model_name
                    )
                    y_pred, _ = ok.execute('points', X[val_idx, 0], X[val_idx, 1])
                    mse = float(np.mean((y[val_idx] - y_pred) ** 2))
                    scores.append(mse)
                except Exception:
                    return float('inf')
            
            return np.mean(scores) if scores else float('inf')

        self.study_ = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=self.random_state))
        self.study_.optimize(objective, n_trials=self.n_trials)

        # ── Extract best results without mutating the Optuna params dict ──────
        # Previously `best_params_.pop('model')` permanently modified the dict,
        # which caused fragile ordering dependency in downstream CV and reporting
        # code.  We now copy the dict first, then extract model name separately.
        _raw_best = self.study_.best_params          # original Optuna dict — untouched
        self.best_model_name_ = _raw_best['model']   # store model name as its own attr
        self.best_params_ = {k: v for k, v in _raw_best.items() if k != 'model'}

        self.model_ = self._get_ok_instance(X, y, self.best_params_, self.best_model_name_)

        return self

    def predict(self, X: np.ndarray, return_std: bool = False):
        X = np.asarray(X)
        y_pred, y_var = self.model_.execute('points', X[:, 0], X[:, 1])
        if return_std:
            return y_pred, np.sqrt(np.abs(y_var))
        return y_pred

    def get_kernel_params(self) -> Dict[str, Any]:
        params = {
            "model_type": "Kriging",
            "best_model": self.best_model_name_,
            "rotation_angle_deg": float(self.best_params_.get('angle', 0)),
            "anisotropy_ratio": float(self.best_params_.get('scaling', 1.0)),
            "psill": float(self.best_params_.get('psill', 0)),
            "range": float(self.best_params_.get('range', 0)),
            "nugget": float(self.best_params_.get('nugget', 0)),
        }
        if 'alpha' in self.best_params_:
            params['alpha'] = float(self.best_params_['alpha'])
        return params
