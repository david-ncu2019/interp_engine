"""
kriging.py - Ordinary Kriging with Optuna-based anisotropic parameter optimization
"""
from typing import Optional, Dict, Any
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cluster import KMeans
import optuna
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

from src.validation import (
    validate_finite,
    validate_min_samples,
    validate_not_constant,
    validate_2d_coordinates,
    validate_shape_match,
)

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
        max_anisotropy: float = 3.0,
        n_jobs: int = 1
    ):
        if max_anisotropy < 1.0:
            raise ValueError(
                f"max_anisotropy must be >= 1.0, got {max_anisotropy}. "
                f"An anisotropy ratio < 1 would invert the major/minor axis."
            )
        if n_splits < 2:
            raise ValueError(
                f"n_splits must be >= 2 for cross-validation, got {n_splits}."
            )
        if n_trials < 1:
            raise ValueError(
                f"n_trials must be >= 1, got {n_trials}."
            )

        self.n_trials = n_trials
        self.n_splits = n_splits
        self.verbose = verbose
        self.random_state = random_state
        self.max_anisotropy = max_anisotropy
        self.n_jobs = n_jobs

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
        # ── Input validation ──────────────────────────────────────────────
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        validate_finite(X, y)
        validate_shape_match(X, y)
        validate_2d_coordinates(X)
        validate_not_constant(y)
        # Require at least n_splits+2 points so each fold has ≥2 training points
        validate_min_samples(X, y, min_samples=max(self.n_splits + 2, 5))

        # Suppress Optuna's default logger to reduce clutter
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Guard: warn the user if n_trials is too low for reliable optimisation.
        if self.n_trials < self._MIN_TRIALS_RECOMMENDED:
            import warnings
            warnings.warn(
                f"AnisotropicKriging: n_trials={self.n_trials} is below the "
                f"recommended minimum of {self._MIN_TRIALS_RECOMMENDED}.",
                UserWarning,
                stacklevel=2,
            )

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
                except Exception as e:
                    logger.debug("Kriging fold %d failed in trial %d: %s",
                                 fold, trial.number, e)
                    return float('inf')
            
            return np.mean(scores) if scores else float('inf')

        self.study_ = optuna.create_study(
            direction='minimize', 
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )

        def logging_callback(study, trial):
            if self.verbose:
                val = trial.value
                val_str = f"{val:.2e}" if val != float('inf') else "inf"
                best_val = f"{study.best_value:.2e}"
                
                # Format parameters with reduced precision
                p = trial.params
                p_str = f"model={p['model']}, psill={p['psill']:.2e}, range={p['range']:.1f}, nugget={p['nugget']:.2e}"
                
                logger.debug("  [Trial %d] val=%s, best=%s | %s", trial.number, val_str, best_val, p_str)

        _show_pbar = self.verbose and self.n_jobs <= 1
        with tqdm(total=self.n_trials, desc="Optuna (Kriging)",
                  disable=not _show_pbar, leave=False) as pbar:
            self.study_.optimize(
                objective, n_trials=self.n_trials,
                callbacks=[logging_callback, lambda s, t: pbar.update(1)],
                n_jobs=self.n_jobs,
            )

        # ── Guard: study may have zero successful trials ───────────────────
        if len(self.study_.trials) == 0 or all(
            t.value is None or not np.isfinite(t.value)
            for t in self.study_.trials
        ):
            raise RuntimeError(
                "Kriging optimisation failed: no trial produced a finite CV score. "
                "This usually means the data is too sparse, too noisy, or has "
                "degenerate geometry (e.g. all points colinear). "
                "Try: (1) increasing n_trials, (2) checking for duplicate/colinear "
                "coordinates, or (3) using a simpler interpolation method."
            )

        # ── Extract best results without mutating the Optuna params dict ──────
        _raw_best = self.study_.best_params          # original Optuna dict — untouched
        self.best_model_name_ = _raw_best['model']   # store model name as its own attr
        self.best_params_ = {k: v for k, v in _raw_best.items() if k != 'model'}

        # ── Final model fit ─────────────────────────────────────────────────
        try:
            self.model_ = self._get_ok_instance(
                X, y, self.best_params_, self.best_model_name_
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to build the final kriging system with the optimised "
                f"parameters. The covariance matrix is likely singular due to "
                f"near-duplicate points or extreme anisotropy. "
                f"Underlying error: {type(e).__name__}: {e}"
            ) from e

        return self

    def fit_with_known_params(self, X: np.ndarray, y: np.ndarray, best_model_name: str, best_params: dict) -> 'AnisotropicKriging':
        """Fit the model instantly using pre-computed optimal parameters (skips Optuna search)."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        validate_finite(X, y)
        validate_shape_match(X, y)
        validate_2d_coordinates(X)
        validate_not_constant(y)

        self.best_model_name_ = best_model_name
        self.best_params_ = best_params

        try:
            self.model_ = self._get_ok_instance(X, y, self.best_params_, self.best_model_name_)
        except Exception as e:
            raise RuntimeError(
                f"Failed to build kriging system with provided parameters. "
                f"The covariance matrix may be singular. "
                f"Underlying error: {type(e).__name__}: {e}"
            ) from e

        return self

    def predict(self, X: np.ndarray, return_std: bool = False):
        X = np.asarray(X, dtype=np.float64)
        validate_finite(X, name="prediction coordinates")
        if self.model_ is None:
            raise RuntimeError(
                "Model has not been fitted. Call fit() or fit_with_known_params() "
                "before predict()."
            )
        y_pred, y_var = self.model_.execute('points', X[:, 0], X[:, 1])
        # Guard against numerical garbage from PyKrige on out-of-domain points
        if not np.all(np.isfinite(y_pred)):
            import warnings
            n_bad = int((~np.isfinite(y_pred)).sum())
            warnings.warn(
                f"{n_bad} prediction(s) are non-finite (NaN/Inf). "
                f"This can happen when predicting far outside the data envelope. "
                f"Consider clipping prediction coordinates to the convex hull."
            )
            y_pred = np.nan_to_num(y_pred, nan=np.nanmean(y_pred), posinf=np.nanmax(y_pred[np.isfinite(y_pred)]), neginf=np.nanmin(y_pred[np.isfinite(y_pred)]))
            y_var = np.nan_to_num(y_var, nan=0.0, posinf=0.0, neginf=0.0)
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
