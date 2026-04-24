"""
gp.py - Rotated Anisotropic Gaussian Process Regression
"""
from typing import Optional, Tuple, Any, Dict
import numpy as np
import warnings
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.optimize import minimize as sp_minimize
import optuna

class RotatedGPR(BaseEstimator, RegressorMixin):
    """
    Gaussian Process Regressor with automatic anisotropy rotation optimization.
    
    Learns optimal rotation angle for anisotropic spatial correlation structure.
    Uses two-level optimization: outer loop for angle, inner for kernel hyperparameters.
    """
    
    def __init__(
        self,
        kernel,
        alpha: float = 1e-10,
        optimizer: str = "fmin_l_bfgs_b",
        n_restarts_optimizer: int = 10,
        angle_search_method: str = "bounded",
        angle_precision: float = 0.5,
        center_coords: bool = True,
        max_anisotropy: Optional[float] = None,
        random_state: Optional[int] = None,
        angle_bounds: Tuple[float, float] = (0, 180)
    ):
        self.kernel = kernel
        self.alpha = alpha
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.angle_search_method = angle_search_method
        self.angle_precision = angle_precision
        self.center_coords = center_coords
        self.max_anisotropy = max_anisotropy
        self.random_state = random_state
        self.angle_bounds = angle_bounds
        
        self.best_angle_deg_ = None
        self.gp_model_ = None
        self.X_center_ = None
        self.log_marginal_likelihood_ = None
        self.angle_search_history_ = []

    def _center_coordinates(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        center = np.mean(X, axis=0)
        return X - center, center

    def _rotate_coords(self, X: np.ndarray, angle_deg: float) -> np.ndarray:
        theta = np.radians(angle_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rotation_matrix = np.array([
            [cos_t, -sin_t],
            [sin_t,  cos_t]
        ])
        return X @ rotation_matrix

    def _eval_lml(self, angle, theta, alpha):
        test_kernel = clone(self.kernel)
        if len(test_kernel.bounds) != len(theta):
            return float('inf')
        test_kernel.theta = theta
        X_rot = self._rotate_coords(self.X_train_centered_, angle)
        gp = GaussianProcessRegressor(
            kernel=test_kernel, alpha=alpha,
            optimizer=None, normalize_y=True,
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp.fit(X_rot, self.y_train_)
                return -gp.log_marginal_likelihood()
        except Exception:
            return float('inf')

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RotatedGPR':
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        self.X_train_ = np.asarray(X, dtype=np.float64)
        self.y_train_ = np.asarray(y, dtype=np.float64)

        if self.X_train_.shape[1] != 2:
            raise ValueError(f"X must have exactly 2 columns, got {self.X_train_.shape[1]}")

        if self.center_coords:
            self.X_train_centered_, self.X_center_ = self._center_coordinates(self.X_train_)
        else:
            self.X_train_centered_ = self.X_train_
            self.X_center_ = np.zeros(2)

        self.angle_search_history_ = []
        k_bounds = self.kernel.bounds
        max_ratio_log = np.log(self.max_anisotropy) if self.max_anisotropy and self.max_anisotropy > 1.0 else 0.0

        def objective(trial):
            angle = trial.suggest_float('angle', self.angle_bounds[0], self.angle_bounds[1])
            c_val = trial.suggest_float('theta_0', k_bounds[0][0], k_bounds[0][1])
            l_maj = trial.suggest_float('l_maj', k_bounds[1][0], k_bounds[1][1])
            ratio_log = trial.suggest_float('ratio_log', 0.0, max_ratio_log) if max_ratio_log > 0 else 0.0
            log_alpha = trial.suggest_float('log_alpha', -14, 0)

            theta = np.array([c_val, l_maj, l_maj - ratio_log])
            alpha = 10.0 ** log_alpha
            nlml = self._eval_lml(angle, theta, alpha)
            if np.isfinite(nlml):
                self.angle_search_history_.append((angle, nlml))
            return nlml

        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        study.optimize(objective, n_trials=200)
        self.study_ = study

        bp = study.best_params
        best_angle = bp['angle']
        best_c     = bp['theta_0']
        best_lmaj  = bp['l_maj']
        best_ratio = bp.get('ratio_log', 0.0)
        best_la    = bp['log_alpha']

        def _pack(angle, c, lmaj, ratio, la):
            return np.array([angle, c, lmaj, ratio, la])

        def _obj_vec(vec):
            a, c, lm, r, la = vec
            theta = np.array([c, lm, lm - r])
            return self._eval_lml(a, theta, 10.0 ** la)

        x0 = _pack(best_angle, best_c, best_lmaj, best_ratio, best_la)
        bounds_refine = [
            self.angle_bounds,
            (k_bounds[0][0], k_bounds[0][1]),
            (k_bounds[1][0], k_bounds[1][1]),
            (0.0, max_ratio_log if max_ratio_log > 0 else 0.01),
            (-14, 0),
        ]
        try:
            res = sp_minimize(_obj_vec, x0, method='L-BFGS-B',
                              bounds=bounds_refine,
                              options={'ftol': 1e-12, 'maxiter': 200})
            if res.fun < study.best_value:
                best_angle, best_c, best_lmaj, best_ratio, best_la = res.x
        except Exception:
            pass

        self.best_angle_deg_ = float(best_angle)
        self.best_alpha_     = float(10.0 ** best_la)

        best_theta = np.array([best_c, best_lmaj, best_lmaj - best_ratio])
        best_kernel = clone(self.kernel)
        best_kernel.theta = best_theta

        X_final = self._rotate_coords(self.X_train_centered_, self.best_angle_deg_)
        self.gp_model_ = GaussianProcessRegressor(
            kernel=best_kernel, alpha=self.best_alpha_,
            optimizer=None, normalize_y=True,
            random_state=self.random_state,
        )
        self.gp_model_.fit(X_final, self.y_train_)
        self.log_marginal_likelihood_ = self.gp_model_.log_marginal_likelihood()

        return self

    def predict(self, X: np.ndarray, return_std: bool = False, return_cov: bool = False):
        X = np.asarray(X, dtype=np.float64)
        X_centered = X - self.X_center_ if self.center_coords else X
        X_rot = self._rotate_coords(X_centered, self.best_angle_deg_)
        return self.gp_model_.predict(X_rot, return_std=return_std, return_cov=return_cov)

    @property
    def kernel_(self):
        return self.gp_model_.kernel_

    def get_kernel_params(self) -> Dict[str, Any]:
        kernel_params = self.kernel_.get_params()
        constant = 1.0
        length_scale = [1.0, 1.0]
        noise = getattr(self, 'best_alpha_', self.alpha)
        
        for key, val in kernel_params.items():
            if "constant_value" in key and "bounds" not in key:
                constant = float(val)
            elif "length_scale" in key and "bounds" not in key:
                length_scale = val if hasattr(val, '__len__') else [val, val]
            elif "noise_level" in key and "bounds" not in key:
                noise = float(val)
        
        length_scale = np.asarray(length_scale, dtype=np.float64)
        anisotropy_ratio = np.max(length_scale) / np.min(length_scale)
        
        return {
            "rotation_angle_deg": float(self.best_angle_deg_),
            "constant_value": constant,
            "length_scale": length_scale.tolist(),
            "noise_level": noise,
            "anisotropy_ratio": float(anisotropy_ratio),
            "log_marginal_likelihood": float(self.log_marginal_likelihood_)
        }
