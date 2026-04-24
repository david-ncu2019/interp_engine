"""
gp.py - Rotated Anisotropic Gaussian Process Regression

Architecture
------------
The central problem with a fixed RBF (Gaussian) kernel is that it forces
infinite smoothness onto the spatial field.  Real geoscientific data —
especially SGS-simulated fields — can be rough (Matérn-3/2), moderately
smooth (Matérn-5/2), or genuinely smooth (RBF = Matérn-∞).  Locking the
kernel to RBF means the optimizer can never compensate for the wrong
smoothness assumption regardless of how well it tunes the length scales.

Fix: a *mixture* kernel that lets Optuna select among three candidate
     kernel types during optimization, each with its own anisotropic
     length scale pair.  The type with the highest log marginal likelihood
     wins and is used for the final fit.

Additionally:
- An explicit WhiteKernel nugget term allows the model to absorb
  measurement noise and high-nugget variogram structure without forcing
  the covariance matrix to near-singularity.
- Length scale bounds are passed in from main.py where they are computed
  adaptively from the data geometry (median NN distance, max pairwise
  distance), not from fixed config values.
- The alpha (jitter) lower bound is raised to 1e-6 to prevent numerical
  instability on ill-conditioned covariance matrices.
"""

from typing import Optional, Tuple, Any, Dict
import numpy as np
import warnings
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel, RBF, Matern, WhiteKernel
)
from scipy.optimize import minimize as sp_minimize
import optuna


# ── Kernel candidate catalogue ────────────────────────────────────────────────
# Each entry maps a short name to the sklearn kernel class and whether it
# accepts a 'nu' smoothness parameter.
_KERNEL_CATALOGUE = {
    "matern_32":  (Matern, {"nu": 1.5}),   # rough: C¹ continuity
    "matern_52":  (Matern, {"nu": 2.5}),   # moderate: C² continuity
    "rbf":        (RBF,    {}),            # smooth: C∞ continuity
}


def build_base_kernel(kind: str, length_scale: float, ls_bounds: tuple) -> object:
    """
    Instantiate a single anisotropic (2-D) spatial kernel.

    Parameters
    ----------
    kind        : one of 'matern_32', 'matern_52', 'rbf'
    length_scale: initial value for both axes (will be optimized)
    ls_bounds   : (ls_min, ls_max) in the same units as the coordinates
    """
    cls, extra = _KERNEL_CATALOGUE[kind]
    return cls(
        length_scale=[length_scale, length_scale],
        length_scale_bounds=ls_bounds,
        **extra,
    )


class RotatedGPR(BaseEstimator, RegressorMixin):
    """
    Gaussian Process Regressor with automatic anisotropy rotation optimization
    and automatic kernel smoothness selection.

    Key design decisions
    --------------------
    1. **Kernel mixture selection** – Optuna searches over three kernel types
       (Matérn-3/2, Matérn-5/2, RBF) simultaneously.  The winning type is
       determined by the log marginal likelihood, not pre-specified by the user.

    2. **Explicit nugget (WhiteKernel)** – an additive white noise term lets
       the model handle high-nugget variogram structures.  Its variance is
       optimized jointly with all other parameters.

    3. **Adaptive length scale bounds** – ls_bounds is passed from main.py
       where it is computed from the data's actual spatial geometry
       (0.5× median nearest-neighbour distance to 0.6× max pairwise distance).

    4. **Safe alpha floor** – alpha (jitter) lower bound is 1e-6, preventing
       the covariance matrix from becoming numerically singular.

    5. **Two-stage optimization** – Optuna TPE (300 trials) for global
       exploration, followed by L-BFGS-B for local refinement.

    Parameters
    ----------
    ls_init     : float – initial length scale value (both axes)
    ls_bounds   : (float, float) – (min, max) for the length scale search
    var_init    : float – initial signal variance (ConstantKernel)
    var_bounds  : (float, float) – (min, max) for signal variance
    nugget_init : float – initial nugget variance (WhiteKernel)
    nugget_bounds: (float, float) – (min, max) for nugget variance
    max_anisotropy : float or None – upper bound on ls_major/ls_minor ratio
    angle_bounds   : (float, float) – search range for rotation angle in degrees
    n_optuna_trials: int – number of Optuna trials (default 300)
    random_state   : int or None
    center_coords  : bool – center training coordinates before fitting
    """

    def __init__(
        self,
        ls_init:          float = 100.0,
        ls_bounds:        Tuple[float, float] = (1.0, 5000.0),
        var_init:         float = 1.0,
        var_bounds:       Tuple[float, float] = (1e-3, 1e3),
        nugget_init:      float = 0.01,
        nugget_bounds:    Tuple[float, float] = (1e-6, 1.0),
        max_anisotropy:   Optional[float] = 15.0,
        angle_bounds:     Tuple[float, float] = (0.0, 180.0),
        n_optuna_trials:  int = 300,
        random_state:     Optional[int] = None,
        center_coords:    bool = True,
    ):
        self.ls_init          = ls_init
        self.ls_bounds        = ls_bounds
        self.var_init         = var_init
        self.var_bounds       = var_bounds
        self.nugget_init      = nugget_init
        self.nugget_bounds    = nugget_bounds
        self.max_anisotropy   = max_anisotropy
        self.angle_bounds     = angle_bounds
        self.n_optuna_trials  = n_optuna_trials
        self.random_state     = random_state
        self.center_coords    = center_coords

        # Set after fit()
        self.best_angle_deg_          = None
        self.best_kernel_type_        = None
        self.gp_model_                = None
        self.X_center_                = None
        self.log_marginal_likelihood_ = None
        self.best_alpha_              = None
        self.angle_search_history_    = []

    # ── coordinate helpers ────────────────────────────────────────────────────

    def _center_coordinates(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        center = np.mean(X, axis=0)
        return X - center, center

    def _rotate_coords(self, X: np.ndarray, angle_deg: float) -> np.ndarray:
        theta  = np.radians(angle_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([[cos_t, -sin_t],
                      [sin_t,  cos_t]])
        return X @ R

    # ── LML evaluator ─────────────────────────────────────────────────────────

    def _eval_lml(
        self,
        angle:       float,
        kernel_type: str,
        log_var:     float,
        log_lmaj:    float,
        ratio_log:   float,
        log_nugget:  float,
        log_alpha:   float,
    ) -> float:
        """
        Evaluate negative log marginal likelihood for one parameter combination.

        Parameters are all in log-space so the optimizer treats them on a
        uniform scale and cannot accidentally set them to zero.

        ratio_log : ln(ls_major / ls_minor) ≥ 0
                    → ls_minor = exp(log_lmaj - ratio_log)
                    → ratio_log = 0 means isotropic
        """
        var      = np.exp(log_var)
        ls_maj   = np.exp(log_lmaj)
        ls_min   = np.exp(log_lmaj - ratio_log)
        nugget   = np.exp(log_nugget)
        alpha    = np.exp(log_alpha)

        base_k  = build_base_kernel(kernel_type, ls_maj, self.ls_bounds)
        base_k.length_scale = [ls_maj, ls_min]

        kernel = ConstantKernel(
            constant_value=var,
            constant_value_bounds=self.var_bounds,
        ) * base_k + WhiteKernel(
            noise_level=nugget,
            noise_level_bounds=self.nugget_bounds,
        )

        X_rot = self._rotate_coords(self.X_train_centered_, angle)
        gp = GaussianProcessRegressor(
            kernel=kernel, alpha=alpha,
            optimizer=None, normalize_y=True,
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp.fit(X_rot, self.y_train_)
                return -gp.log_marginal_likelihood()
        except Exception:
            return float("inf")

    # ── coarse angle scan helper ──────────────────────────────────────────────

    def _coarse_angle_scan(
        self,
        step_deg:    float,
        log_ls_mid:  float,
        log_var_mid: float,
        log_nug_mid: float,
    ) -> Tuple[float, float]:
        """
        Evaluate NLL on a coarse angular grid to find a good starting angle.

        This is Stage 0 of the optimization sequence. It runs very cheaply
        (isotropic kernel, median log-space parameter values) across
        ``n_steps = ceil(angle_range / step_deg)`` candidate angles and
        returns the angle with the lowest NLL together with its NLL value.

        Scientific rationale
        --------------------
        For extreme anisotropy (S5 scenario, ratio ~15×), the LML landscape
        has a single very narrow valley in the angle dimension.  When Optuna
        explores angles uniformly over [0°, 180°], the probability of
        landing inside the valley in the first 50–100 trials is low, so the
        TPE model never learns to focus there.

        A cheap 10°-resolution grid scan costs only ~18 LML evaluations
        but reliably identifies the correct valley.  Optuna then refines
        within a ±20° window, which is ~7× cheaper per trial (smaller
        search space) and ~10× more likely to converge to the true optimum.

        Args:
            step_deg    : angular resolution of the coarse grid (default 10°)
            log_ls_mid  : ln(length_scale) mid-point — isotropic starting guess
            log_var_mid : ln(signal_variance) mid-point
            log_nug_mid : ln(nugget_variance) mid-point

        Returns:
            (best_angle_deg, best_nll)
        """
        angle_min, angle_max = self.angle_bounds
        candidates = np.arange(angle_min, angle_max + 1e-6, step_deg)
        # Use zero ratio_log (isotropic) and mid-range alpha for speed
        log_alpha_mid = (np.log(1e-6) + np.log(0.1)) / 2.0

        best_angle = angle_min
        best_nll   = float("inf")
        for ang in candidates:
            nll = self._eval_lml(
                angle       = float(ang),
                kernel_type = "matern_52",   # moderate smoothness for the scan
                log_var     = log_var_mid,
                log_lmaj    = log_ls_mid,
                ratio_log   = 0.0,           # isotropic — scan is about direction only
                log_nugget  = log_nug_mid,
                log_alpha   = log_alpha_mid,
            )
            if nll < best_nll:
                best_nll   = nll
                best_angle = float(ang)

        return best_angle, best_nll

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RotatedGPR":
        """
        Fit the model to spatial data (X: N×2 coordinates, y: N values).

        Optimization sequence
        ---------------------
        0. Coarse angle grid scan (0°→180° in 10° steps, isotropic kernel).
           Identifies which angular band contains the anisotropy axis.
        1. Optuna TPE — refined search over kernel type, angle (±20° window
           around coarse best), length scales, nugget, and jitter alpha.
        2. L-BFGS-B  — local refinement starting from Optuna's best point.
        3. Final GaussianProcessRegressor fit with the winning parameters.

        The two-stage angle search (Steps 0 + 1) is the key improvement for
        extreme-anisotropy scenarios.  Without it, Optuna's TPE sampler must
        discover the narrow angular valley on its own, which requires many
        more trials and often misses it entirely with n_trials=300.
        """
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        self.X_train_ = np.asarray(X, dtype=np.float64)
        self.y_train_ = np.asarray(y, dtype=np.float64)

        if self.X_train_.shape[1] != 2:
            raise ValueError(
                f"X must have exactly 2 columns, got {self.X_train_.shape[1]}"
            )

        if self.center_coords:
            self.X_train_centered_, self.X_center_ = \
                self._center_coordinates(self.X_train_)
        else:
            self.X_train_centered_ = self.X_train_
            self.X_center_         = np.zeros(2)

        self.angle_search_history_ = []

        # ── Derived log-space bounds ──────────────────────────────────────────
        log_ls_min  = np.log(max(self.ls_bounds[0], 1e-6))
        log_ls_max  = np.log(self.ls_bounds[1])
        log_var_min = np.log(max(self.var_bounds[0], 1e-9))
        log_var_max = np.log(self.var_bounds[1])
        log_nug_min = np.log(max(self.nugget_bounds[0], 1e-9))
        log_nug_max = np.log(self.nugget_bounds[1])
        # ratio_log = ln(anisotropy_ratio) ≥ 0
        max_ratio_log = (
            np.log(self.max_anisotropy)
            if self.max_anisotropy and self.max_anisotropy > 1.0
            else 0.0
        )
        # Alpha (jitter): floor at 1e-6 to prevent singular covariance matrices.
        # Ceiling at 0.1 to prevent noise-dominated fits.
        log_alpha_min = np.log(1e-6)
        log_alpha_max = np.log(0.1)

        # ── Stage 0: Coarse angular grid scan ────────────────────────────────
        # Use mid-point log-space values for the cheap isotropic scan.
        log_ls_mid  = (log_ls_min  + log_ls_max)  / 2.0
        log_var_mid = (log_var_min + log_var_max) / 2.0
        log_nug_mid = (log_nug_min + log_nug_max) / 2.0

        coarse_best_angle, _ = self._coarse_angle_scan(
            step_deg    = 10.0,
            log_ls_mid  = log_ls_mid,
            log_var_mid = log_var_mid,
            log_nug_mid = log_nug_mid,
        )

        # Build a ±20° window around the coarse winner, clipped to angle_bounds.
        fine_window = 20.0
        angle_min_fine = max(self.angle_bounds[0], coarse_best_angle - fine_window)
        angle_max_fine = min(self.angle_bounds[1], coarse_best_angle + fine_window)
        # Safety: if clipping collapses the window (edge of range), expand back.
        if angle_max_fine - angle_min_fine < 5.0:
            angle_min_fine = max(self.angle_bounds[0], coarse_best_angle - fine_window)
            angle_max_fine = min(self.angle_bounds[1], angle_min_fine + 40.0)

        print(f"       [GP angle scan] coarse best angle: {coarse_best_angle:.1f}°  "
              f"→ Optuna search window: [{angle_min_fine:.1f}°, {angle_max_fine:.1f}°]")

        # ── Stage 1: Optuna refined search inside the angle window ────────────
        def objective(trial):
            kernel_type = trial.suggest_categorical(
                "kernel_type", list(_KERNEL_CATALOGUE.keys())
            )
            angle     = trial.suggest_float("angle",      angle_min_fine, angle_max_fine)
            log_var   = trial.suggest_float("log_var",    log_var_min, log_var_max)
            log_lmaj  = trial.suggest_float("log_lmaj",   log_ls_min,  log_ls_max)
            ratio_log = (
                trial.suggest_float("ratio_log", 0.0, max_ratio_log)
                if max_ratio_log > 0.0 else 0.0
            )
            log_nugget = trial.suggest_float("log_nugget", log_nug_min, log_nug_max)
            log_alpha  = trial.suggest_float("log_alpha",  log_alpha_min, log_alpha_max)

            nlml = self._eval_lml(
                angle, kernel_type, log_var, log_lmaj,
                ratio_log, log_nugget, log_alpha
            )
            if np.isfinite(nlml):
                self.angle_search_history_.append((angle, nlml))
            return nlml

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        study.optimize(objective, n_trials=self.n_optuna_trials)
        self.study_ = study

        bp = study.best_params
        best_kernel_type = bp["kernel_type"]
        best_angle    = bp["angle"]
        best_log_var  = bp["log_var"]
        best_log_lmaj = bp["log_lmaj"]
        best_ratio    = bp.get("ratio_log", 0.0)
        best_log_nug  = bp["log_nugget"]
        best_log_alpha = bp["log_alpha"]

        # ── Stage 2: L-BFGS-B local refinement (continuous params only) ───────
        # The kernel type is categorical so we fix it at the Optuna winner and
        # refine all continuous parameters with a gradient-based method.
        def _obj_continuous(vec):
            lv, llm, rl, ln, la = vec
            return self._eval_lml(
                vec[0],    # angle (first element, re-used below)
                best_kernel_type,
                lv, llm, rl, ln, la,
            )

        def _obj_with_angle(vec):
            ang, lv, llm, rl, ln, la = vec
            return self._eval_lml(ang, best_kernel_type, lv, llm, rl, ln, la)

        x0 = np.array([
            best_angle, best_log_var, best_log_lmaj,
            best_ratio, best_log_nug, best_log_alpha,
        ])
        lbfgs_bounds = [
            self.angle_bounds,
            (log_var_min,   log_var_max),
            (log_ls_min,    log_ls_max),
            (0.0,           max_ratio_log if max_ratio_log > 0 else 1e-6),
            (log_nug_min,   log_nug_max),
            (log_alpha_min, log_alpha_max),
        ]
        try:
            res = sp_minimize(
                _obj_with_angle, x0,
                method="L-BFGS-B",
                bounds=lbfgs_bounds,
                options={"ftol": 1e-12, "maxiter": 400},
            )
            if res.fun < study.best_value:
                (best_angle, best_log_var, best_log_lmaj,
                 best_ratio, best_log_nug, best_log_alpha) = res.x
        except Exception:
            pass  # keep Optuna result if L-BFGS-B fails

        # ── Stage 3: Build final kernel and fit ───────────────────────────────
        final_var    = np.exp(best_log_var)
        final_ls_maj = np.exp(best_log_lmaj)
        final_ls_min = np.exp(best_log_lmaj - best_ratio)
        final_nugget = np.exp(best_log_nug)
        final_alpha  = float(np.exp(best_log_alpha))

        base_k = build_base_kernel(best_kernel_type, final_ls_maj, self.ls_bounds)
        base_k.length_scale = [final_ls_maj, final_ls_min]

        final_kernel = ConstantKernel(
            constant_value=final_var,
            constant_value_bounds=self.var_bounds,
        ) * base_k + WhiteKernel(
            noise_level=final_nugget,
            noise_level_bounds=self.nugget_bounds,
        )

        self.best_angle_deg_   = float(best_angle)
        self.best_kernel_type_ = best_kernel_type
        self.best_alpha_       = final_alpha

        X_final = self._rotate_coords(self.X_train_centered_, self.best_angle_deg_)
        self.gp_model_ = GaussianProcessRegressor(
            kernel=final_kernel,
            alpha=final_alpha,
            optimizer=None,
            normalize_y=True,
            random_state=self.random_state,
        )
        self.gp_model_.fit(X_final, self.y_train_)
        self.log_marginal_likelihood_ = self.gp_model_.log_marginal_likelihood()

        # ── Post-fit verification ─────────────────────────────────────────────
        self._post_fit_report(final_ls_maj, final_ls_min, final_nugget)

        return self

    def _post_fit_report(
        self,
        ls_maj:  float,
        ls_min:  float,
        nugget:  float,
    ) -> None:
        """Print diagnostics and issue warnings if model quality is suspect."""

        actual_ratio = ls_maj / max(ls_min, 1e-12)

        # Warn if anisotropy ratio drifted outside the allowed bound
        if self.max_anisotropy and actual_ratio > self.max_anisotropy * 1.05:
            warnings.warn(
                f"RotatedGPR: final anisotropy ratio ({actual_ratio:.2f}) exceeds "
                f"max_anisotropy ({self.max_anisotropy}) by >5%%. "
                f"L-BFGS-B may have drifted outside bounds. "
                f"Consider increasing n_optuna_trials.",
                UserWarning, stacklevel=3,
            )

        # Warn if alpha is still high (noise-dominated)
        if self.best_alpha_ > 0.05:
            warnings.warn(
                f"RotatedGPR: fitted jitter alpha={self.best_alpha_:.5f} > 0.05. "
                f"The model may be noise-dominated. Check for outliers.",
                UserWarning, stacklevel=3,
            )

        # Console report
        print(f"       [GP post-fit] kernel type                : {self.best_kernel_type_}")
        print(f"       [GP post-fit] log marginal likelihood     : {self.log_marginal_likelihood_:.4f}")
        print(f"       [GP post-fit] rotation angle              : {self.best_angle_deg_:.2f} deg")
        print(f"       [GP post-fit] length scales (maj / min)   : {ls_maj:.2f} / {ls_min:.2f}")
        print(f"       [GP post-fit] anisotropy ratio            : {actual_ratio:.3f}")
        print(f"       [GP post-fit] nugget variance             : {nugget:.6f}")
        print(f"       [GP post-fit] jitter alpha                : {self.best_alpha_:.6f}")

    # ── predict ───────────────────────────────────────────────────────────────

    def predict(
        self,
        X:          np.ndarray,
        return_std: bool = False,
        return_cov: bool = False,
    ):
        X = np.asarray(X, dtype=np.float64)
        X_c   = X - self.X_center_ if self.center_coords else X
        X_rot = self._rotate_coords(X_c, self.best_angle_deg_)
        return self.gp_model_.predict(X_rot, return_std=return_std, return_cov=return_cov)

    # ── kernel property ───────────────────────────────────────────────────────

    @property
    def kernel_(self):
        return self.gp_model_.kernel_

    # ── parameter reporting ───────────────────────────────────────────────────

    def get_kernel_params(self) -> Dict[str, Any]:
        """
        Extract interpretable parameters from the fitted composite kernel.

        The kernel structure is:
            ConstantKernel * (Matérn or RBF) + WhiteKernel

        sklearn stores these after fitting under kernel_.k1 (product) and
        kernel_.k2 (WhiteKernel).  Within k1: k1.k1 = ConstantKernel,
        k1.k2 = spatial kernel.
        """
        k = self.gp_model_.kernel_   # fitted kernel

        # ── Extract constant (signal variance) ────────────────────────────────
        try:
            constant_value = float(k.k1.k1.constant_value)
        except AttributeError:
            constant_value = 1.0

        # ── Extract length scales from the spatial kernel (k1.k2) ─────────────
        try:
            ls = np.atleast_1d(k.k1.k2.length_scale).astype(np.float64)
        except AttributeError:
            ls = np.array([1.0, 1.0])

        if len(ls) < 2:
            ls = np.array([ls[0], ls[0]])

        # ── Extract nugget variance ────────────────────────────────────────────
        try:
            nugget = float(k.k2.noise_level)
        except AttributeError:
            nugget = float(self.best_alpha_)

        anisotropy_ratio = float(np.max(ls) / max(np.min(ls), 1e-12))

        return {
            "kernel_type":           self.best_kernel_type_,
            "rotation_angle_deg":    float(self.best_angle_deg_),
            "constant_value":        constant_value,
            "length_scale":          ls.tolist(),
            "nugget_variance":       nugget,
            "jitter_alpha":          float(self.best_alpha_),
            "anisotropy_ratio":      anisotropy_ratio,
            "log_marginal_likelihood": float(self.log_marginal_likelihood_),
        }
