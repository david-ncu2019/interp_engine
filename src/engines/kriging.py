"""
kriging.py - Ordinary Kriging with deterministic variogram-based parameter optimization

Supports two optimization paths:
  1. fit_deterministic() — ArcMap-style: user selects model + n_lags, program finds
     the unique optimal parameters via composite objective (WLS + CV + SSPE).
  2. fit() — Legacy Optuna TPE search across all models (backward compatibility).
"""
from typing import Optional, Dict, Any
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cluster import KMeans
from scipy.optimize import differential_evolution, minimize as sp_minimize, least_squares
import optuna

# Cap on the number of points used to ESTIMATE the empirical variogram. The variogram
# is a statistical estimate; a few thousand points capture its shape, while the full
# pairwise computation is O(N^2) in time and memory. Above this, deterministically
# subsample. The final OrdinaryKriging model is always built on ALL points.
VARIO_MAX_POINTS = 2000


# ─────────────────────────────────────────────────────────────────────────────
# Custom variogram functions for PyKrige (params, dists) interface
# ─────────────────────────────────────────────────────────────────────────────

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

def matern32_variogram_model(params, dists):
    psill, r, nugget = params
    u = np.sqrt(3.0) * dists / r
    return psill * (1.0 - (1.0 + u) * np.exp(-u)) + nugget

def matern52_variogram_model(params, dists):
    psill, r, nugget = params
    u = np.sqrt(5.0) * dists / r
    return psill * (1.0 - (1.0 + u + u**2 / 3.0) * np.exp(-u)) + nugget

def linear_variogram_model(params, dists):
    psill, r, nugget = params
    return nugget + np.minimum(psill * dists / r, psill)

def power_variogram_model(params, dists):
    psill, r, nugget = params
    return nugget + psill * (dists / r)


# ─────────────────────────────────────────────────────────────────────────────
# Variogram evaluators for WLS fitting — γ(h) given (h, psill, range, nugget)
# These duplicate the math from ui/variogram_panel.py to keep engine independent.
# ─────────────────────────────────────────────────────────────────────────────

def _eval_spherical(h, psill, r, nugget):
    h = np.asarray(h, dtype=float)
    g = np.where(h <= r,
                 nugget + psill * (1.5 * (h / r) - 0.5 * (h / r) ** 3),
                 nugget + psill)
    return np.where(h == 0, 0.0, g)

def _eval_exponential(h, psill, r, nugget):
    h = np.asarray(h, dtype=float)
    return np.where(h == 0, 0.0, nugget + psill * (1.0 - np.exp(-h / r)))

def _eval_gaussian(h, psill, r, nugget):
    h = np.asarray(h, dtype=float)
    return np.where(h == 0, 0.0, nugget + psill * (1.0 - np.exp(-(h / r) ** 2)))

def _eval_matern32(h, psill, r, nugget):
    h = np.asarray(h, dtype=float)
    u = np.sqrt(3.0) * h / r
    return np.where(h == 0, 0.0, nugget + psill * (1.0 - (1.0 + u) * np.exp(-u)))

def _eval_matern52(h, psill, r, nugget):
    h = np.asarray(h, dtype=float)
    u = np.sqrt(5.0) * h / r
    return np.where(h == 0, 0.0, nugget + psill * (1.0 - (1.0 + u + u**2 / 3.0) * np.exp(-u)))

def _eval_linear(h, psill, r, nugget):
    h = np.asarray(h, dtype=float)
    return np.where(h == 0, 0.0, nugget + np.minimum(psill * h / r, psill))

def _eval_power(h, psill, r, nugget):
    h = np.asarray(h, dtype=float)
    return np.where(h == 0, 0.0, nugget + psill * (h / r))

def _eval_hole_effect(h, psill, r, nugget):
    h = np.asarray(h, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        g = nugget + psill * (1.0 - np.sinc(h / r))
    return np.where(h == 0, 0.0, g)

def _eval_stable(h, psill, r, nugget, alpha=1.5):
    h = np.asarray(h, dtype=float)
    return np.where(h == 0, 0.0, nugget + psill * (1.0 - np.exp(-(h / r) ** alpha)))

def _eval_circular(h, psill, r, nugget):
    h = np.asarray(h, dtype=float)
    mask = h < r
    g = np.full_like(h, nugget + psill)
    hr = np.where(mask, h / r, 1.0)
    g[mask] = nugget + psill * (1.0 - (2.0 / np.pi) *
              (np.arccos(hr[mask]) - hr[mask] * np.sqrt(1.0 - hr[mask] ** 2)))
    return np.where(h == 0, 0.0, g)

def _eval_rational_quadratic(h, psill, r, nugget, alpha=1.0):
    h = np.asarray(h, dtype=float)
    return np.where(h == 0, 0.0,
                    nugget + psill * (1.0 - (1.0 + h**2 / (2.0 * alpha * r**2)) ** (-alpha)))

VARIOGRAM_EVALUATORS = {
    'spherical':          _eval_spherical,
    'exponential':        _eval_exponential,
    'gaussian':           _eval_gaussian,
    'matern_32':          _eval_matern32,
    'matern_52':          _eval_matern52,
    'linear':             _eval_linear,
    'power':              _eval_power,
    'hole-effect':        _eval_hole_effect,
    'stable':             _eval_stable,
    'circular':           _eval_circular,
    'rational-quadratic': _eval_rational_quadratic,
}

HAS_ALPHA = {'stable', 'rational-quadratic'}

# Models whose γ(h) grows without bound — use equal weights instead of Cressie weights
_UNBOUNDED_MODELS = {'linear', 'power'}

# Effective-range → mathematical-range correction factors per model
_RANGE_CORRECTION = {
    'spherical':          1.0,
    'exponential':        1.0 / 3.0,
    'gaussian':           1.0 / np.sqrt(3.0),
    'matern_32':          1.0 / np.sqrt(3.0),
    'matern_52':          np.sqrt(3.0) / np.sqrt(5.0),
    'linear':             1.0,
    'power':              1.0,
    'hole-effect':        1.0,
    'stable':             1.0,
    'circular':           1.0,
    'rational-quadratic': 1.0,
}


def _estimate_initial_params(emp_lags, emp_sv, emp_npairs, model_name):
    """Data-driven initial variogram parameter estimates from empirical variogram."""
    # Nugget: extrapolate first two bins back to h=0
    if len(emp_lags) >= 2 and emp_lags[1] > emp_lags[0]:
        slope = (emp_sv[1] - emp_sv[0]) / (emp_lags[1] - emp_lags[0])
        nugget0 = max(0.0, float(emp_sv[0] - slope * emp_lags[0]))
    else:
        nugget0 = 0.0

    # Sill: weighted mean of last 3 bins (plateau region)
    n_tail = min(3, len(emp_sv))
    w_tail = emp_npairs[-n_tail:]
    w_sum = w_tail.sum()
    sill_est = float(np.average(emp_sv[-n_tail:], weights=w_tail)) if w_sum > 0 else float(emp_sv[-1])
    psill0 = max(sill_est - nugget0, 1e-6)

    # Range: lag where γ(h) reaches ~95% of sill
    target = nugget0 + 0.95 * psill0
    above = np.where(emp_sv >= target)[0]
    if len(above) > 0:
        range0 = float(emp_lags[above[0]])
    else:
        range0 = float(emp_lags[-1]) * 0.8

    # Apply model-specific correction
    range0 *= _RANGE_CORRECTION.get(model_name, 1.0)
    range0 = max(range0, float(emp_lags[0]) * 0.5)

    return psill0, range0, nugget0


class AnisotropicKriging(BaseEstimator, RegressorMixin):
    """
    Ordinary Kriging with anisotropic parameter optimization.

    Two optimization paths:
      - fit_deterministic(): Deterministic composite-objective optimizer
        (WLS variogram fit + CV RMSE + SSPE calibration).
      - fit(): Legacy Optuna TPE stochastic search (backward compatibility).
    """

    NATIVE_MODELS = ['gaussian', 'spherical', 'exponential', 'hole-effect']
    CUSTOM_MODELS = {
        'stable': stable_variogram_model,
        'circular': circular_variogram_model,
        'rational-quadratic': rational_quadratic_variogram_model,
        'matern_32': matern32_variogram_model,
        'matern_52': matern52_variogram_model,
        'linear': linear_variogram_model,
        'power': power_variogram_model,
    }

    _MIN_TRIALS_RECOMMENDED = 100

    def __init__(
        self,
        n_trials: int = 150,
        n_splits: int = 5,
        verbose: bool = False,
        random_state: Optional[int] = None,
        max_anisotropy: float = 3.0,
        n_jobs: int = 1,
        n_lags: int = 12,
    ):
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.verbose = verbose
        self.random_state = random_state
        self.max_anisotropy = max_anisotropy
        self.n_jobs = n_jobs
        self.n_lags = n_lags

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

    # ─────────────────────────────────────────────────────────────────────────
    # NEW: Deterministic ArcMap-style optimizer
    # ─────────────────────────────────────────────────────────────────────────

    def fit_deterministic(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str,
        n_lags: int = 12,
        max_lag: Optional[float] = None,
        lag_width: Optional[float] = None,
        lag_tolerance: Optional[float] = None,
        n_folds: int = 5,
        wls_weight: float = 0.4,
        cv_weight: float = 0.2,
        sspe_weight: float = 0.4,
        lock_n_lags: bool = True,
        lock_max_lag: bool = True,
        compute_cv: bool = False,
    ) -> 'AnisotropicKriging':
        """Fast deterministic variogram optimizer with lag-parameter search.

        Stage 1: Data-driven initial estimates from empirical variogram.
        Stage 2: WLS fit via multi-start least_squares (psill, range, nugget, [alpha]).
        Stage 3: Directional anisotropy from 4 directional variograms.
        Stage 4: Optional CV for quality reporting (off by default), final model build.

        When lock_n_lags or lock_max_lag is False, runs an outer search over the
        unlocked lag parameter(s) using a composite objective that balances
        variogram fit (WLS) against uncertainty calibration (|RMSS - 1|).

        max_lag=None means auto (0.5 * max_pairwise_distance).

        Performance:
          - The empirical variogram is estimated from at most VARIO_MAX_POINTS
            (deterministic subsample) so the O(N^2) cost stays bounded for large N.
          - Stage-4 CV (the only O(N^3) work) is skipped unless compute_cv=True.
        Target: <2 seconds with locks; ~5-15s with lag search. Fully deterministic:
        same data + model → same result.
        """
        import time as _time
        from utils import compute_empirical_variogram, make_spatial_block_folds

        t0 = _time.perf_counter()

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        data_var = max(float(np.var(y)), 1e-12)
        max_dist = float(np.sqrt((X[:, 0].max() - X[:, 0].min())**2 +
                                  (X[:, 1].max() - X[:, 1].min())**2))

        # Deterministically subsample the points used to ESTIMATE the variogram when
        # N is large — keeps the O(N^2) empirical-variogram cost (and the directional
        # variant in Stage 3) bounded. The final model is still built on all points.
        if len(X) > VARIO_MAX_POINTS:
            rng = np.random.default_rng(0)
            sub = rng.choice(len(X), size=VARIO_MAX_POINTS, replace=False)
            sub.sort()
            X_emp, y_emp = X[sub], y[sub]
        else:
            X_emp, y_emp = X, y
        max_lag_str = f", max_lag={max_lag:.1f}" if max_lag else ""
        nl_str = f"{n_lags}{max_lag_str}"
        print(f"       [fast-opt] Model: {model_name}, n_lags: {nl_str}, "
              f"N={len(y)}"
              + (f" (variogram from {VARIO_MAX_POINTS}-pt subsample)"
                 if len(X) > VARIO_MAX_POINTS else ""),
              flush=True)

        # Precompute the N×N pairwise data ONCE — reused by every empirical-
        # variogram call during the lag search + Stage-1 + Stage-3 (directional).
        from scipy.spatial.distance import pdist, squareform
        _dists_cache = squareform(pdist(X_emp))
        _diff_sq_cache = np.subtract.outer(y_emp, y_emp) ** 2

        # ── Lag parameter search (when unlocked) ────────────────────────────
        if not lock_n_lags or not lock_max_lag:
            from utils import make_spatial_block_folds as _msbf

            # Candidate grid
            nl_candidates = [6, 8, 10, 12, 15, 20, 25, 30, 40] if not lock_n_lags else [n_lags]
            max_lag_auto = 0.5 * max_dist
            if not lock_max_lag:
                ml_candidates = np.linspace(
                    max_lag_auto * 0.3, max_lag_auto * 1.5, 8)
            else:
                ml_candidates = [max_lag_auto if max_lag is None else max_lag]

            best_score = float('inf')
            best_nl, best_ml = n_lags, (max_lag_auto if max_lag is None else max_lag)
            fold_ids = _msbf(X, n_folds) if n_folds > 1 else np.zeros(len(X), dtype=int)

            print(f"       [fast-opt] Searching lag params: "
                  f"n_lags={'unlocked' if not lock_n_lags else 'locked='+str(n_lags)}, "
                  f"max_lag={'unlocked' if not lock_max_lag else 'locked'}",
                  flush=True)

            for nl in nl_candidates:
                for ml in ml_candidates:
                    ml_val = float(ml)
                    # Compute empirical variogram with candidate params
                    emp_c = compute_empirical_variogram(
                        X_emp, y_emp, n_lags=int(nl),
                        max_lag=ml_val if not lock_max_lag else None,
                        lag_width=lag_width if lock_max_lag else None,
                        lag_tolerance=lag_tolerance,
                        _dists=_dists_cache, _diff_sq=_diff_sq_cache)
                    lags_c = emp_c['lags']
                    sv_c = emp_c['semivariance']
                    npairs_c = emp_c['n_pairs']
                    vld = npairs_c > 0
                    if vld.sum() < 3:
                        continue

                    # WLS fit
                    try:
                        wls_res = self._fit_wls_inner(
                            lags_c[vld], sv_c[vld], npairs_c[vld],
                            model_name, data_var, max_dist)
                        if wls_res is None:
                            continue
                    except Exception:
                        continue

                    # Quick inline CV — compute RMSS
                    rmss_val = self._quick_cv_rmss(
                        X, y, model_name, wls_res, fold_ids, n_folds)
                    if rmss_val is None:
                        continue

                    # Composite score
                    wls_err = wls_res.get('cost_norm', 1.0)
                    sspe_err = abs(rmss_val - 1.0)
                    score = wls_weight * wls_err + sspe_weight * sspe_err

                    if score < best_score:
                        best_score = score
                        best_nl = int(nl)
                        best_ml = ml_val
                        best_wls = wls_res

            n_lags = best_nl
            if not lock_max_lag:
                max_lag = best_ml
            print(f"       [fast-opt] Lag search done: n_lags={best_nl}, "
                  f"max_lag={best_ml:.1f}, score={best_score:.4f}", flush=True)

        # ── Stage 1: Empirical variogram + initial estimates ─────────────────
        # When lag distance is unlocked the search has chosen max_lag; do NOT also pass
        # the stale caller lag_width / lag_tolerance, or utils' GSLIB branch overrides
        # max_lag back to (n_lags × stale_lag_width). Mirrors the lock_max_lag gating
        # the search loop already uses for these same args.
        emp = compute_empirical_variogram(
            X_emp, y_emp,
            n_lags=n_lags,
            max_lag=max_lag if max_lag else None,
            lag_width=lag_width if lock_max_lag else None,
            lag_tolerance=lag_tolerance if lock_max_lag else None,
            _dists=_dists_cache, _diff_sq=_diff_sq_cache,
        )
        emp_lags = emp['lags']
        emp_sv = emp['semivariance']
        emp_npairs = emp['n_pairs']

        # Persist the binning actually used (searched / manual / auto) so it can be
        # surfaced back to the UI instead of leaving the controls reading "auto".
        self.n_lags = int(n_lags)
        self.lag_width_ = float(emp.get('lag_width'))
        self.max_lag_ = float(emp.get('max_lag'))
        self.lag_tolerance_ = float(emp.get('lag_tolerance'))
        print(f"       [fast-opt] Lag binning used: n_lags={self.n_lags}, "
              f"lag_distance={self.lag_width_:.1f}, tolerance={self.lag_tolerance_:.1f}",
              flush=True)

        valid = emp_npairs > 0
        emp_lags = emp_lags[valid]
        emp_sv = emp_sv[valid]
        emp_npairs = emp_npairs[valid].astype(float)

        if len(emp_lags) < 3:
            raise ValueError(
                f"Only {len(emp_lags)} non-empty variogram bins — need at least 3. "
                f"Try fewer lags or check data spacing."
            )

        psill0, range0, nugget0 = _estimate_initial_params(
            emp_lags, emp_sv, emp_npairs, model_name
        )
        print(f"       [fast-opt] Initial: psill={psill0:.4f}, "
              f"range={range0:.1f}, nugget={nugget0:.4f}", flush=True)

        # ── Stage 2: WLS-only DE on 3D/4D ───────────────────────────────────
        evaluator = VARIOGRAM_EVALUATORS[model_name]
        has_alpha = model_name in HAS_ALPHA
        use_cressie = model_name not in _UNBOUNDED_MODELS

        # Bound the search by what the empirical variogram can actually support:
        # range can't exceed the observable window; psill can't exceed the observed
        # max. This kills the Cressie-weighting degenerate basin where range ≫
        # max(lag) "buys" tiny residuals at the origin via 1/γ² weighting.
        range_upper = min(max_dist * 1.5, float(emp_lags[-1]) * 1.5)
        psill_upper = max(float(data_var), float(np.max(emp_sv))) * 2.0
        bounds_wls = [
            (data_var * 0.01, psill_upper),       # psill
            (max_dist * 0.02, range_upper),       # range
            (0.0,             data_var * 1.0),    # nugget
        ]
        if has_alpha:
            bounds_wls.append((0.1, 2.0))          # alpha

        def _gamma(params_vec):
            psill, v_range, nugget = params_vec[0], max(params_vec[1], 1e-6), params_vec[2]
            alpha = params_vec[3] if has_alpha else None
            if alpha is not None:
                gm = evaluator(emp_lags, max(psill, 1e-12), v_range, nugget, alpha)
            else:
                gm = evaluator(emp_lags, max(psill, 1e-12), v_range, nugget)
            return np.maximum(gm, 1e-12)

        def wls_residuals(params_vec):
            """Weighted residual VECTOR for least_squares (Cressie weighting)."""
            gamma_model = _gamma(params_vec)
            if use_cressie:
                weights = emp_npairs / (gamma_model ** 2)
            else:
                weights = emp_npairs
            return np.sqrt(weights) * (emp_sv - gamma_model)

        def wls_objective(params_vec):
            """Scalar WLS cost = sum of squared weighted residuals."""
            return float(np.sum(wls_residuals(params_vec) ** 2))

        print("       [fast-opt] Stage 2: WLS multi-start least_squares ...", flush=True)
        t1 = _time.perf_counter()

        lower = np.array([b[0] for b in bounds_wls], dtype=float)
        upper = np.array([b[1] for b in bounds_wls], dtype=float)

        # Deterministic multi-start set around the data-driven initial estimate.
        # Each fit on ~n_lags points is sub-millisecond, so a small grid is free.
        starts = []
        for rf in (0.5, 1.0, 2.0):
            for pf in (0.5, 1.0, 1.5):
                cand = [psill0 * pf, range0 * rf, nugget0]
                if has_alpha:
                    cand.append(1.0)
                starts.append(np.clip(np.asarray(cand, float), lower, upper))

        best_wls_x = None
        best_fun = np.inf
        for s in starts:
            try:
                res = least_squares(
                    wls_residuals, x0=s, bounds=(lower, upper),
                    method='trf', xtol=1e-12, ftol=1e-12, max_nfev=400,
                )
                f = wls_objective(res.x)
                if f < best_fun:
                    best_fun, best_wls_x = f, res.x
            except Exception:
                continue

        # Fallback: differential evolution if every least_squares start failed.
        if best_wls_x is None:
            print("       [fast-opt] least_squares failed — falling back to DE.", flush=True)
            try:
                result_de = differential_evolution(
                    wls_objective, bounds=bounds_wls,
                    seed=0, tol=1e-6, maxiter=200, polish=True, init='sobol',
                )
            except TypeError:
                result_de = differential_evolution(
                    wls_objective, bounds=bounds_wls,
                    seed=0, tol=1e-6, maxiter=200, polish=True, init='latinhypercube',
                )
            best_wls_x = result_de.x

        opt_psill = float(best_wls_x[0])
        opt_range = float(best_wls_x[1])
        opt_nugget = float(best_wls_x[2])
        opt_alpha = float(best_wls_x[3]) if has_alpha else None

        t2 = _time.perf_counter()
        print(f"       [fast-opt] WLS done in {t2 - t1:.2f}s: psill={opt_psill:.4f}, "
              f"range={opt_range:.1f}, nugget={opt_nugget:.4f}"
              + (f", alpha={opt_alpha:.3f}" if opt_alpha is not None else ""),
              flush=True)

        # ── Stage 3: Directional anisotropy estimation ───────────────────────
        opt_angle = 0.0
        opt_scaling = 1.0

        skip_aniso = len(X_emp) < 100
        if not skip_aniso:
            print("       [fast-opt] Stage 3: Directional anisotropy ...", flush=True)
            t3 = _time.perf_counter()

            dir_result = compute_empirical_variogram(
                X_emp, y_emp, n_lags=n_lags, directions=[0, 45, 90, 135],
                _dists=_dists_cache, _diff_sq=_diff_sq_cache,
            )

            dir_ranges = {}
            too_few_pairs = False

            for dv in dir_result:
                d_lags = dv['lags']
                d_sv = dv['semivariance']
                d_np = dv['n_pairs']
                d_valid = d_np > 0
                d_lags = d_lags[d_valid]
                d_sv = d_sv[d_valid]
                d_np = d_np[d_valid].astype(float)

                if len(d_lags) < 3 or d_np.mean() < 30:
                    too_few_pairs = True
                    break

                # Fit WLS to this direction using L-BFGS-B from isotropic solution
                def _dir_wls(pv, _lags=d_lags, _sv=d_sv, _np=d_np):
                    ps, rg, ng = pv[0], max(pv[1], 1e-6), pv[2]
                    al = pv[3] if has_alpha else None
                    if al is not None:
                        gm = evaluator(_lags, max(ps, 1e-12), rg, ng, al)
                    else:
                        gm = evaluator(_lags, max(ps, 1e-12), rg, ng)
                    gm = np.maximum(gm, 1e-12)
                    w = _np / (gm ** 2) if use_cressie else _np
                    return float(np.sum(w * (_sv - gm) ** 2))

                x0_dir = list(best_wls_x)
                try:
                    res = sp_minimize(
                        _dir_wls, x0=x0_dir,
                        method='L-BFGS-B', bounds=bounds_wls,
                        options={'ftol': 1e-10, 'maxiter': 200},
                    )
                    dir_ranges[dv['direction']] = float(res.x[1])
                except Exception:
                    dir_ranges[dv['direction']] = opt_range

            if too_few_pairs or len(dir_ranges) < 4:
                print("       [fast-opt] Anisotropy: too few pairs, using isotropic",
                      flush=True)
            else:
                dirs = sorted(dir_ranges.keys())
                ranges = np.array([dir_ranges[d] for d in dirs])
                idx_max = int(np.argmax(ranges))
                idx_min = int(np.argmin(ranges))
                r_major = ranges[idx_max]
                r_minor = max(ranges[idx_min], 1e-6)

                opt_angle = float(dirs[idx_max])
                opt_scaling = min(r_major / r_minor, self.max_anisotropy)
                if opt_scaling < 1.05:
                    opt_angle = 0.0
                    opt_scaling = 1.0

                t4 = _time.perf_counter()
                print(f"       [fast-opt] Anisotropy done in {t4 - t3:.2f}s: "
                      f"angle={opt_angle:.1f}°, scaling={opt_scaling:.2f}", flush=True)
                print(f"       [fast-opt] Directional ranges: "
                      + ", ".join(f"{d}°={r:.1f}" for d, r in sorted(dir_ranges.items())),
                      flush=True)
        else:
            print(f"       [fast-opt] N={len(X_emp)} < 100, skipping anisotropy",
                  flush=True)

        # ── Stage 4: Assemble params + single CV + final model ───────────────
        self.best_model_name_ = model_name
        self.best_params_ = {
            'psill':   opt_psill,
            'range':   opt_range,
            'nugget':  opt_nugget,
            'angle':   opt_angle,
            'scaling': opt_scaling,
        }
        if has_alpha:
            self.best_params_['alpha'] = opt_alpha

        # Optional CV check for quality reporting. This is the only O(N^3) work in the
        # pipeline (it builds + solves a kriging system per fold), and it is NOT needed
        # to fit the empirical variogram — so it is off by default. When enabled, the
        # folds are independent and run in parallel across all cores via joblib.
        if compute_cv:
            print("       [fast-opt] Stage 4: CV check (parallel) ...", flush=True)
            t5 = _time.perf_counter()

            fold_ids = make_spatial_block_folds(X, n_folds)

            def _run_fold(fold):
                tr_idx = np.where(fold_ids != fold)[0]
                te_idx = np.where(fold_ids == fold)[0]
                if len(tr_idx) < 5 or len(te_idx) < 1:
                    return None
                try:
                    ok = self._get_ok_instance(
                        X[tr_idx], y[tr_idx], self.best_params_, model_name
                    )
                    y_pred, y_var = ok.execute('points', X[te_idx, 0], X[te_idx, 1])
                    y_pred = np.asarray(y_pred, dtype=np.float64)
                    y_var = np.asarray(y_var, dtype=np.float64)
                    residuals = y[te_idx] - y_pred
                    std = np.sqrt(np.abs(y_var))
                    z_sq = (residuals / (std + 1e-12)) ** 2
                    return (residuals ** 2).tolist(), z_sq.tolist()
                except Exception:
                    return None

            from joblib import Parallel, delayed
            n_jobs = self.n_jobs if self.n_jobs and self.n_jobs != 1 else -1
            fold_out = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_run_fold)(fold) for fold in range(n_folds)
            )

            cv_errors_sq, sspe_values = [], []
            for item in fold_out:
                if item is not None:
                    cv_errors_sq.extend(item[0])
                    sspe_values.extend(item[1])

            t6 = _time.perf_counter()
            if cv_errors_sq:
                cv_rmse = float(np.sqrt(np.mean(cv_errors_sq)))
                mean_sspe = float(np.mean(sspe_values)) if sspe_values else float('nan')
                print(f"       [fast-opt] CV done in {t6 - t5:.2f}s: "
                      f"RMSE={cv_rmse:.4f}, mean_SSPE={mean_sspe:.3f}", flush=True)
            else:
                print(f"       [fast-opt] CV: no valid folds", flush=True)
        else:
            print("       [fast-opt] Stage 4: CV skipped (compute_cv=False).", flush=True)

        # Build final model on all data
        self.model_ = self._get_ok_instance(X, y, self.best_params_, model_name)

        t_total = _time.perf_counter() - t0
        print(f"       [fast-opt] TOTAL: {t_total:.2f}s", flush=True)
        print(f"       [fast-opt] psill={opt_psill:.4f}, "
              f"range={opt_range:.1f}, nugget={opt_nugget:.4f}, "
              f"angle={opt_angle:.1f}°, ratio={opt_scaling:.2f}"
              + (f", alpha={opt_alpha:.3f}" if opt_alpha is not None else ""),
              flush=True)

        return self

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers for lag-parameter search (used by fit_deterministic)
    # ─────────────────────────────────────────────────────────────────────────

    def _fit_wls_inner(self, lags, sv, npairs, model_name, data_var, max_dist):
        """Core WLS fit for a single (n_lags, max_lag) candidate.
        Returns dict with fitted params and normalized cost, or None on failure."""
        import warnings
        from scipy.optimize import least_squares, differential_evolution

        model_fn = VARIOGRAM_EVALUATORS.get(model_name)
        if model_fn is None:
            return None
        has_alpha = model_name in HAS_ALPHA

        def _gamma(params_vec):
            # Match the evaluator's signature: 9 of 11 evaluators take 4 args (no alpha);
            # 2 (stable, rational-quadratic) take alpha as a 5th. Vectorized — every
            # evaluator already accepts array lags.
            if has_alpha:
                psill, range_, nug, alpha = params_vec
                return model_fn(lags, psill, range_, nug, alpha)
            psill, range_, nug = params_vec
            return model_fn(lags, psill, range_, nug)

        use_cressie = model_name not in ("linear", "power")

        # Initial estimates
        init_nug = float(np.interp(0, lags[:2], sv[:2])) if len(lags) >= 2 else 0.0
        init_nug = max(0.0, min(init_nug, data_var * 0.5))
        init_psill = max(float(np.mean(sv[-3:])) if len(sv) >= 3 else data_var,
                         data_var * 0.1) - init_nug
        init_psill = max(init_psill, data_var * 0.1)
        init_range = float(lags[-1] * 0.5)
        # Match Stage-2 bounds: range/psill capped by the observable window.
        range_upper = min(max_dist * 1.5, float(lags[-1]) * 1.5)
        psill_upper = max(float(data_var), float(np.max(sv))) * 2.0
        bounds_list = [(1e-6, psill_upper), (1.0, range_upper),
                       (1e-12, data_var * 2.0)]
        x0_list = [init_psill, init_range, init_nug]
        if has_alpha:
            bounds_list.append((0.1, 3.0))
            x0_list.append(1.0)

        def _wls_obj(p):
            g = _gamma(p)
            if use_cressie:
                w = npairs / np.maximum(g, 1e-12) ** 2
            else:
                w = npairs.astype(float)
            return float(np.sum(w * (sv - g) ** 2))

        best_cost = float('inf')
        best_params = None

        # Multi-start L-BFGS-B
        starts = [
            x0_list,
            [init_psill * 1.5, init_range * 0.7, init_nug * 2.0]
            + ([1.0] if has_alpha else []),
            [init_psill * 0.5, init_range * 1.3, init_nug * 0.5]
            + ([2.0] if has_alpha else []),
            [data_var, max_dist * 0.3, 0.0] + ([1.0] if has_alpha else []),
        ]
        for x0 in starts:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = least_squares(
                        lambda p: np.sqrt(npairs) * (sv - _gamma(p)),
                        x0, bounds=list(zip(*bounds_list)),
                        method='trf', max_nfev=200)
                cost = _wls_obj(res.x)
                if cost < best_cost:
                    best_cost = cost
                    best_params = res.x.copy()
            except Exception:
                continue

        if best_params is None:
            # Fallback to DE
            try:
                res = differential_evolution(
                    _wls_obj, bounds_list, seed=0, maxiter=100, polish=False)
                best_cost = _wls_obj(res.x)
                best_params = res.x
            except Exception:
                return None

        result = {
            'psill': float(best_params[0]),
            'range': float(best_params[1]),
            'nugget': float(best_params[2]),
            'cost': float(best_cost),
            'cost_norm': float(best_cost) / max(float(np.sum(sv ** 2)), 1e-12),
        }
        if has_alpha and len(best_params) >= 4:
            result['alpha'] = float(best_params[3])
        return result

    def _quick_cv_rmss(self, X, y, model_name, wls_params, fold_ids, n_folds):
        """Quick inline CV to compute RMSS for a candidate fit.
        Returns RMSS or None on failure.

        Uses fast_kriging.ok_predict (lu_factor + batched lu_solve) instead of
        PyKrige's `inv`-per-fold path. ~3-4x faster, byte-equivalent results.
        """
        try:
            psill = wls_params['psill']
            range_ = wls_params['range']
            nugget = wls_params['nugget']
            alpha = wls_params.get('alpha', 1.0)
        except KeyError:
            return None

        from src.engines.fast_kriging import ok_predict

        # Quick CV runs during the lag search before Stage 3 has fitted anisotropy,
        # so the predictor is isotropic. alpha is only consumed for stable/r-quad.
        params = {'psill': psill, 'range': range_, 'nugget': nugget,
                  'angle': 0.0, 'scaling': 1.0, 'alpha': alpha}

        z_scores = []
        for fold in range(n_folds):
            tr_idx = np.where(fold_ids != fold)[0]
            te_idx = np.where(fold_ids == fold)[0]
            if len(tr_idx) < 10 or len(te_idx) < 3:
                continue
            try:
                mean, std = ok_predict(
                    X[tr_idx], y[tr_idx], X[te_idx],
                    model_name, params, return_std=True)
                resid = y[te_idx] - mean
                z_sq = (resid / np.maximum(std, 1e-10)) ** 2
                z_scores.extend(z_sq.tolist())
            except Exception:
                continue

        if not z_scores:
            return None
        mean_sspe = float(np.mean(z_scores))
        return float(np.sqrt(mean_sspe))

    # ─────────────────────────────────────────────────────────────────────────
    # Legacy Optuna-based optimizer (backward compatibility)
    # ─────────────────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AnisotropicKriging':
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        if self.n_trials < self._MIN_TRIALS_RECOMMENDED:
            import warnings
            warnings.warn(
                f"AnisotropicKriging: n_trials={self.n_trials} is below the "
                f"recommended minimum of {self._MIN_TRIALS_RECOMMENDED}.",
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

        self.study_ = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )

        def logging_callback(study, trial):
            if self.verbose:
                val = trial.value
                val_str = f"{val:.2e}" if val != float('inf') else "inf"
                best_val = f"{study.best_value:.2e}"
                p = trial.params
                p_str = f"model={p['model']}, psill={p['psill']:.2e}, range={p['range']:.1f}, nugget={p['nugget']:.2e}"
                print(f"  [Trial {trial.number}] val={val_str}, best={best_val} | {p_str}", flush=True)

        self.study_.optimize(objective, n_trials=self.n_trials, callbacks=[logging_callback], n_jobs=self.n_jobs)

        _raw_best = self.study_.best_params
        self.best_model_name_ = _raw_best['model']
        self.best_params_ = {k: v for k, v in _raw_best.items() if k != 'model'}

        self.model_ = self._get_ok_instance(X, y, self.best_params_, self.best_model_name_)

        return self

    # ─────────────────────────────────────────────────────────────────────────
    # Preset (manual slider) fit
    # ─────────────────────────────────────────────────────────────────────────

    def fit_with_known_params(self, X: np.ndarray, y: np.ndarray, best_model_name: str, best_params: dict) -> 'AnisotropicKriging':
        """Fit the model instantly using pre-computed optimal parameters (skips optimization)."""
        X = np.asarray(X)
        y = np.asarray(y)

        self.best_model_name_ = best_model_name
        self.best_params_ = best_params

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
        # Report the lag binning actually used by the deterministic optimizer so the
        # UI can show concrete values instead of "auto". Guarded: the preset/Run path
        # (fit_with_known_params) never sets these, leaving the dict unchanged.
        if getattr(self, 'lag_width_', None) is not None:
            params['n_lags'] = int(self.n_lags)
            params['lag_width'] = float(self.lag_width_)
            params['lag_tolerance'] = float(self.lag_tolerance_)
        return params
