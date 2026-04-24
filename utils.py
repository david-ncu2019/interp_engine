"""
Diagnostics module for interpolation engine validation.
Provides empirical variogram, cross-validation, and publication-quality plots.
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.base import clone

warnings.filterwarnings("ignore")


# ========================================================================
# 1. EMPIRICAL VARIOGRAM
# ========================================================================

def auto_lag_params(X, min_pairs_per_bin=30, min_lags=6, max_lags=30):
    """
    Determine adaptive variogram lag parameters from data geometry.

    Strategy
    --------
    1. **lag_width** = 2 x median nearest-neighbour distance.
       This ensures the first lag bin captures short-range structure and
       contains enough pairs for a stable semivariance estimate.
    2. **max_lag**   = min(0.5 * max_pairwise_distance, extent that
       guarantees >= min_pairs_per_bin pairs on average per bin).
    3. **n_lags**    = max_lag / lag_width, clamped to [min_lags, max_lags].

    Parameters
    ----------
    X : (n, 2) array – spatial coordinates
    min_pairs_per_bin : int – target minimum pair count per lag bin
    min_lags : int – hard lower bound on lag count
    max_lags : int – hard upper bound on lag count

    Returns
    -------
    dict with keys: n_lags, lag_width, max_lag, max_dist, median_nn,
                    n_total_pairs, method_notes (str)
    """
    from scipy.spatial.distance import pdist, squareform

    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    n_total_pairs = n * (n - 1) // 2

    dists_sq = squareform(pdist(X))
    max_dist = dists_sq.max()

    # Nearest-neighbour distances (excluding self)
    np.fill_diagonal(dists_sq, np.inf)
    nn_dists = dists_sq.min(axis=1)
    median_nn = float(np.median(nn_dists))

    notes = []

    # --- lag_width: 2x median nearest-neighbour ---
    lag_width = 2.0 * median_nn
    # Safety floor: at least 1% of max distance
    if lag_width < max_dist * 0.01:
        lag_width = max_dist * 0.01
        notes.append("lag_width floored to 1% of max_dist")

    # --- max_lag: half of max pairwise distance ---
    max_lag = 0.5 * max_dist

    # --- n_lags from max_lag / lag_width, then clamp ---
    n_lags = int(np.round(max_lag / lag_width))
    if n_lags < min_lags:
        n_lags = min_lags
        notes.append(f"n_lags raised to min_lags={min_lags}")
    if n_lags > max_lags:
        n_lags = max_lags
        notes.append(f"n_lags capped at max_lags={max_lags}")

    # Adjust lag_width and max_lag for consistency
    lag_width = max_lag / n_lags

    # Check pair density: if avg pairs/bin is too low, reduce n_lags
    avg_pairs = n_total_pairs / n_lags
    while avg_pairs < min_pairs_per_bin and n_lags > min_lags:
        n_lags -= 1
        lag_width = max_lag / n_lags
        avg_pairs = n_total_pairs / n_lags
    if avg_pairs < min_pairs_per_bin:
        notes.append(f"WARNING: avg {avg_pairs:.0f} pairs/bin < target {min_pairs_per_bin}")

    return {
        "n_lags": n_lags,
        "lag_width": float(lag_width),
        "max_lag": float(max_lag),
        "max_dist": float(max_dist),
        "median_nn": float(median_nn),
        "n_total_pairs": n_total_pairs,
        "avg_pairs_per_bin": float(n_total_pairs / n_lags),
        "method_notes": "; ".join(notes) if notes else "all defaults applied",
    }


def compute_empirical_variogram(X, y, n_lags=None, max_lag=None,
                                lag_width=None, max_lag_frac=None,
                                directions=None, tol_angle=22.5):
    """
    Compute omnidirectional or directional empirical semivariogram.

    Lag parameters are **adaptive by default**.  If none of n_lags /
    max_lag / lag_width / max_lag_frac are given, the function calls
    ``auto_lag_params`` to choose values based on point density and
    domain extent.  You can override any subset; the rest will be
    inferred consistently.

    Parameters
    ----------
    X : (n, 2) array  -- spatial coordinates
    y : (n,) array    -- values
    n_lags : int or None
        Number of lag bins.  Auto-determined if None.
    max_lag : float or None
        Maximum lag distance (m).  Auto-determined if None.
    lag_width : float or None
        Width of each lag bin (m).  Auto-determined if None.
    max_lag_frac : float or None
        LEGACY fallback: max lag as fraction of max pairwise distance.
        Only used when max_lag is None AND auto mode is overridden by
        providing n_lags explicitly.
    directions : list of float or None
        If None  -> omnidirectional.
        If list  -> compute for each angle (degrees CCW from +X).
    tol_angle : float -- angular tolerance (degrees) for directional bins

    Returns
    -------
    dict with keys:
        lags, semivariance, n_pairs  (arrays, length n_lags)
        lag_width, max_lag           (scalars)
        auto_params                  (dict, from auto_lag_params)
    If directional, returns list of such dicts (one per direction).
    """
    from scipy.spatial.distance import pdist, squareform

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # ---- Resolve lag parameters adaptively ----
    ap = auto_lag_params(X)  # always compute for reporting

    if n_lags is None and max_lag is None and lag_width is None and max_lag_frac is None:
        # Full auto mode
        n_lags   = ap["n_lags"]
        max_lag  = ap["max_lag"]
        lag_width = ap["lag_width"]
    else:
        # Partial override
        dists_tmp = pdist(X)
        max_dist = dists_tmp.max()
        if max_lag is None:
            frac = max_lag_frac if max_lag_frac is not None else 0.5
            max_lag = max_dist * frac
        if n_lags is not None and lag_width is None:
            lag_width = max_lag / n_lags
        elif lag_width is not None and n_lags is None:
            n_lags = max(6, int(np.round(max_lag / lag_width)))
            lag_width = max_lag / n_lags   # reconcile
        elif n_lags is None and lag_width is None:
            n_lags   = ap["n_lags"]
            lag_width = max_lag / n_lags

    lag_edges = np.linspace(0, max_lag, n_lags + 1)
    lag_centers = 0.5 * (lag_edges[:-1] + lag_edges[1:])

    # Pairwise distances & squared differences
    dists = squareform(pdist(X))
    n = len(y)
    diff_sq = np.subtract.outer(y, y) ** 2

    def _compute_bins(angle_mask=None):
        semivar = np.zeros(n_lags)
        counts = np.zeros(n_lags, dtype=int)
        for k in range(n_lags):
            lo, hi = lag_edges[k], lag_edges[k + 1]
            mask = (dists > lo) & (dists <= hi)
            if angle_mask is not None:
                mask &= angle_mask
            np.fill_diagonal(mask, False)
            counts[k] = mask.sum() // 2
            if counts[k] > 0:
                semivar[k] = 0.5 * diff_sq[mask].sum() / mask.sum()
        return semivar, counts

    base_result = {
        "lag_width": lag_width,
        "max_lag": max_lag,
        "auto_params": ap,
    }

    if directions is None:
        sv, ct = _compute_bins()
        return {**base_result, "lags": lag_centers,
                "semivariance": sv, "n_pairs": ct}

    # Directional
    dx = X[:, 0][:, None] - X[:, 0][None, :]
    dy = X[:, 1][:, None] - X[:, 1][None, :]
    angles_deg = np.degrees(np.arctan2(dy, dx)) % 360

    results = []
    for d in directions:
        d_mod = d % 360
        d_opp = (d + 180) % 360
        ang_ok = (np.abs(angles_deg - d_mod) <= tol_angle) | \
                 (np.abs(angles_deg - d_opp) <= tol_angle) | \
                 (np.abs(angles_deg - d_mod + 360) <= tol_angle) | \
                 (np.abs(angles_deg - d_mod - 360) <= tol_angle)
        sv, ct = _compute_bins(angle_mask=ang_ok)
        results.append({
            **base_result, "direction": d,
            "lags": lag_centers, "semivariance": sv, "n_pairs": ct,
        })
    return results


def gaussian_model(h, psill, a, nugget):
    """Gaussian variogram model: nugget + psill*(1 - exp(-3*(h/a)^2))"""
    return nugget + psill * (1.0 - np.exp(-3.0 * (h / a) ** 2))


# ========================================================================
# 2. VARIOGRAM PLOT (publication quality)
# ========================================================================

def plot_variogram(vario_dict, true_params=None, fitted_params=None,
                   engine_name="", scenario_name="", save_path=None):
    """
    Publication-quality variogram plot with empirical points, pair counts,
    and optional fitted/true model curves.

    Parameters
    ----------
    vario_dict : dict from compute_empirical_variogram (omnidirectional)
    true_params : dict  {psill, range_major, nugget}   (ground truth)
    fitted_params : dict {psill, range, nugget}         (engine recovered)
    """
    lags = vario_dict["lags"]
    sv = vario_dict["semivariance"]
    np_ = vario_dict["n_pairs"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7),
                                    gridspec_kw={"height_ratios": [3, 1]},
                                    sharex=True)

    # -- upper: semivariance --
    valid = np_ > 0
    ax1.scatter(lags[valid], sv[valid], s=60, c="crimson", edgecolor="k",
                zorder=5, label="Empirical")

    h_fine = np.linspace(0, lags.max() * 1.1, 300)

    if true_params is not None:
        y_true = gaussian_model(h_fine, true_params["psill"],
                                true_params["range_major"],
                                true_params["nugget"])
        ax1.plot(h_fine, y_true, "b-", lw=2, label="True model")

    if fitted_params is not None:
        y_fit = gaussian_model(h_fine, fitted_params.get("psill", 1),
                               fitted_params.get("range", 200),
                               fitted_params.get("nugget", 0))
        ax1.plot(h_fine, y_fit, "g--", lw=2, label=f"Fitted ({engine_name})")

    ax1.set_ylabel("Semivariance", fontsize=12)
    ax1.set_title(f"Empirical Variogram  -  {scenario_name}  [{engine_name}]",
                  fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, ls="--", alpha=0.4)

    # lag info text
    ap = vario_dict.get("auto_params", {})
    info_lines = [
        f"Lags: {len(lags)}  |  Lag width: {vario_dict['lag_width']:.1f} m  "
        f"|  Max lag: {vario_dict['max_lag']:.0f} m",
    ]
    if ap:
        info_lines.append(
            f"Median NN: {ap.get('median_nn', 0):.1f} m  "
            f"|  Avg pairs/bin: {ap.get('avg_pairs_per_bin', 0):.0f}"
        )
    info = "\n".join(info_lines)
    ax1.text(0.02, 0.97, info, transform=ax1.transAxes, fontsize=8,
             va="top", bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    # -- lower: pair counts --
    ax2.bar(lags, np_, width=vario_dict["lag_width"] * 0.8,
            color="steelblue", edgecolor="k", alpha=0.7)
    ax2.set_xlabel("Lag Distance (m)", fontsize=12)
    ax2.set_ylabel("# Pairs", fontsize=12)
    ax2.grid(True, ls="--", alpha=0.4)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_directional_variogram(dir_vario_list, true_params=None,
                               scenario_name="", save_path=None):
    """Plot directional variograms on a single figure."""
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.tab10

    for i, vd in enumerate(dir_vario_list):
        valid = vd["n_pairs"] > 0
        ax.plot(vd["lags"][valid], vd["semivariance"][valid], "o-",
                color=cmap(i), ms=5, lw=1.2,
                label=f'{vd["direction"]:.0f} deg')

    if true_params is not None:
        h = np.linspace(0, dir_vario_list[0]["max_lag"], 200)
        y = gaussian_model(h, true_params["psill"],
                           true_params["range_major"], true_params["nugget"])
        ax.plot(h, y, "k--", lw=2, label="True (major axis)")

    ax.set_xlabel("Lag Distance (m)", fontsize=12)
    ax.set_ylabel("Semivariance", fontsize=12)
    ax.set_title(f"Directional Variograms  -  {scenario_name}",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


# ========================================================================
# 3. CROSS-VALIDATION (GPR & Kriging)
# ========================================================================

def make_spatial_block_folds(X: np.ndarray, n_folds: int) -> np.ndarray:
    """Assign spatial block fold labels to data points.

    Instead of KMeans clustering (which creates compact, spatially disjoint
    clusters that force the test set *outside* the correlation range of the
    training set), this function sorts points along a space-filling diagonal
    direction and divides them into N contiguous geographic strips.

    Each strip is entirely surrounded by training data from adjacent strips,
    so test points remain *within* the correlation range — true interpolation
    rather than extrapolation.

    Scientific rationale
    --------------------
    KMeans CV tends to underestimate prediction skill for spatially correlated
    data because test clusters are far from all training clusters (Wadoux et al.,
    2021, Environ. Model. Softw.). Spatial block CV gives a more honest estimate
    of interpolation performance.

    Args:
        X: (n, 2) array of spatial coordinates.
        n_folds: Number of CV folds (geographic strips).

    Returns:
        fold_ids: (n,) integer array in [0, n_folds-1] assigning each point
                  to a fold.
    """
    # Project onto 45° diagonal so both X and Y variation contribute equally
    # to the sort order.  Points with similar X+Y end up in the same strip.
    diagonal_proj = X[:, 0] + X[:, 1]
    sort_idx = np.argsort(diagonal_proj)

    fold_ids = np.empty(len(X), dtype=int)
    # Split sorted indices into n_folds roughly equal groups
    for fold, chunk in enumerate(np.array_split(sort_idx, n_folds)):
        fold_ids[chunk] = fold
    return fold_ids


def perform_gpr_kfold_cv(rgpr_model, X, y, n_folds=5, seed=42, nst=None):
    """
    K-fold spatial cross-validation for RotatedGPR using fixed parameters.

    Uses the angle, kernel, and jitter alpha learned on the full dataset.
    Each fold re-fits a GaussianProcessRegressor with those fixed parameters
    on the training split, then predicts the held-out split.

    Args:
        rgpr_model : fitted RotatedGPR instance
        X          : (n, 2) coordinate array
        y          : (n,) value array — may be in normal-score units if NST
                     was applied upstream
        n_folds    : number of spatial block folds
        seed       : random seed (kept for API compatibility)
        nst        : optional NormalScoreTransform instance.  If provided,
                     both Observed and Predicted are back-transformed to
                     original data units before being written to the result
                     DataFrame.  This ensures CV metrics (R², MAE, RMSE)
                     are always reported in the same units as the raw data,
                     regardless of whether NST was applied during training.

    Returns DataFrame with Observed, Predicted, Uncertainty, Residual, Z_Score.
    All columns are in original data units when nst is provided.
    """
    params = rgpr_model.get_kernel_params()
    angle = params["rotation_angle_deg"]
    # jitter_alpha is the key name in the new composite kernel architecture;
    # fall back to best_alpha_ if the key is missing (e.g. older model objects).
    alpha = params.get("jitter_alpha", getattr(rgpr_model, "best_alpha_", 1e-6))

    # Clone the full fitted composite kernel (ConstantKernel * Spatial + White)
    fitted_kernel = clone(rgpr_model.gp_model_.kernel_)

    # Spatial block folds: contiguous geographic strips along diagonal direction.
    clusters = make_spatial_block_folds(X, n_folds)

    results = []
    for fold in range(n_folds):
        tr = clusters != fold
        te = clusters == fold
        X_tr, y_tr = X[tr], y[tr]
        X_te, y_te = X[te], y[te]

        # Center + rotate using model's convention
        if rgpr_model.center_coords:
            center = np.mean(X_tr, axis=0)
            X_tr_c = X_tr - center
            X_te_c = X_te - center
        else:
            X_tr_c, X_te_c = X_tr, X_te

        X_tr_rot = rgpr_model._rotate_coords(X_tr_c, angle)
        X_te_rot = rgpr_model._rotate_coords(X_te_c, angle)

        gp = GaussianProcessRegressor(
            kernel=clone(fitted_kernel), alpha=alpha,
            optimizer=None, normalize_y=True, random_state=seed,
        )
        gp.fit(X_tr_rot, y_tr)
        pred, std = gp.predict(X_te_rot, return_std=True)

        # ── NST back-transform (approach b: original units) ───────────────
        # If NST was applied before training, y and pred are in normal-score
        # space.  We back-transform both observed and predicted values so
        # that CV metrics are always in the original data domain.
        if nst is not None:
            obs_bt  = nst.inverse_transform(y_te)
            pred_bt = nst.inverse_transform(pred)
            # Propagate std through the nonlinear inverse: finite-difference
            # derivative dx/dz evaluated at the predicted normal-score value.
            delta   = 0.01
            dnst    = 0.5 * np.abs(
                nst.inverse_transform(pred + delta) -
                nst.inverse_transform(pred - delta)
            ) / delta
            std_bt  = dnst * std
        else:
            obs_bt, pred_bt, std_bt = y_te, pred, std

        for j in range(len(y_te)):
            res = obs_bt[j] - pred_bt[j]
            results.append({
                "X": X_te[j, 0], "Y": X_te[j, 1],
                "Observed":    float(obs_bt[j]),
                "Predicted":   float(pred_bt[j]),
                "Uncertainty": float(std_bt[j]),
                "Residual":    float(res),
                "Z_Score":     float(res / (std_bt[j] + 1e-12)),
                "Abs_Error":   float(abs(res)),
                "Fold": fold,
            })
    return pd.DataFrame(results)


def perform_kriging_kfold_cv(ak_model, X, y, n_folds=5, seed=42, nst=None):
    """K-fold spatial CV for AnisotropicKriging with per-fold nugget estimation.

    Two key improvements over a naive KMeans CV:

    1. **Spatial block folds** (``make_spatial_block_folds``): test points fall
       inside geographic strips surrounded by training data, so predictions are
       genuine spatial interpolation — not extrapolation into isolated clusters.

    2. **Per-fold nugget re-estimation**: the nugget (C₀) quantifies measurement
       noise + micro-scale variability.  When a fold removes one geographic band
       of data the local noise level can differ from the global estimate.  A
       lightweight variogram fit on just the training split re-estimates C₀ for
       that fold, giving each OrdinaryKriging instance a more accurate nugget
       and thus better-calibrated prediction variances.

    Args:
        ak_model: Fitted AnisotropicKriging instance.
        X: (n, 2) spatial coordinate array.
        y: (n,) observation array — may be in normal-score units if NST applied.
        n_folds: Number of geographic block folds.
        seed: Unused (kept for API compatibility with GP version).
        nst: optional NormalScoreTransform instance.  If provided, both
             Observed and Predicted are back-transformed to original data
             units before being written to the result DataFrame, so CV
             metrics (R², MAE, RMSE) are always in original data units.

    Returns:
        pd.DataFrame with columns Observed, Predicted, Uncertainty, Residual,
        Z_Score, Abs_Error, Fold, X, Y.  All in original units when nst given.
    """
    from pykrige.ok import OrdinaryKriging

    # Spatial block folds — contiguous geographic strips (see make_spatial_block_folds)
    clusters = make_spatial_block_folds(X, n_folds)

    global_bp = dict(ak_model.best_params_)
    model_name = ak_model.best_model_name_

    results = []
    for fold in range(n_folds):
        tr = clusters != fold
        te = clusters == fold
        X_tr, y_tr = X[tr], y[tr]
        X_te, y_te = X[te], y[te]

        if len(X_tr) < 4 or len(X_te) == 0:
            # Too few training points to fit a variogram in this fold — skip
            continue

        # ── Per-fold nugget re-estimation ─────────────────────────────────
        # Fit a fast single-structure variogram on the training split using
        # the same model family as the global best, but only optimise the
        # nugget (C₀) and sill (C), fixing the range to the global value.
        # This prevents the global nugget (estimated on all data) from being
        # used blindly in a fold that may have locally higher or lower noise.
        fold_bp = dict(global_bp)  # start from global best
        try:
            from skgstat import Variogram as _SVario
            _sv = _SVario(
                X_tr, y_tr,
                model=model_name,
                n_lags=10,
                maxlag=global_bp.get("range", None),
                fit_method="trf",
                estimator="matheron",
            )
            _params = _sv.parameters  # [range, sill, nugget]
            if _params is not None and len(_params) >= 3:
                fold_nugget = float(max(_params[2], 0.0))
                fold_psill  = float(max(_params[1], 1e-6))
                fold_bp["nugget"] = fold_nugget
                fold_bp["psill"]  = fold_psill
        except Exception:
            # If skgstat is unavailable or the fit fails, fall back to global
            pass
        # ──────────────────────────────────────────────────────────────────

        try:
            ok = ak_model._get_ok_instance(X_tr, y_tr, fold_bp, model_name)
            pred, var = ok.execute("points", X_te[:, 0], X_te[:, 1])
            pred = np.asarray(pred, dtype=np.float64)
            std  = np.sqrt(np.abs(np.asarray(var, dtype=np.float64)))
        except Exception:
            continue

        # ── NST back-transform (approach b: original units) ───────────────
        # If NST was applied before training, y and pred are in normal-score
        # space.  Back-transform both so CV metrics are in original units.
        # The ±3.5 clip inside nst.inverse_transform() prevents Kriging
        # prediction overshoots from causing catastrophic tail extrapolation.
        if nst is not None:
            obs_bt  = nst.inverse_transform(y_te)
            pred_bt = nst.inverse_transform(pred)
            delta   = 0.01
            dnst    = 0.5 * np.abs(
                nst.inverse_transform(pred + delta) -
                nst.inverse_transform(pred - delta)
            ) / delta
            std_bt  = dnst * std
        else:
            obs_bt, pred_bt, std_bt = y_te, pred, std

        for j in range(len(y_te)):
            res = obs_bt[j] - pred_bt[j]
            results.append({
                "X": X_te[j, 0], "Y": X_te[j, 1],
                "Observed":    float(obs_bt[j]),
                "Predicted":   float(pred_bt[j]),
                "Uncertainty": float(std_bt[j]),
                "Residual":    float(res),
                "Z_Score":     float(res / (std_bt[j] + 1e-12)),
                "Abs_Error":   float(abs(res)),
                "Fold": fold,
            })
    return pd.DataFrame(results)


# ========================================================================
# 4. CV DASHBOARD PLOT
# ========================================================================

def plot_cv_dashboard(cv_df, engine_name="", scenario_name="",
                      save_path=None):
    """
    3-panel cross-validation dashboard:
      A) Observed vs Predicted scatter + 1:1 line + stats
      B) Spatial map of Z-scores
      C) Consistency bar chart (predicted vs observed with error bars)
    """
    fig = plt.figure(figsize=(14, 9), dpi=150)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.35, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    # ---- A: Observed vs Predicted ----
    mn = min(cv_df["Observed"].min(), cv_df["Predicted"].min())
    mx = max(cv_df["Observed"].max(), cv_df["Predicted"].max())
    ax1.plot([mn, mx], [mn, mx], "k--", alpha=0.5, label="1:1")
    sc = ax1.scatter(cv_df["Observed"], cv_df["Predicted"],
                     c=cv_df["Z_Score"].abs(), cmap="Reds",
                     edgecolor="k", s=30, vmin=0, vmax=3)
    plt.colorbar(sc, ax=ax1, label="|Z-score|", fraction=0.046)

    mae = cv_df["Abs_Error"].mean()
    rmse = np.sqrt((cv_df["Residual"] ** 2).mean())
    rmss = np.sqrt((cv_df["Z_Score"] ** 2).mean())
    corr = cv_df["Observed"].corr(cv_df["Predicted"])
    stats = f"MAE={mae:.3f}\nRMSE={rmse:.3f}\nRMSS={rmss:.3f}\nCorr={corr:.3f}"
    ax1.text(0.05, 0.95, stats, transform=ax1.transAxes, va="top",
             fontsize=9, family="monospace",
             bbox=dict(boxstyle="round", fc="white", alpha=0.9))
    ax1.set_xlabel("Observed")
    ax1.set_ylabel("Predicted")
    ax1.set_title("Goodness of Fit", fontweight="bold")
    ax1.grid(True, ls=":", alpha=0.5)

    # ---- B: Spatial Z-score ----
    sc2 = ax2.scatter(cv_df["X"], cv_df["Y"], c=cv_df["Z_Score"],
                      cmap="RdBu_r", s=30, edgecolor="k", lw=0.5,
                      vmin=-3, vmax=3)
    plt.colorbar(sc2, ax=ax2, label="Z-score", fraction=0.046)
    ax2.set_aspect("equal")
    ax2.set_title("Spatial Z-Score Map", fontweight="bold")
    ax2.grid(True, ls=":", alpha=0.5)

    # ---- C: Consistency ----
    df_s = cv_df.sort_values("Observed").reset_index(drop=True)
    idx = np.arange(len(df_s))
    ax3.errorbar(idx, df_s["Predicted"], yerr=1.96 * df_s["Uncertainty"],
                 fmt="none", ecolor="gray", alpha=0.4, capsize=2,
                 label="95% CI")
    ax3.scatter(idx, df_s["Predicted"], c="red", marker="x", s=20,
                label="Predicted", zorder=5)
    ax3.scatter(idx, df_s["Observed"], c="black", marker="o", s=15,
                alpha=0.6, label="Observed", zorder=4)
    ax3.set_xlabel("Point index (sorted by observed)")
    ax3.set_ylabel("Value")
    ax3.set_title("Consistency Check", fontweight="bold")
    ax3.legend(fontsize=8, ncol=3, loc="upper left")
    ax3.grid(True, ls=":", alpha=0.5)

    fig.suptitle(f"CV Dashboard  -  {scenario_name}  [{engine_name}]",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


# ========================================================================
# 5. ANISOTROPY ELLIPSE PLOT
# ========================================================================

def plot_anisotropy_ellipse(params, true_params=None, engine_name="",
                            scenario_name="", save_path=None):
    """Draw correlation ellipse comparing recovered vs true anisotropy."""
    angle = params.get("rotation_angle_deg", 0)
    
    # Kriging uses 'range' and 'anisotropy_ratio'
    # GP uses 'length_scale' (which can be a list [ls_major, ls_minor])
    ls = params.get("length_scale")
    ratio = params.get("anisotropy_ratio", 1.0)
    
    if ls is not None:
        if not hasattr(ls, '__len__'):
            ls_major = float(ls)
            ls_minor = float(ls) / ratio
        else:
            ls_major = float(ls[0])
            ls_minor = float(ls[1])
    else:
        # Fallback to 'range' for Kriging
        ls_major = float(params.get("range", 1.0))
        ls_minor = ls_major / ratio

    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)

    # Recovered ellipse
    # width: diameter along the major axis
    # height: diameter along the minor axis
    ell = Ellipse((0, 0), width=2 * ls_major, height=2 * ls_minor, angle=angle,
                  ec="red", fc="salmon", alpha=0.3, lw=2,
                  label=f"Recovered (angle={angle:.1f}, ratio={ratio:.2f})")
    ax.add_patch(ell)

    if true_params is not None:
        ta = true_params.get("angle", 0)
        tr = true_params.get("range_major", 1)
        tmin = true_params.get("range_minor", 1)
        ell2 = Ellipse((0, 0), 2 * tr, 2 * tmin, angle=ta,
                       ec="blue", fc="none", lw=2, ls="--",
                       label=f"True (angle={ta:.1f}, ratio={tr/tmin:.2f})")
        ax.add_patch(ell2)

    lim = max(ls_major, true_params.get("range_major", 1) if true_params else 1) * 1.3
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.grid(True, ls="--", alpha=0.3)
    ax.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax.axvline(0, color="k", lw=0.5, alpha=0.3)
    ax.set_title(f"Anisotropy Ellipse  -  {scenario_name}  [{engine_name}]",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


# ========================================================================
# 6. CONVEX HULL PLOT
# ========================================================================

def plot_convex_hull(X, Y, Z, hull_vertices, X_grid, Y_grid, mask,
                     scenario_name="", save_path=None):
    """
    Plot input sample points coloured by value, overlaid with the buffered
    convex hull polygon and the valid prediction grid region.

    Parameters
    ----------
    X, Y, Z      : 1D arrays – input data coordinates and values
    hull_vertices : (M, 2) array – closed polygon of the buffered hull
    X_grid, Y_grid : 2D meshgrid arrays
    mask          : 1D boolean array (flattened grid)
    """
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

    # Shade the valid prediction area
    valid_x = X_grid.flatten()[mask]
    valid_y = Y_grid.flatten()[mask]
    ax.scatter(valid_x, valid_y, c="lightyellow", s=2, marker="s",
               alpha=0.4, label="Prediction grid", zorder=1)

    # Hull polygon
    ax.plot(hull_vertices[:, 0], hull_vertices[:, 1], "b-", lw=2,
            label="Buffered convex hull", zorder=3)
    ax.fill(hull_vertices[:, 0], hull_vertices[:, 1],
            fc="cornflowerblue", alpha=0.08, zorder=2)

    # Input points
    sc = ax.scatter(X, Y, c=Z, cmap="viridis", s=30, edgecolor="k",
                    lw=0.5, zorder=4, label="Input data")
    plt.colorbar(sc, ax=ax, label="Value", fraction=0.046, pad=0.04)

    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_title(f"Convex Hull & Prediction Domain  –  {scenario_name}",
                 fontsize=13, fontweight="bold")
    ax.set_aspect("equal")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, ls="--", alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


# ========================================================================
# 7. PREDICTION SURFACE PLOT
# ========================================================================

def plot_prediction_surface(X_grid, Y_grid, pred_mean, pred_std,
                            X_obs=None, Y_obs=None,
                            hull_vertices=None,
                            scenario_name="", engine_name="",
                            save_path=None):
    """
    Side-by-side filled contour plots of predicted mean and uncertainty.

    Parameters
    ----------
    X_grid, Y_grid : 2D meshgrid arrays
    pred_mean      : 2D array (NaN outside hull)
    pred_std       : 2D array (NaN outside hull)
    X_obs, Y_obs   : optional 1D arrays of observation locations
    hull_vertices  : optional (M, 2) closed polygon
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=150)

    # -- Mean surface --
    c1 = ax1.contourf(X_grid, Y_grid, pred_mean, levels=30,
                      cmap="viridis", extend="both")
    plt.colorbar(c1, ax=ax1, label="Predicted Mean", fraction=0.046)
    if hull_vertices is not None:
        ax1.plot(hull_vertices[:, 0], hull_vertices[:, 1],
                 "w-", lw=1.5, alpha=0.8)
    if X_obs is not None:
        ax1.scatter(X_obs, Y_obs, c="red", s=8, marker="x",
                    alpha=0.7, label="Observations")
        ax1.legend(fontsize=8, loc="upper right")
    ax1.set_aspect("equal")
    ax1.set_title("Predicted Mean", fontweight="bold")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")

    # -- Uncertainty surface --
    c2 = ax2.contourf(X_grid, Y_grid, pred_std, levels=30,
                      cmap="magma_r", extend="both")
    plt.colorbar(c2, ax=ax2, label="Std Dev (Uncertainty)", fraction=0.046)
    if hull_vertices is not None:
        ax2.plot(hull_vertices[:, 0], hull_vertices[:, 1],
                 "w-", lw=1.5, alpha=0.8)
    if X_obs is not None:
        ax2.scatter(X_obs, Y_obs, c="cyan", s=8, marker="x",
                    alpha=0.7, label="Observations")
        ax2.legend(fontsize=8, loc="upper right")
    ax2.set_aspect("equal")
    ax2.set_title("Prediction Uncertainty", fontweight="bold")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")

    fig.suptitle(
        f"Interpolation Surface  –  {scenario_name}  [{engine_name}]",
        fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


# ========================================================================
# 8. GROUND TRUTH SURFACE PLOT
# ========================================================================

def plot_ground_truth(gt_X, gt_Y, gt_Z, sample_X=None, sample_Y=None,
                      scenario_name="", save_path=None):
    """
    Contour plot of the dense noise-free ground truth field.

    Parameters
    ----------
    gt_X, gt_Y, gt_Z : 1D arrays from the ground truth CSV
    sample_X, sample_Y : optional 1D arrays – locations of the sparse samples
    """
    from matplotlib.tri import Triangulation

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

    tri = Triangulation(gt_X, gt_Y)
    cf = ax.tricontourf(tri, gt_Z, levels=40, cmap="viridis", extend="both")
    plt.colorbar(cf, ax=ax, label="Value", fraction=0.046, pad=0.04)

    if sample_X is not None:
        ax.scatter(sample_X, sample_Y, c="red", s=18, marker="x",
                   linewidths=0.8, alpha=0.9, zorder=5,
                   label=f"Sample points (n={len(sample_X)})")
        ax.legend(fontsize=9, loc="upper right")

    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_title(f"Ground Truth (noise-free)  –  {scenario_name}",
                 fontsize=13, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(True, ls="--", alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


# ========================================================================
# 9. COMPARISON: GROUND TRUTH vs PREDICTION
# ========================================================================

def plot_comparison(X_grid, Y_grid, pred_mean, gt_X, gt_Y, gt_Z,
                    hull_vertices=None, scenario_name="", engine_name="",
                    save_path=None):
    """
    Three-panel comparison: Ground Truth | Prediction | Absolute Error.

    The ground truth is re-gridded onto the prediction meshgrid via
    nearest-neighbour interpolation so that pixel-wise error can be computed.
    """
    from scipy.interpolate import griddata

    fig, axes = plt.subplots(1, 3, figsize=(20, 7), dpi=150)

    # Re-grid ground truth onto the prediction mesh
    gt_on_grid = griddata(
        np.column_stack((gt_X, gt_Y)), gt_Z,
        (X_grid, Y_grid), method="cubic",
    )

    # Shared colour limits for truth and prediction
    valid_pred = pred_mean[~np.isnan(pred_mean)]
    valid_gt = gt_on_grid[~np.isnan(gt_on_grid)]
    if len(valid_pred) > 0 and len(valid_gt) > 0:
        vmin = min(valid_pred.min(), valid_gt.min())
        vmax = max(valid_pred.max(), valid_gt.max())
    else:
        vmin, vmax = None, None

    # --- Panel A: Ground Truth ---
    c0 = axes[0].contourf(X_grid, Y_grid, gt_on_grid, levels=30,
                          cmap="viridis", vmin=vmin, vmax=vmax, extend="both")
    plt.colorbar(c0, ax=axes[0], fraction=0.046, label="Value")
    if hull_vertices is not None:
        axes[0].plot(hull_vertices[:, 0], hull_vertices[:, 1],
                     "w-", lw=1.5, alpha=0.8)
    axes[0].set_title("Ground Truth", fontweight="bold")
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")

    # --- Panel B: Prediction ---
    c1 = axes[1].contourf(X_grid, Y_grid, pred_mean, levels=30,
                          cmap="viridis", vmin=vmin, vmax=vmax, extend="both")
    plt.colorbar(c1, ax=axes[1], fraction=0.046, label="Value")
    if hull_vertices is not None:
        axes[1].plot(hull_vertices[:, 0], hull_vertices[:, 1],
                     "w-", lw=1.5, alpha=0.8)
    axes[1].set_title(f"Predicted ({engine_name})", fontweight="bold")
    axes[1].set_aspect("equal")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")

    # --- Panel C: Absolute Error ---
    abs_err = np.abs(pred_mean - gt_on_grid)
    c2 = axes[2].contourf(X_grid, Y_grid, abs_err, levels=30,
                          cmap="hot_r", extend="max")
    plt.colorbar(c2, ax=axes[2], fraction=0.046, label="|Error|")
    if hull_vertices is not None:
        axes[2].plot(hull_vertices[:, 0], hull_vertices[:, 1],
                     "k-", lw=1.5, alpha=0.8)
    axes[2].set_title("Absolute Error", fontweight="bold")
    axes[2].set_aspect("equal")
    axes[2].set_xlabel("X")
    axes[2].set_ylabel("Y")

    # Summary stats
    valid_err = abs_err[~np.isnan(abs_err)]
    if len(valid_err) > 0:
        stats = (f"MAE={np.mean(valid_err):.4f}  |  "
                 f"Max={np.max(valid_err):.4f}  |  "
                 f"RMSE={np.sqrt(np.mean(valid_err**2)):.4f}")
    else:
        stats = ""

    fig.suptitle(
        f"Comparison  –  {scenario_name}  [{engine_name}]\n{stats}",
        fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


# ========================================================================
# 10. TREND COMPONENTS PLOT
# ========================================================================

def plot_trend_components(X_coord, Y_coord, Z_val, Z_fit, processor,
                          X_grid, Y_grid, mask, hull_vertices=None,
                          scenario_name="", save_path=None):
    """
    Plot the detrending components: Original Data, Estimated Trend, and Residuals.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), dpi=150)

    valid_x = X_grid.flatten()[mask]
    valid_y = Y_grid.flatten()[mask]
    
    # Calculate trend on the grid
    trend_grid = np.full(X_grid.shape, np.nan)
    if len(valid_x) > 0:
        trend_vals = processor.get_trend(valid_x, valid_y)
        trend_grid.flat[mask] = trend_vals

    # Common limits for original data and trend
    vmin = np.min(Z_val)
    vmax = np.max(Z_val)

    # A: Original Data
    sc0 = axes[0].scatter(X_coord, Y_coord, c=Z_val, cmap="viridis", vmin=vmin, vmax=vmax, s=30, edgecolor="k", lw=0.5)
    plt.colorbar(sc0, ax=axes[0], fraction=0.046, label="Value")
    if hull_vertices is not None:
        axes[0].plot(hull_vertices[:, 0], hull_vertices[:, 1], "k-", lw=1.5, alpha=0.8)
    axes[0].set_title("Original Data", fontweight="bold")
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    
    # B: Estimated Trend
    c1 = axes[1].contourf(X_grid, Y_grid, trend_grid, levels=30, cmap="viridis", vmin=vmin, vmax=vmax, extend="both")
    plt.colorbar(c1, ax=axes[1], fraction=0.046, label="Trend Value")
    if hull_vertices is not None:
        axes[1].plot(hull_vertices[:, 0], hull_vertices[:, 1], "k-", lw=1.5, alpha=0.8)
    axes[1].set_title(f"Estimated Trend (Order {processor.order})", fontweight="bold")
    axes[1].set_aspect("equal")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")

    # C: Residuals (Z_fit)
    res_max = np.max(np.abs(Z_fit))
    sc2 = axes[2].scatter(X_coord, Y_coord, c=Z_fit, cmap="RdBu_r", vmin=-res_max, vmax=res_max, s=30, edgecolor="k", lw=0.5)
    plt.colorbar(sc2, ax=axes[2], fraction=0.046, label="Residual")
    if hull_vertices is not None:
        axes[2].plot(hull_vertices[:, 0], hull_vertices[:, 1], "k-", lw=1.5, alpha=0.8)
    axes[2].set_title("Stochastic Residuals", fontweight="bold")
    axes[2].set_aspect("equal")
    axes[2].set_xlabel("X")
    axes[2].set_ylabel("Y")

    fig.suptitle(f"Trend Components  –  {scenario_name}", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig
