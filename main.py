"""
main.py - Entry point for the standalone Interpolation Engine.

Usage:
    conda run -n fafalab python main.py
    python main.py

Output files are alphabetically prefixed so that the natural sort order
matches the logical viewing sequence:

    A_ground_truth.png        (only if ground_truth_filepath is set)
    B_convex_hull.png
    C_trend_components.png    (only if detrending is enabled)
    D_variogram_omni_<engine>.png
    E_variogram_directional_<engine>.png
    F_anisotropy_ellipse_<engine>.png
    G_prediction_surface_<engine>.png
    H_comparison_<engine>.png          (only if ground_truth_filepath is set)
    I_cv_dashboard_<engine>.png
    cv_results_<engine>.csv
    parameters_<engine>.txt / .json
    predicted_<engine>.nc
"""
import yaml
import json
from pathlib import Path
import numpy as np
import pandas as pd
import time

from src.data_loader import load_input_data, load_custom_prediction_points
from src.geometry import generate_prediction_grid
from src.exporter import export_grid, _VALID_POINT_FORMATS
from src.engines.gp import RotatedGPR
from src.engines.kriging import AnisotropicKriging
from src.preprocessor import TrendProcessor, analyze_trend, NormalScoreTransform
from utils import (
    compute_empirical_variogram,
    plot_variogram,
    plot_directional_variogram,
    plot_anisotropy_ellipse,
    plot_convex_hull,
    plot_prediction_surface,
    plot_ground_truth,
    plot_comparison,
    perform_gpr_kfold_cv,
    perform_kriging_kfold_cv,
    plot_cv_dashboard,
    plot_trend_components,
)
# sklearn kernel imports are no longer needed in main.py:
# the composite kernel (ConstantKernel * Matern/RBF + WhiteKernel) is now
# constructed entirely inside RotatedGPR.fit() using adaptive bounds passed
# as constructor arguments.  Removing the import keeps the namespace clean.


# ── helpers ───────────────────────────────────────────────────────────────

def load_config(filepath: str) -> dict:
    """Load YAML configuration file."""
    with open(filepath, "r") as f:
        return yaml.safe_load(f)


def derive_output_dir(config: dict) -> Path:
    """
    Build output directory from the input filename stem.
    e.g.  test_data/S3_Sparse_Aniso_120deg.csv  →  output/S3_Sparse_Aniso_120deg/
    """
    base = Path(config.get("output", {}).get("base_directory", "output"))
    input_stem = Path(config["input"]["filepath"]).stem
    return base / input_stem


def load_ground_truth(config: dict):
    """
    Load the optional ground truth CSV.
    Returns (gt_X, gt_Y, gt_Z) or None if not configured.
    """
    gt_path_str = config.get("input", {}).get("ground_truth_filepath", "")
    if not gt_path_str:
        return None

    gt_path = Path(gt_path_str)
    if not gt_path.exists():
        print(f"  ⚠ Ground truth file not found: {gt_path}  (skipping)")
        return None

    cols = config["input"].get("columns", {})
    col_x = cols.get("x", "X")
    col_y = cols.get("y", "Y")
    col_val = cols.get("value", "Value")

    df = pd.read_csv(gt_path)
    return df[col_x].values, df[col_y].values, df[col_val].values


def save_parameter_summary(params: dict, mode: str, out_dir: Path):
    """Write human-readable and JSON parameter summaries."""
    with open(out_dir / f"parameters_{mode}.txt", "w") as f:
        f.write(f"Engine : {mode.upper()}\n")
        f.write("=" * 50 + "\n")
        for k, v in params.items():
            f.write(f"{k:30s}: {v}\n")

    with open(out_dir / f"parameters_{mode}.json", "w") as f:
        json.dump({"engine": mode, **params}, f, indent=2, default=str)


# ── coordinate cleaning ───────────────────────────────────────────────────

def check_and_clean_duplicates(
    X: np.ndarray,
    Z: np.ndarray,
    min_separation: float,
) -> tuple:
    """
    Detect and resolve duplicate or near-duplicate sample locations.

    Why this matters
    ----------------
    Two samples at (or very close to) the same spatial location with
    different values create a fundamental contradiction for any spatial
    model: the covariance matrix becomes (nearly) singular because the
    distance between the pair is essentially zero, yet their values differ.

    Kriging with OrdinaryKriging will crash with a LinAlgError.
    GP with alpha=1e-6 will silently produce nonsense predictions because
    the jitter cannot overcome the near-singularity.

    Cause in real data: GPS snapping, survey rounding, digitisation errors.

    Resolution strategy (in order of severity):
      1. **Exact duplicates** (distance == 0): replace both points with one
         point whose value is the arithmetic mean of all co-located values.
      2. **Near-duplicates** (0 < distance < min_separation): add a small
         independent spatial jitter to each point so they are at least
         min_separation apart.  The jitter magnitude is min_separation / 10
         so the displacement is negligible relative to the variogram range.

    Args:
        X              : (n, 2) coordinate array
        Z              : (n,) value array
        min_separation : minimum allowed distance between two distinct points

    Returns:
        X_clean : (m, 2) cleaned coordinate array  (m ≤ n)
        Z_clean : (m,) cleaned value array
        report  : dict with keys n_exact, n_near, n_removed, jitter_applied
    """
    from scipy.spatial.distance import pdist, squareform

    n_original = len(Z)
    X = np.asarray(X, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)

    dist_mat = squareform(pdist(X))

    # ── Step 1: merge exact duplicates (distance == 0) ────────────────────────
    # Build clusters of co-located points iteratively.
    visited   = np.zeros(len(X), dtype=bool)
    keep_mask = np.ones(len(X),  dtype=bool)
    n_exact   = 0

    for i in range(len(X)):
        if visited[i]:
            continue
        exact_group = np.where(dist_mat[i] == 0.0)[0]
        if len(exact_group) > 1:
            # Replace the first occurrence with the group mean; discard the rest
            Z[exact_group[0]] = np.mean(Z[exact_group])
            keep_mask[exact_group[1:]] = False
            n_exact += len(exact_group) - 1
        visited[exact_group] = True

    X = X[keep_mask]
    Z = Z[keep_mask]

    # ── Step 2: jitter near-duplicates (0 < distance < min_separation) ───────
    dist_mat2 = squareform(pdist(X))
    np.fill_diagonal(dist_mat2, np.inf)

    rng         = np.random.default_rng(42)
    jitter_mag  = min_separation / 10.0
    jitter_mask = np.zeros(len(X), dtype=bool)

    for i in range(len(X)):
        if np.any(dist_mat2[i] < min_separation):
            jitter_mask[i] = True

    n_near = int(jitter_mask.sum())
    if n_near > 0:
        jitter = rng.uniform(-jitter_mag, jitter_mag, size=(n_near, 2))
        X[jitter_mask] += jitter

    report = {
        "n_original":    n_original,
        "n_exact":       n_exact,
        "n_near":        n_near,
        "n_removed":     n_original - len(Z),
        "jitter_applied": n_near > 0,
        "min_separation": min_separation,
    }
    return X, Z, report


# ── main pipeline ─────────────────────────────────────────────────────────

def run_pipeline():
    import sys
    config_path_str = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    config_path = Path(config_path_str)
    if not config_path.exists():
        print(f"Error: {config_path} not found.")
        return

    config = load_config(config_path)
    out_dir = derive_output_dir(config)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_cfg = config.get("output", {})
    scenario_name = Path(config["input"]["filepath"]).stem

    print("=" * 60)
    print(f"  Interpolation Engine  –  {scenario_name}")
    print("=" * 60)

    # ── 1. Load Data ──────────────────────────────────────────────────
    print("\n[1/7] Loading data ...")
    X_coord, Y_coord, Z_val = load_input_data(config)
    X = np.column_stack((X_coord, Y_coord))
    print(f"       {len(Z_val)} sample points loaded.")

    # ── 1.1. Duplicate / near-duplicate coordinate guard ──────────────
    # Two samples at (or very near) the same location will make the
    # covariance matrix singular and cause a crash or silent failure.
    # This step merges exact duplicates (taking the mean value) and
    # jitters near-duplicates to enforce a minimum separation distance.
    dup_cfg = config.get("preprocessing", {}).get("duplicates", {})
    # Compute default min_separation from data geometry if not set in config
    from scipy.spatial.distance import pdist as _pdist_dup, squareform as _sq_dup
    _d_dup = _sq_dup(_pdist_dup(X))
    np.fill_diagonal(_d_dup, np.inf)
    _median_nn_dup = float(np.median(_d_dup.min(axis=1)))
    default_sep = max(_median_nn_dup * 0.1, 1e-3)
    min_sep = dup_cfg.get("min_separation", None) or default_sep

    X, Z_val, dup_report = check_and_clean_duplicates(X, Z_val, min_separation=min_sep)
    X_coord, Y_coord = X[:, 0], X[:, 1]

    if dup_report["n_exact"] > 0 or dup_report["n_near"] > 0:
        print(f"  ⚠  Duplicate guard: {dup_report['n_exact']} exact duplicates merged, "
              f"{dup_report['n_near']} near-duplicates jittered "
              f"(min_sep={min_sep:.3f} m).")
        print(f"       Points after cleaning: {len(Z_val)} "
              f"(removed {dup_report['n_removed']})")
    else:
        print(f"       No duplicate coordinates detected (min_sep={min_sep:.3f} m). ✓")

    gt = load_ground_truth(config)
    if gt is not None:
        gt_X, gt_Y, gt_Z = gt
        print(f"       {len(gt_Z)} ground-truth points loaded.")

    # ── 1.5. Preprocessing (Detrending) ───────────────────────────────
    pre_cfg = config.get("preprocessing", {}).get("detrend", {})
    do_detrend = pre_cfg.get("enabled", False)
    auto_detect = pre_cfg.get("auto_detect", True)
    trend_order = pre_cfg.get("order", 1)

    print("\n[1.5/7] Trend Analysis & Preprocessing ...")
    if auto_detect:
        # Pass trend_order so the F-test matches the polynomial order that will
        # actually be used for detrending — detection and correction are consistent.
        trend_stats = analyze_trend(X_coord, Y_coord, Z_val, order=trend_order)
        print(f"       Polynomial order tested       : {trend_stats['tested_order']}")
        print(f"       Trend F-test p-value          : {trend_stats['f_pvalue']:.4e}")
        print(f"       Trend R²                      : {trend_stats['r2']:.4f}")
        if not np.isnan(trend_stats['moran_i']):
            print(f"       Moran's I (raw data)         : {trend_stats['moran_i']:.4f} (p={trend_stats['moran_p']:.4e})")

        # ── Normality report (from analyze_trend) ─────────────────────────────
        norm = trend_stats.get("normality", {})
        if norm:
            sw_p = norm.get("shapiro_p", float("nan"))
            sk   = norm.get("skewness",  0.0)
            kurt = norm.get("kurtosis",  0.0)
            print(f"       Shapiro-Wilk p-value         : {sw_p:.4e}  "
                  f"({'normal' if norm.get('is_normal') else 'NON-NORMAL'})")
            print(f"       Skewness / Excess Kurtosis   : {sk:.3f} / {kurt:.3f}")

        do_detrend = trend_stats['recommend_detrend']
        if do_detrend:
            print("       -> Significant trend detected. Detrending ENABLED.")
        else:
            print("       -> No significant trend. Detrending DISABLED.")
    else:
        print(f"       Auto-detect disabled. Detrending is {'ENABLED' if do_detrend else 'DISABLED'} by config.")
        norm = {}

    if do_detrend:
        print(f"       Detrending (order={trend_order}) ...")
        processor = TrendProcessor(order=trend_order)
        processor.fit(X_coord, Y_coord, Z_val)
        Z_fit = processor.detrend(X_coord, Y_coord, Z_val)

        t_params = processor.get_params()
        print(f"       Trend fitted. Intercept: {t_params['intercept']:.3f}")
    else:
        processor = None
        Z_fit = Z_val

    # ── Normal-Score Transform (NST) ──────────────────────────────────────────
    # Applied AFTER detrending (the residuals should be Gaussian; if they are
    # not, NST corrects the marginal distribution before spatial modelling).
    #
    # Priority order:
    #   1. config key  preprocessing.nst.enabled  (explicit user override)
    #   2. auto-detect via Shapiro-Wilk + skewness (from analyze_trend)
    #   3. default: disabled
    #
    # The NST is always invertible: back-transform is applied to predictions
    # in [6/7] before writing outputs, so all output files are in original units.
    nst_cfg      = config.get("preprocessing", {}).get("nst", {})
    nst_enabled  = nst_cfg.get("enabled", None)   # None = auto-detect

    if nst_enabled is None:
        # Auto-detect: use the normality check result from analyze_trend
        nst_enabled = bool(norm.get("recommend_nst", False))

    # ── NST preflight summary ─────────────────────────────────────────────────
    # Print a clear, user-readable report of the distribution diagnostic and
    # the NST decision BEFORE any model fitting.  This makes preprocessing
    # decisions transparent and allows the user to verify the auto-detect
    # logic is behaving correctly.
    print("\n       ── Distribution Diagnostic ──")
    if norm:
        _sw_sym  = "✓" if norm.get("is_normal") else "✗"
        _sk_sym  = "✓" if abs(norm.get("skewness", 0)) <= 0.5 else "✗"
        _ku_sym  = "✓" if abs(norm.get("kurtosis", 0)) <= 1.0 else "✗"
        print(f"       {_sw_sym} Shapiro-Wilk p   : {norm.get('shapiro_p', float('nan')):.4e}"
              f"  ({'normal' if norm.get('is_normal') else 'NON-NORMAL'})")
        print(f"       {_sk_sym} Skewness         : {norm.get('skewness', 0):.3f}"
              f"  (threshold |skew| > 0.5)")
        print(f"       {_ku_sym} Excess kurtosis  : {norm.get('kurtosis', 0):.3f}"
              f"  (threshold |kurt| > 1.0)")
    else:
        print("       (normality check skipped — auto_detect is disabled)")

    nst = None
    if nst_enabled:
        nst = NormalScoreTransform(tail_extrapolation=True)
        Z_fit = nst.fit_transform(Z_fit)
        nst_summary = nst.summary()
        print(f"       → NST APPLIED  (all 3 criteria met)")
        print(f"         Knots: {nst_summary['n_knots']}  "
              f"|  original range: [{nst_summary['x_min']:.3f}, {nst_summary['x_max']:.3f}]")
        print(f"         Z_fit is now N(0,1)-distributed.")
        print(f"         Predictions will be back-transformed to original units.")
    else:
        print(f"       → NST SKIPPED  (data is sufficiently Gaussian — no transform needed)")

    # ── 2. Geometry ───────────────────────────────────────────────────
    pred_cfg = config.get("prediction_points")
    is_point_mode = pred_cfg is not None
    
    if is_point_mode:
        print("[2/7] Point Mode active: Loading custom prediction points...")
        pts_file = pred_cfg.get("filepath")
        cols = pred_cfg.get("columns", {})
        col_x = cols.get("x") or "X"
        col_y = cols.get("y") or "Y"
        
        X_out, Y_out, df_out = load_custom_prediction_points(pts_file, col_x, col_y)
        print(f"       Loaded {len(X_out)} custom points.")
        print(f"       Note: Assuming custom points share the same CRS as input data.")
        
        X_grid, Y_grid, mask, grid_shape, hull_verts = None, None, None, None, None
        valid_points = np.column_stack((X_out, Y_out))
    else:
        print("[2/7] Building prediction grid (buffered convex hull) ...")
        X_grid, Y_grid, mask, grid_shape, hull_verts = generate_prediction_grid(
            X_coord, Y_coord, config
        )
        print(f"       Grid: {grid_shape[1]}x{grid_shape[0]}  |  "
              f"Inside hull: {np.sum(mask)} pts")

    # ── 3. Ground truth visualisation (A_) ────────────────────────────
    print("[3/7] Ground truth visualisation ...")
    if gt is not None:
        plot_ground_truth(
            gt_X, gt_Y, gt_Z,
            sample_X=X_coord, sample_Y=Y_coord,
            scenario_name=scenario_name,
            save_path=out_dir / "A_ground_truth.png",
        )
        print("       ✓ A_ground_truth.png")
    else:
        print("       (skipped – no ground_truth_filepath in config)")

    # ── 4. Set up engine ──────────────────────────────────────────────
    engine_cfg = config.get("engine", {})
    mode = engine_cfg.get("mode", "gp").lower()

    if mode == "gp":
        print("[4/7] Initialising RotatedGPR ...")
        gp_cfg = engine_cfg.get("gp", {})

        # ── Adaptive length scale bounds ──────────────────────────────────────
        # Fixed config values like ls_min=10 and ls_max=5000 are dangerous:
        # they are unrelated to the actual point spacing and domain extent of
        # the dataset being processed.  When the true spatial range falls
        # outside these bounds, the optimizer cannot find it and the kernel
        # collapses (length scale pinned to a boundary → constant predictor).
        #
        # We compute bounds directly from the data geometry:
        #   ls_min = 0.5 × median nearest-neighbour distance
        #            (below this, no pair of training points is correlated)
        #   ls_max = 0.6 × max pairwise distance
        #            (beyond this, all points are globally correlated → no info)
        #
        # The user can still override with explicit config keys if desired.
        from scipy.spatial.distance import pdist, squareform as _squareform
        _dists_sq = _squareform(pdist(X))
        np.fill_diagonal(_dists_sq, np.inf)
        _median_nn = float(np.median(_dists_sq.min(axis=1)))
        np.fill_diagonal(_dists_sq, 0.0)
        _max_dist  = float(_dists_sq.max())

        ls_min_auto = max(_median_nn * 0.5, 1e-3)
        ls_max_auto = _max_dist * 0.6

        ls_min = gp_cfg.get("length_scale_min_override", ls_min_auto)
        ls_max = gp_cfg.get("length_scale_max_override", ls_max_auto)
        ls_init = gp_cfg.get("length_scale", (ls_min + ls_max) / 2.0)

        print(f"       Adaptive ls bounds: [{ls_min:.2f},  {ls_max:.2f}]  "
              f"(median NN={_median_nn:.2f}, max_dist={_max_dist:.2f})")

        # ── Signal variance bounds from data ──────────────────────────────────
        data_var  = float(np.var(Z_fit))
        var_min   = max(data_var * 0.01, 1e-4)
        var_max   = data_var * 20.0

        # ── Nugget bounds: allow from near-zero up to 80% of data variance ────
        nug_min = gp_cfg.get("nugget_min", 1e-6)
        nug_max = gp_cfg.get("nugget_max", data_var * 0.8)

        model = RotatedGPR(
            ls_init        = ls_init,
            ls_bounds      = (ls_min, ls_max),
            var_init       = data_var,
            var_bounds     = (var_min, var_max),
            nugget_init    = data_var * 0.05,
            nugget_bounds  = (nug_min, nug_max),
            max_anisotropy = gp_cfg.get("max_anisotropy", 15.0),
            angle_bounds   = (
                gp_cfg.get("angle_min", 0.0),
                gp_cfg.get("angle_max", 180.0),
            ),
            n_optuna_trials = gp_cfg.get("n_optuna_trials", 300),
            random_state   = gp_cfg.get("random_state", None),
        )
    elif mode == "kriging":
        print("[4/7] Initialising AnisotropicKriging ...")
        k_cfg = engine_cfg.get("kriging", {})
        model = AnisotropicKriging(
            n_trials=k_cfg.get("n_trials", 50),
            n_splits=k_cfg.get("n_splits", 5),
            max_anisotropy=k_cfg.get("max_anisotropy", 3.0),
            verbose=False,
        )
    else:
        raise ValueError(f"Unknown engine mode: {mode}")

    # ── 5. Fit ────────────────────────────────────────────────────────
    print("[5/7] Fitting model ...")
    t0 = time.time()
    model.fit(X, Z_fit)
    elapsed = time.time() - t0
    print(f"       Completed in {elapsed:.2f} s.")

    params = model.get_kernel_params()
    print("       ── Recovered Parameters ──")
    for k, v in params.items():
        print(f"       {k:30s}: {v}")

    save_parameter_summary(params, mode, out_dir)

    # ── 6. Predict ────────────────────────────────────────────
    if is_point_mode:
        print("[6/7] Predicting over custom points ...")
    else:
        print("[6/7] Predicting over spatial grid ...")
        pred_mean = np.full(grid_shape, np.nan)
        pred_std = np.full(grid_shape, np.nan)

        grid_points = np.column_stack((X_grid.flatten(), Y_grid.flatten()))
        valid_points = grid_points[mask]

    if len(valid_points) > 0:
        means, stds = model.predict(valid_points, return_std=True)
        # Back-transform NST first (normal scores → original distribution),
        # then add the polynomial trend back (detrended residuals → original units).
        # Order matters: NST was applied to the detrended residuals, so NST
        # inverse must come before re-trending.
        if nst is not None:
            # std is in normal-score units; approximate back-transform via
            # finite-difference of the NST inverse *before* means is overwritten.
            delta = 0.01
            dnst  = 0.5 * np.abs(
                nst.inverse_transform(means + delta) -
                nst.inverse_transform(means - delta)
            ) / delta          # local derivative dx/dz at each prediction point
            means = nst.inverse_transform(means)
            stds  = dnst * stds   # propagate std through the nonlinear transform
        if processor is not None:
            means = processor.retrend(valid_points[:, 0], valid_points[:, 1], means)
            
        if is_point_mode:
            point_pred_mean = means
            point_pred_std = stds
        else:
            pred_mean.flat[mask] = means
            pred_std.flat[mask] = stds

    # ── 7. Diagnostics & export ───────────────────────────────────────
    print("[7/7] Generating outputs ...")

    # B_ Convex hull
    if not is_point_mode:
        plot_convex_hull(
            X_coord, Y_coord, Z_val,
            hull_verts, X_grid, Y_grid, mask,
            scenario_name=scenario_name,
            save_path=out_dir / "B_convex_hull.png",
        )
        print("       ✓ B_convex_hull.png")

    if processor is not None and out_cfg.get("save_diagnostics", True) and not is_point_mode:
        plot_trend_components(
            X_coord, Y_coord, Z_val, Z_fit, processor,
            X_grid, Y_grid, mask, hull_verts,
            scenario_name=scenario_name,
            save_path=out_dir / "C_trend_components.png"
        )
        print("       ✓ C_trend_components.png")

    if out_cfg.get("save_diagnostics", True):
        # D_ Omnidirectional variogram
        omni_var = compute_empirical_variogram(X, Z_fit)
        plot_variogram(
            omni_var,
            fitted_params=params,
            engine_name=mode.upper(),
            scenario_name=scenario_name,
            save_path=out_dir / f"D_variogram_omni_{mode}.png",
        )
        print(f"       ✓ D_variogram_omni_{mode}.png")

        # E_ Directional variogram (15° intervals)
        directions = np.arange(0, 180, 15)
        dir_vars = compute_empirical_variogram(X, Z_fit, directions=directions)
        plot_directional_variogram(
            dir_vars,
            scenario_name=scenario_name,
            save_path=out_dir / f"E_variogram_directional_{mode}.png",
        )
        print(f"       ✓ E_variogram_directional_{mode}.png")

        # F_ Anisotropy ellipse
        plot_anisotropy_ellipse(
            params,
            engine_name=mode.upper(),
            scenario_name=scenario_name,
            save_path=out_dir / f"F_anisotropy_ellipse_{mode}.png",
        )
        print(f"       ✓ F_anisotropy_ellipse_{mode}.png")

    if not is_point_mode:
        # G_ Prediction surface
        plot_prediction_surface(
            X_grid, Y_grid, pred_mean, pred_std,
            X_obs=X_coord, Y_obs=Y_coord,
            hull_vertices=hull_verts,
            scenario_name=scenario_name,
            engine_name=mode.upper(),
            save_path=out_dir / f"G_prediction_surface_{mode}.png",
        )
        print(f"       ✓ G_prediction_surface_{mode}.png")

        # H_ Comparison (only if ground truth available)
        if gt is not None:
            plot_comparison(
                X_grid, Y_grid, pred_mean,
                gt_X, gt_Y, gt_Z,
                hull_vertices=hull_verts,
                scenario_name=scenario_name,
                engine_name=mode.upper(),
                save_path=out_dir / f"H_comparison_{mode}.png",
            )
            print(f"       ✓ H_comparison_{mode}.png")

    # I_ Cross-validation dashboard
    if out_cfg.get("save_diagnostics", True):
        print("       Running cross-validation ...")
        if mode == "gp":
            cv_df = perform_gpr_kfold_cv(model, X, Z_fit, nst=nst)
        else:
            cv_df = perform_kriging_kfold_cv(model, X, Z_fit, nst=nst)
        cv_df.to_csv(out_dir / f"cv_results_{mode}.csv", index=False)
        plot_cv_dashboard(
            cv_df,
            engine_name=mode.upper(),
            scenario_name=scenario_name,
            save_path=out_dir / f"I_cv_dashboard_{mode}.png",
        )
        print(f"       ✓ I_cv_dashboard_{mode}.png")
        print(f"       ✓ cv_results_{mode}.csv")
        
        # Calculate and print evaluation metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        obs = cv_df['Observed']
        pred = cv_df['Predicted']
        mae = mean_absolute_error(obs, pred)
        rmse = np.sqrt(mean_squared_error(obs, pred))
        r2 = r2_score(obs, pred)

        # ── Theoretical R² ceiling from nugget ratio ──────────────────────
        # The nugget (C₀) represents variance that no spatial model can ever
        # explain: pure measurement noise + sub-resolution variability.
        # R²_ceiling = 1 − C₀ / (C₀ + C)   where C is the structured sill.
        # If the achieved R² is near the ceiling the engine is performing as
        # well as the data physically allows; if it is far below the ceiling
        # there is room to improve the model or data collection strategy.
        params = model.get_kernel_params() if mode == "gp" else dict(model.best_params_)
        r2_ceiling: float | None = None
        try:
            if mode == "gp":
                # WhiteKernel stores noise variance; ConstantKernel stores signal variance
                nugget_var = float(params.get("nugget_variance", 0.0))
                signal_var = float(params.get("constant_value", 1.0))
                total_var = nugget_var + signal_var
                if total_var > 0:
                    r2_ceiling = 1.0 - nugget_var / total_var
            else:
                # Kriging params contain 'nugget' (C₀) and 'psill' (structured sill C)
                nugget_var = float(params.get("nugget", 0.0))
                psill_var  = float(params.get("psill",  1.0))
                total_var = nugget_var + psill_var
                if total_var > 0:
                    r2_ceiling = 1.0 - nugget_var / total_var
        except Exception:
            r2_ceiling = None
        # ──────────────────────────────────────────────────────────────────

        # ── CV sanity check — catch silent numerical failures ─────────────
        # R² must lie in roughly [-1, 1] for a meaningful interpolation result.
        # RMSE must be smaller than the data standard deviation (otherwise the
        # model is worse than predicting the mean for every point).
        # Values outside these bounds indicate numerical failure — most commonly
        # caused by NST inverse-transform overflow (e.g., Kriging overshoot on
        # clustered data) or a singular covariance matrix in one or more folds.
        _data_std = float(np.std(Z_val))   # std of original (pre-NST) data
        _sanity_ok = True
        if r2 < -0.1:
            print(f"\n  ⚠⚠  CV SANITY FAILURE: R² = {r2:.4g} is physically impossible.")
            print(f"       This usually means NST inverse-transform produced extreme")
            print(f"       values in one or more folds (Kriging overshoot + tail extrap).")
            print(f"       CV metrics are unreliable for this run.")
            _sanity_ok = False
        if _data_std > 0 and rmse > 10.0 * _data_std:
            print(f"\n  ⚠⚠  CV SANITY FAILURE: RMSE = {rmse:.4g} is {rmse/_data_std:.1f}× "
                  f"the data std ({_data_std:.4g}).")
            print(f"       Predictions are far worse than the global mean predictor.")
            _sanity_ok = False

        print("\n       ── Cross-Validation Metrics ──")
        print(f"       MAE        : {mae:.4f}")
        print(f"       RMSE       : {rmse:.4f}")
        print(f"       R²         : {r2:.4f}")
        if not _sanity_ok:
            print(f"       ⚠  Metrics above are flagged as unreliable (see warnings).")
        if r2_ceiling is not None:
            nugget_pct = (1.0 - r2_ceiling) * 100.0
            print(f"       R² ceiling : {r2_ceiling:.4f}  "
                  f"(nugget accounts for {nugget_pct:.1f}% of total variance)")
            gap = r2_ceiling - r2
            if gap < 0.05:
                print("       ✓ Engine is near the theoretical limit for this dataset.")
            elif gap < 0.20:
                print(f"       ↑ Gap to ceiling: {gap:.4f} — moderate room for improvement.")
            else:
                print(f"       ↑ Gap to ceiling: {gap:.4f} — consider data density / model selection.")

    # Export
    requested_formats = out_cfg.get("formats", None)

    if is_point_mode:
        df_out['Predicted_Mean'] = point_pred_mean
        df_out['Predicted_Std'] = point_pred_std

        point_formats = requested_formats if requested_formats is not None else ["csv", "xz"]

        unknown_pt = set(point_formats) - _VALID_POINT_FORMATS
        if unknown_pt:
            raise ValueError(
                f"Unknown point-mode export format(s): {sorted(unknown_pt)}. "
                f"Valid options are: {sorted(_VALID_POINT_FORMATS)}"
            )

        if "csv" in point_formats:
            csv_path = out_dir / f"predicted_points_{mode}.csv"
            df_out.to_csv(csv_path, index=False)
            print(f"       ✓ predicted_points_{mode}.csv")

        if "xz" in point_formats:
            xz_path = out_dir / f"predicted_points_{mode}.xz"
            df_out.to_pickle(xz_path)
            print(f"       ✓ predicted_points_{mode}.xz")
    else:
        grid_formats = requested_formats if requested_formats is not None else ["nc"]
        z_dim = out_cfg.get("netcdf_z_dim_name", "Depth")
        export_grid(
            grid_formats,
            X_grid, Y_grid, pred_mean, pred_std,
            output_dir=out_dir,
            engine_name=mode,
            z_dim_name=z_dim,
        )

    # Parameters (already saved earlier, just confirm)
    print(f"       ✓ parameters_{mode}.txt")
    print(f"       ✓ parameters_{mode}.json")

    print("\n" + "=" * 60)
    print(f"  All outputs saved to:  {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
