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
import logging
logger = logging.getLogger(__name__)
from dataclasses import dataclass, field
from typing import Optional, Any

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


# ── Pipeline state dataclasses ──────────────────────────────────────────────

@dataclass
class LoadedData:
    """Result of _load_and_validate_data()."""
    X_coord: np.ndarray
    Y_coord: np.ndarray
    Z_val: np.ndarray
    X: np.ndarray           # stacked (n,2)
    gt: Optional[tuple]     # (gt_X, gt_Y, gt_Z) or None
    scenario_name: str


@dataclass
class PreprocessResult:
    """Result of _preprocess_trend_and_nst()."""
    Z_fit: np.ndarray
    processor: Optional[TrendProcessor]
    nst: Optional[NormalScoreTransform]
    norm: dict


@dataclass
class GeometryResult:
    """Result of _setup_prediction_targets()."""
    is_point_mode: bool
    X_grid: Optional[np.ndarray]
    Y_grid: Optional[np.ndarray]
    mask: Optional[np.ndarray]
    grid_shape: Optional[tuple]
    hull_verts: Optional[np.ndarray]
    valid_points: np.ndarray
    X_out: Optional[np.ndarray]
    Y_out: Optional[np.ndarray]
    df_out: Optional[Any]   # pd.DataFrame or None


@dataclass
class PredictResult:
    """Result of _predict()."""
    pred_mean: Optional[np.ndarray]
    pred_std: Optional[np.ndarray]
    point_pred_mean: Optional[np.ndarray]
    point_pred_std: Optional[np.ndarray]


# ── helpers ─────────────────────────────────────────────────────────────────

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
        logger.warning("Ground truth file not found: %s  (skipping)", gt_path)
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


# ── coordinate cleaning ─────────────────────────────────────────────────────

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

    # ── Step 1: merge exact duplicates (distance == 0) ──────────────────────────
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

    # ── Step 2: jitter near-duplicates (0 < distance < min_separation) ─────────
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


# ── pipeline stage functions ─────────────────────────────────────────────────

def _load_and_validate_data(config: dict, out_dir: Path) -> LoadedData:
    """[1/7] Load input data, clean duplicates, load ground truth."""
    scenario_name = Path(config["input"]["filepath"]).stem

    logger.info("\n[1/7] Loading data ...")
    X_coord, Y_coord, Z_val = load_input_data(config)
    X = np.column_stack((X_coord, Y_coord))
    logger.info(f"       {len(Z_val)} sample points loaded.")

    if len(Z_val) < 5:
        raise ValueError(
            f"Only {len(Z_val)} data point(s) loaded. At least 5 are required "
            f"for variogram fitting and kriging. Please provide a larger dataset."
        )

    # Duplicate / near-duplicate coordinate guard
    dup_cfg = config.get("preprocessing", {}).get("duplicates", {})
    from scipy.spatial.distance import pdist as _pdist_dup, squareform as _sq_dup
    _d_dup = _sq_dup(_pdist_dup(X))
    np.fill_diagonal(_d_dup, np.inf)
    _median_nn_dup = float(np.median(_d_dup.min(axis=1)))
    default_sep = max(_median_nn_dup * 0.1, 1e-3)
    min_sep = dup_cfg.get("min_separation", None) or default_sep

    X, Z_val, dup_report = check_and_clean_duplicates(X, Z_val, min_separation=min_sep)
    X_coord, Y_coord = X[:, 0], X[:, 1]

    if len(Z_val) < 3:
        raise ValueError(
            f"After duplicate removal, only {len(Z_val)} unique point(s) remain. "
            f"At least 3 spatially distinct points are required for any "
            f"kriging or GP model. The original data had "
            f"{dup_report['n_original']} points, of which "
            f"{dup_report['n_exact']} were exact duplicates."
        )

    if dup_report["n_exact"] > 0 or dup_report["n_near"] > 0:
        logger.info(f"  ⚠  Duplicate guard: {dup_report['n_exact']} exact duplicates merged, "
              f"{dup_report['n_near']} near-duplicates jittered "
              f"(min_sep={min_sep:.3f} m).")
        logger.info(f"       Points after cleaning: {len(Z_val)} "
              f"(removed {dup_report['n_removed']})")
    else:
        logger.info(f"       No duplicate coordinates detected (min_sep={min_sep:.3f} m). ✓")

    gt = load_ground_truth(config)
    if gt is not None:
        logger.info(f"       {len(gt[2])} ground-truth points loaded.")

    return LoadedData(
        X_coord=X_coord, Y_coord=Y_coord, Z_val=Z_val, X=X,
        gt=gt, scenario_name=scenario_name,
    )


def _preprocess_trend_and_nst(
    data: LoadedData, config: dict
) -> PreprocessResult:
    """[1.5/7] Trend detection, detrending, and Normal-Score Transform."""
    X_coord, Y_coord, Z_val = data.X_coord, data.Y_coord, data.Z_val

    pre_cfg = config.get("preprocessing", {}).get("detrend", {})
    do_detrend = pre_cfg.get("enabled", False)
    auto_detect = pre_cfg.get("auto_detect", True)
    trend_order = pre_cfg.get("order", 1)

    logger.info("\n[1.5/7] Trend Analysis & Preprocessing ...")
    if auto_detect:
        trend_stats = analyze_trend(X_coord, Y_coord, Z_val, order=trend_order)
        logger.info(f"       Polynomial order tested       : {trend_stats['tested_order']}")
        logger.info(f"       Trend F-test p-value          : {trend_stats['f_pvalue']:.4e}")
        logger.info(f"       Trend R²                      : {trend_stats['r2']:.4f}")
        if not np.isnan(trend_stats['moran_i']):
            logger.info(f"       Moran's I (raw data)         : {trend_stats['moran_i']:.4f} "
                  f"(p={trend_stats['moran_p']:.4e})")

        norm = trend_stats.get("normality", {})
        if norm:
            sw_p = norm.get("shapiro_p", float("nan"))
            sk   = norm.get("skewness",  0.0)
            kurt = norm.get("kurtosis",  0.0)
            logger.info(f"       Shapiro-Wilk p-value         : {sw_p:.4e}  "
                  f"({'normal' if norm.get('is_normal') else 'NON-NORMAL'})")
            logger.info(f"       Skewness / Excess Kurtosis   : {sk:.3f} / {kurt:.3f}")

        do_detrend = trend_stats['recommend_detrend']
        if do_detrend:
            logger.info("       -> Significant trend detected. Detrending ENABLED.")
        else:
            logger.info("       -> No significant trend. Detrending DISABLED.")
    else:
        logger.info(f"       Auto-detect disabled. Detrending is "
              f"{'ENABLED' if do_detrend else 'DISABLED'} by config.")
        norm = {}

    if do_detrend:
        logger.info(f"       Detrending (order={trend_order}) ...")
        processor = TrendProcessor(order=trend_order)
        processor.fit(X_coord, Y_coord, Z_val)
        Z_fit = processor.detrend(X_coord, Y_coord, Z_val)
        t_params = processor.get_params()
        logger.info(f"       Trend fitted. Intercept: {t_params['intercept']:.3f}")
    else:
        processor = None
        Z_fit = Z_val

    # Normal-Score Transform (NST)
    nst_cfg     = config.get("preprocessing", {}).get("nst", {})
    nst_enabled = nst_cfg.get("enabled", None)

    if nst_enabled is None:
        nst_enabled = bool(norm.get("recommend_nst", False))

    logger.info("\n       ── Distribution Diagnostic ──")
    if norm:
        _sw_sym = "✓" if norm.get("is_normal") else "✗"
        _sk_sym = "✓" if abs(norm.get("skewness", 0)) <= 0.5 else "✗"
        _ku_sym = "✓" if abs(norm.get("kurtosis", 0)) <= 1.0 else "✗"
        logger.info(f"       {_sw_sym} Shapiro-Wilk p   : {norm.get('shapiro_p', float('nan')):.4e}"
              f"  ({'normal' if norm.get('is_normal') else 'NON-NORMAL'})")
        logger.info(f"       {_sk_sym} Skewness         : {norm.get('skewness', 0):.3f}"
              f"  (threshold |skew| > 0.5)")
        logger.info(f"       {_ku_sym} Excess kurtosis  : {norm.get('kurtosis', 0):.3f}"
              f"  (threshold |kurt| > 1.0)")
    else:
        logger.info("       (normality check skipped — auto_detect is disabled)")

    nst = None
    if nst_enabled:
        nst = NormalScoreTransform(tail_extrapolation=True)
        Z_fit = nst.fit_transform(Z_fit)
        nst_summary = nst.summary()
        logger.info(f"       → NST APPLIED  (all 3 criteria met)")
        logger.info(f"         Knots: {nst_summary['n_knots']}  "
              f"|  original range: [{nst_summary['x_min']:.3f}, {nst_summary['x_max']:.3f}]")
        logger.info(f"         Z_fit is now N(0,1)-distributed.")
        logger.info(f"         Predictions will be back-transformed to original units.")
    else:
        logger.info(f"       → NST SKIPPED  (data is sufficiently Gaussian — no transform needed)")

    return PreprocessResult(Z_fit=Z_fit, processor=processor, nst=nst, norm=norm)


def _setup_prediction_targets(
    data: LoadedData, config: dict
) -> GeometryResult:
    """[2/7] Build prediction grid or load custom prediction points."""
    X_coord, Y_coord = data.X_coord, data.Y_coord
    pred_cfg = config.get("prediction_points")
    is_point_mode = pred_cfg is not None

    if is_point_mode:
        logger.info("[2/7] Point Mode active: Loading custom prediction points...")
        pts_file = pred_cfg.get("filepath")
        cols = pred_cfg.get("columns", {})
        col_x = cols.get("x") or "X"
        col_y = cols.get("y") or "Y"

        X_out, Y_out, df_out = load_custom_prediction_points(pts_file, col_x, col_y)
        logger.info(f"       Loaded {len(X_out)} custom points.")
        logger.info(f"       Note: Assuming custom points share the same CRS as input data.")

        return GeometryResult(
            is_point_mode=True,
            X_grid=None, Y_grid=None, mask=None, grid_shape=None, hull_verts=None,
            valid_points=np.column_stack((X_out, Y_out)),
            X_out=X_out, Y_out=Y_out, df_out=df_out,
        )
    else:
        logger.info("[2/7] Building prediction grid (buffered convex hull) ...")
        X_grid, Y_grid, mask, grid_shape, hull_verts = generate_prediction_grid(
            X_coord, Y_coord, config
        )
        logger.info(f"       Grid: {grid_shape[1]}x{grid_shape[0]}  |  "
              f"Inside hull: {np.sum(mask)} pts")

        grid_points = np.column_stack((X_grid.flatten(), Y_grid.flatten()))
        valid_points = grid_points[mask]

        return GeometryResult(
            is_point_mode=False,
            X_grid=X_grid, Y_grid=Y_grid, mask=mask,
            grid_shape=grid_shape, hull_verts=hull_verts,
            valid_points=valid_points,
            X_out=None, Y_out=None, df_out=None,
        )


def _ground_truth_visualisation(
    data: LoadedData, out_dir: Path
) -> None:
    """[3/7] Plot ground truth if available."""
    logger.info("[3/7] Ground truth visualisation ...")
    if data.gt is not None:
        gt_X, gt_Y, gt_Z = data.gt
        plot_ground_truth(
            gt_X, gt_Y, gt_Z,
            sample_X=data.X_coord, sample_Y=data.Y_coord,
            scenario_name=data.scenario_name,
            save_path=out_dir / "A_ground_truth.png",
        )
        logger.info("       ✓ A_ground_truth.png")
    else:
        logger.info("       (skipped – no ground_truth_filepath in config)")


def _create_engine(
    X: np.ndarray, Z_fit: np.ndarray, config: dict
) -> tuple:
    """[4/7] Instantiate and configure the interpolation engine.

    Returns (model, mode).
    """
    engine_cfg = config.get("engine", {})
    mode = engine_cfg.get("mode", "gp").lower()

    if mode == "gp":
        logger.info("[4/7] Initialising RotatedGPR ...")
        gp_cfg = engine_cfg.get("gp", {})

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

        logger.info(f"       Adaptive ls bounds: [{ls_min:.2f},  {ls_max:.2f}]  "
              f"(median NN={_median_nn:.2f}, max_dist={_max_dist:.2f})")

        data_var  = float(np.var(Z_fit))
        var_min   = max(data_var * 0.01, 1e-4)
        var_max   = data_var * 20.0

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
            n_jobs         = gp_cfg.get("n_jobs", 1),
        )
    elif mode == "kriging":
        logger.info("[4/7] Initialising AnisotropicKriging ...")
        k_cfg = engine_cfg.get("kriging", {})
        model = AnisotropicKriging(
            n_trials=k_cfg.get("n_trials", 50),
            n_splits=k_cfg.get("n_splits", 5),
            max_anisotropy=k_cfg.get("max_anisotropy", 3.0),
            n_jobs=k_cfg.get("n_jobs", 1),
            verbose=False,
        )
    else:
        raise ValueError(f"Unknown engine mode: {mode}")

    return model, mode


def _fit_model(
    model: Any, X: np.ndarray, Z_fit: np.ndarray, mode: str, out_dir: Path
) -> dict:
    """[5/7] Fit the model, print parameter summary, return params dict."""
    logger.info("[5/7] Fitting model ...")
    t0 = time.time()
    model.fit(X, Z_fit)
    elapsed = time.time() - t0
    logger.info(f"       Completed in {elapsed:.2f} s.")

    params = model.get_kernel_params()
    logger.info("       ── Recovered Parameters ──")
    for k, v in params.items():
        logger.info(f"       {k:30s}: {v}")

    save_parameter_summary(params, mode, out_dir)
    return params


def _predict(
    model: Any,
    geo: GeometryResult,
    pp: PreprocessResult,
) -> PredictResult:
    """[6/7] Predict on grid or points, back-transform NST and trend."""
    point_pred_mean = None
    point_pred_std = None
    pred_mean = None
    pred_std = None

    if geo.is_point_mode:
        logger.info("[6/7] Predicting over custom points ...")
    else:
        logger.info("[6/7] Predicting over spatial grid ...")
        pred_mean = np.full(geo.grid_shape, np.nan)
        pred_std = np.full(geo.grid_shape, np.nan)

    if len(geo.valid_points) > 0:
        means, stds = model.predict(geo.valid_points, return_std=True)

        if pp.nst is not None:
            delta = 0.01
            dnst = 0.5 * np.abs(
                pp.nst.inverse_transform(means + delta) -
                pp.nst.inverse_transform(means - delta)
            ) / delta
            means = pp.nst.inverse_transform(means)
            stds = dnst * stds
        if pp.processor is not None:
            means = pp.processor.retrend(
                geo.valid_points[:, 0], geo.valid_points[:, 1], means
            )

        if geo.is_point_mode:
            point_pred_mean = means
            point_pred_std = stds
        else:
            pred_mean.flat[geo.mask] = means
            pred_std.flat[geo.mask] = stds

    return PredictResult(
        pred_mean=pred_mean, pred_std=pred_std,
        point_pred_mean=point_pred_mean, point_pred_std=point_pred_std,
    )


def _export_and_diagnose(
    model: Any,
    data: LoadedData,
    geo: GeometryResult,
    pp: PreprocessResult,
    pred: PredictResult,
    params: dict,
    mode: str,
    out_dir: Path,
    config: dict,
) -> None:
    """[7/7] All diagnostics, cross-validation, metrics, and file export."""
    out_cfg = config.get("output", {})
    scenario_name = data.scenario_name
    X = data.X
    Z_fit = pp.Z_fit
    Z_val = data.Z_val

    logger.info("[7/7] Generating outputs ...")

    # B_ Convex hull
    if not geo.is_point_mode:
        plot_convex_hull(
            data.X_coord, data.Y_coord, Z_val,
            geo.hull_verts, geo.X_grid, geo.Y_grid, geo.mask,
            scenario_name=scenario_name,
            save_path=out_dir / "B_convex_hull.png",
        )
        logger.info("       ✓ B_convex_hull.png")

    if (pp.processor is not None and out_cfg.get("save_diagnostics", True)
            and not geo.is_point_mode):
        plot_trend_components(
            data.X_coord, data.Y_coord, Z_val, Z_fit, pp.processor,
            geo.X_grid, geo.Y_grid, geo.mask, geo.hull_verts,
            scenario_name=scenario_name,
            save_path=out_dir / "C_trend_components.png"
        )
        logger.info("       ✓ C_trend_components.png")

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
        logger.info(f"       ✓ D_variogram_omni_{mode}.png")

        # E_ Directional variogram (15° intervals)
        directions = np.arange(0, 180, 15)
        dir_vars = compute_empirical_variogram(X, Z_fit, directions=directions)
        plot_directional_variogram(
            dir_vars,
            scenario_name=scenario_name,
            save_path=out_dir / f"E_variogram_directional_{mode}.png",
        )
        logger.info(f"       ✓ E_variogram_directional_{mode}.png")

        # F_ Anisotropy ellipse
        plot_anisotropy_ellipse(
            params,
            engine_name=mode.upper(),
            scenario_name=scenario_name,
            save_path=out_dir / f"F_anisotropy_ellipse_{mode}.png",
        )
        logger.info(f"       ✓ F_anisotropy_ellipse_{mode}.png")

    if not geo.is_point_mode:
        # G_ Prediction surface
        plot_prediction_surface(
            geo.X_grid, geo.Y_grid, pred.pred_mean, pred.pred_std,
            X_obs=data.X_coord, Y_obs=data.Y_coord,
            hull_vertices=geo.hull_verts,
            scenario_name=scenario_name,
            engine_name=mode.upper(),
            save_path=out_dir / f"G_prediction_surface_{mode}.png",
        )
        logger.info(f"       ✓ G_prediction_surface_{mode}.png")

        # H_ Comparison (only if ground truth available)
        if data.gt is not None:
            gt_X, gt_Y, gt_Z = data.gt
            plot_comparison(
                geo.X_grid, geo.Y_grid, pred.pred_mean,
                gt_X, gt_Y, gt_Z,
                hull_vertices=geo.hull_verts,
                scenario_name=scenario_name,
                engine_name=mode.upper(),
                save_path=out_dir / f"H_comparison_{mode}.png",
            )
            logger.info(f"       ✓ H_comparison_{mode}.png")

    # I_ Cross-validation dashboard
    if out_cfg.get("save_diagnostics", True):
        logger.info("       Running cross-validation ...")
        if mode == "gp":
            cv_df = perform_gpr_kfold_cv(model, X, Z_fit, nst=pp.nst)
        else:
            cv_df = perform_kriging_kfold_cv(model, X, Z_fit, nst=pp.nst)
        cv_df.to_csv(out_dir / f"cv_results_{mode}.csv", index=False)
        plot_cv_dashboard(
            cv_df,
            engine_name=mode.upper(),
            scenario_name=scenario_name,
            save_path=out_dir / f"I_cv_dashboard_{mode}.png",
        )
        logger.info(f"       ✓ I_cv_dashboard_{mode}.png")
        logger.info(f"       ✓ cv_results_{mode}.csv")

        # Calculate and print evaluation metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        obs = cv_df['Observed']
        pred_cv = cv_df['Predicted']
        mae = mean_absolute_error(obs, pred_cv)
        rmse = np.sqrt(mean_squared_error(obs, pred_cv))
        r2 = r2_score(obs, pred_cv)

        # Theoretical R² ceiling from nugget ratio
        cv_params = model.get_kernel_params() if mode == "gp" else dict(model.best_params_)
        r2_ceiling: Optional[float] = None
        try:
            if mode == "gp":
                nugget_var = float(cv_params.get("nugget_variance", 0.0))
                signal_var = float(cv_params.get("constant_value", 1.0))
                total_var = nugget_var + signal_var
                if total_var > 0:
                    r2_ceiling = 1.0 - nugget_var / total_var
            else:
                nugget_var = float(cv_params.get("nugget", 0.0))
                psill_var  = float(cv_params.get("psill",  1.0))
                total_var = nugget_var + psill_var
                if total_var > 0:
                    r2_ceiling = 1.0 - nugget_var / total_var
        except (KeyError, TypeError):
            r2_ceiling = None

        # CV sanity check
        _data_std = float(np.std(Z_val))
        _sanity_ok = True
        if r2 < -0.1:
            logger.warning(f"\n  ⚠⚠  CV SANITY FAILURE: R² = {r2:.4g} is physically impossible.")
            logger.info(f"       This usually means NST inverse-transform produced extreme")
            logger.info(f"       values in one or more folds (Kriging overshoot + tail extrap).")
            logger.info(f"       CV metrics are unreliable for this run.")
            _sanity_ok = False
        if _data_std > 0 and rmse > 10.0 * _data_std:
            logger.warning(f"\n  ⚠⚠  CV SANITY FAILURE: RMSE = {rmse:.4g} is {rmse/_data_std:.1f}× "
                  f"the data std ({_data_std:.4g}).")
            logger.info(f"       Predictions are far worse than the global mean predictor.")
            _sanity_ok = False

        logger.info("\n       ── Cross-Validation Metrics ──")
        logger.info(f"       MAE        : {mae:.4f}")
        logger.info(f"       RMSE       : {rmse:.4f}")
        logger.info(f"       R²         : {r2:.4f}")
        if not _sanity_ok:
            logger.warning(f"       ⚠  Metrics above are flagged as unreliable (see warnings).")
        if r2_ceiling is not None:
            nugget_pct = (1.0 - r2_ceiling) * 100.0
            logger.info(f"       R² ceiling : {r2_ceiling:.4f}  "
                  f"(nugget accounts for {nugget_pct:.1f}% of total variance)")
            gap = r2_ceiling - r2
            if gap < 0.05:
                logger.info("       ✓ Engine is near the theoretical limit for this dataset.")
            elif gap < 0.20:
                logger.info(f"       ↑ Gap to ceiling: {gap:.4f} — moderate room for improvement.")
            else:
                logger.info(f"       ↑ Gap to ceiling: {gap:.4f} — consider data density / model selection.")

    # Export
    requested_formats = out_cfg.get("formats", None)

    if geo.is_point_mode:
        if len(geo.valid_points) == 0:
            raise RuntimeError(
                "No valid prediction points to export. This can happen if all "
                "custom prediction points lie outside the data envelope."
            )
        geo.df_out['Predicted_Mean'] = pred.point_pred_mean
        geo.df_out['Predicted_Std'] = pred.point_pred_std

        point_formats = requested_formats if requested_formats is not None else ["csv", "xz"]

        unknown_pt = set(point_formats) - _VALID_POINT_FORMATS
        if unknown_pt:
            raise ValueError(
                f"Unknown point-mode export format(s): {sorted(unknown_pt)}. "
                f"Valid options are: {sorted(_VALID_POINT_FORMATS)}"
            )

        if "csv" in point_formats:
            csv_path = out_dir / f"predicted_points_{mode}.csv"
            geo.df_out.to_csv(csv_path, index=False)
            logger.info(f"       ✓ predicted_points_{mode}.csv")

        if "xz" in point_formats:
            xz_path = out_dir / f"predicted_points_{mode}.xz"
            geo.df_out.to_pickle(xz_path)
            logger.info(f"       ✓ predicted_points_{mode}.xz")
    else:
        grid_formats = requested_formats if requested_formats is not None else ["nc"]
        z_dim = out_cfg.get("netcdf_z_dim_name", "Depth")
        export_grid(
            grid_formats,
            geo.X_grid, geo.Y_grid, pred.pred_mean, pred.pred_std,
            output_dir=out_dir,
            engine_name=mode,
            z_dim_name=z_dim,
        )

    logger.info(f"       ✓ parameters_{mode}.txt")
    logger.info(f"       ✓ parameters_{mode}.json")


# ── main pipeline ───────────────────────────────────────────────────────────

def run_pipeline():
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Spatial Interpolation Engine — kriging and GP backends",
    )
    parser.add_argument(
        "config", nargs="?", default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show debug-level output (Optuna trials, post-fit details)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress info-level output; show only warnings and errors",
    )
    args = parser.parse_args()

    # Configure logging
    if args.quiet:
        logging.basicConfig(level=logging.WARNING, format="%(message)s")
    elif args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        return

    config = load_config(config_path)
    out_dir = derive_output_dir(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    scenario_name = Path(config["input"]["filepath"]).stem

    logger.info("=" * 60)
    logger.info(f"  Interpolation Engine  –  {scenario_name}")
    logger.info("=" * 60)

    # 1. Load & validate
    data = _load_and_validate_data(config, out_dir)

    # 1.5. Preprocess
    pp = _preprocess_trend_and_nst(data, config)

    # 2. Geometry
    geo = _setup_prediction_targets(data, config)

    # 3. Ground truth visualisation
    _ground_truth_visualisation(data, out_dir)

    # 4. Create engine
    model, mode = _create_engine(data.X, pp.Z_fit, config)

    # 5. Fit
    params = _fit_model(model, data.X, pp.Z_fit, mode, out_dir)

    # 6. Predict
    pred = _predict(model, geo, pp)

    # 7. Diagnostics & export
    _export_and_diagnose(model, data, geo, pp, pred, params, mode, out_dir, config)

    logger.info("\n" + "=" * 60)
    logger.info(f"  All outputs saved to:  {out_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_pipeline()
