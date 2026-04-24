"""
main.py - Entry point for the standalone Interpolation Engine.

Usage:
    conda run -n fafalab python main.py
    python main.py

Output files are alphabetically prefixed so that the natural sort order
matches the logical viewing sequence:

    A_ground_truth.png        (only if ground_truth_filepath is set)
    B_convex_hull.png
    C_variogram_omni.png
    D_variogram_directional.png
    E_anisotropy_ellipse.png
    F_prediction_surface.png
    G_comparison.png          (only if ground_truth_filepath is set)
    H_cv_dashboard.png
    cv_results.csv
    parameters.txt / .json
    predicted_<engine>.nc
"""
import yaml
import json
from pathlib import Path
import numpy as np
import pandas as pd
import time

from src.data_loader import load_input_data
from src.geometry import generate_prediction_grid
from src.exporter import export_to_netcdf
from src.engines.gp import RotatedGPR
from src.engines.kriging import AnisotropicKriging
from src.preprocessor import TrendProcessor
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
from sklearn.gaussian_process.kernels import RBF, ConstantKernel


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
    with open(out_dir / "parameters.txt", "w") as f:
        f.write(f"Engine : {mode.upper()}\n")
        f.write("=" * 50 + "\n")
        for k, v in params.items():
            f.write(f"{k:30s}: {v}\n")

    with open(out_dir / "parameters.json", "w") as f:
        json.dump({"engine": mode, **params}, f, indent=2, default=str)


# ── main pipeline ─────────────────────────────────────────────────────────

def run_pipeline():
    config_path = Path("config.yaml")
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

    gt = load_ground_truth(config)
    if gt is not None:
        gt_X, gt_Y, gt_Z = gt
        print(f"       {len(gt_Z)} ground-truth points loaded.")

    # ── 1.5. Preprocessing (Detrending) ───────────────────────────────
    pre_cfg = config.get("preprocessing", {}).get("detrend", {})
    do_detrend = pre_cfg.get("enabled", False)
    trend_order = pre_cfg.get("order", 1)

    if do_detrend:
        print(f"[1.5/7] Detrending (order={trend_order}) ...")
        processor = TrendProcessor(order=trend_order)
        processor.fit(X_coord, Y_coord, Z_val)
        Z_fit = processor.detrend(X_coord, Y_coord, Z_val)

        t_params = processor.get_params()
        print(f"       Trend fitted. Intercept: {t_params['intercept']:.3f}")
    else:
        processor = None
        Z_fit = Z_val

    # ── 2. Geometry ───────────────────────────────────────────────────
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
        ls = gp_cfg.get("length_scale", 500.0)
        ls_bounds = (
            gp_cfg.get("length_scale_min", 10.0),
            gp_cfg.get("length_scale_max", 5000.0),
        )
        kernel = ConstantKernel(
            constant_value=np.var(Z_fit),
            constant_value_bounds=(np.var(Z_fit) * 0.1, np.var(Z_fit) * 10),
        ) * RBF(length_scale=[ls, ls], length_scale_bounds=ls_bounds)

        model = RotatedGPR(
            kernel=kernel,
            n_restarts_optimizer=gp_cfg.get("n_restarts", 10),
            angle_search_method=gp_cfg.get("angle_search", "bounded"),
            max_anisotropy=gp_cfg.get("max_anisotropy", 10.0),
            angle_bounds=(
                gp_cfg.get("angle_min", 0.0),
                gp_cfg.get("angle_max", 180.0),
            ),
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

    # ── 6. Predict on grid ────────────────────────────────────────────
    print("[6/7] Predicting over spatial grid ...")
    pred_mean = np.full(grid_shape, np.nan)
    pred_std = np.full(grid_shape, np.nan)

    grid_points = np.column_stack((X_grid.flatten(), Y_grid.flatten()))
    valid_points = grid_points[mask]

    if len(valid_points) > 0:
        means, stds = model.predict(valid_points, return_std=True)
        if processor is not None:
            means = processor.retrend(valid_points[:, 0], valid_points[:, 1], means)
        pred_mean.flat[mask] = means
        pred_std.flat[mask] = stds

    # ── 7. Diagnostics & export ───────────────────────────────────────
    print("[7/7] Generating outputs ...")

    # B_ Convex hull
    plot_convex_hull(
        X_coord, Y_coord, Z_val,
        hull_verts, X_grid, Y_grid, mask,
        scenario_name=scenario_name,
        save_path=out_dir / "B_convex_hull.png",
    )
    print("       ✓ B_convex_hull.png")

    if processor is not None and out_cfg.get("save_diagnostics", True):
        plot_trend_components(
            X_coord, Y_coord, Z_val, Z_fit, processor,
            X_grid, Y_grid, mask, hull_verts,
            scenario_name=scenario_name,
            save_path=out_dir / "Trend_components.png"
        )
        print("       ✓ Trend_components.png")

    if out_cfg.get("save_diagnostics", True):
        # C_ Omnidirectional variogram
        omni_var = compute_empirical_variogram(X, Z_fit)
        plot_variogram(
            omni_var,
            fitted_params=params,
            engine_name=mode.upper(),
            scenario_name=scenario_name,
            save_path=out_dir / "C_variogram_omni.png",
        )
        print("       ✓ C_variogram_omni.png")

        # D_ Directional variogram (15° intervals)
        directions = np.arange(0, 180, 15)
        dir_vars = compute_empirical_variogram(X, Z_fit, directions=directions)
        plot_directional_variogram(
            dir_vars,
            scenario_name=scenario_name,
            save_path=out_dir / "D_variogram_directional.png",
        )
        print("       ✓ D_variogram_directional.png")

        # E_ Anisotropy ellipse
        plot_anisotropy_ellipse(
            params,
            engine_name=mode.upper(),
            scenario_name=scenario_name,
            save_path=out_dir / "E_anisotropy_ellipse.png",
        )
        print("       ✓ E_anisotropy_ellipse.png")

    # F_ Prediction surface
    plot_prediction_surface(
        X_grid, Y_grid, pred_mean, pred_std,
        X_obs=X_coord, Y_obs=Y_coord,
        hull_vertices=hull_verts,
        scenario_name=scenario_name,
        engine_name=mode.upper(),
        save_path=out_dir / "F_prediction_surface.png",
    )
    print("       ✓ F_prediction_surface.png")

    # G_ Comparison (only if ground truth available)
    if gt is not None:
        plot_comparison(
            X_grid, Y_grid, pred_mean,
            gt_X, gt_Y, gt_Z,
            hull_vertices=hull_verts,
            scenario_name=scenario_name,
            engine_name=mode.upper(),
            save_path=out_dir / "G_comparison.png",
        )
        print("       ✓ G_comparison.png")

    # H_ Cross-validation dashboard
    if out_cfg.get("save_diagnostics", True):
        print("       Running cross-validation ...")
        if mode == "gp":
            cv_df = perform_gpr_kfold_cv(model, X, Z_fit)
        else:
            cv_df = perform_kriging_kfold_cv(model, X, Z_fit)
        cv_df.to_csv(out_dir / "cv_results.csv", index=False)
        plot_cv_dashboard(
            cv_df,
            engine_name=mode.upper(),
            scenario_name=scenario_name,
            save_path=out_dir / "H_cv_dashboard.png",
        )
        print("       ✓ H_cv_dashboard.png")
        print("       ✓ cv_results.csv")

    # NetCDF
    z_dim = out_cfg.get("netcdf_z_dim_name", "Depth")
    export_to_netcdf(
        X_grid, Y_grid, pred_mean, pred_std,
        output_dir=out_dir,
        engine_name=mode,
        z_dim_name=z_dim,
    )
    print("       ✓ predicted_{}.nc".format(mode))

    # Parameters (already saved earlier, just confirm)
    print("       ✓ parameters.txt")
    print("       ✓ parameters.json")

    print("\n" + "=" * 60)
    print(f"  All outputs saved to:  {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
