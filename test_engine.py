"""
test_engine.py — Fast comprehensive test suite for the 2D Kriging Engine.

Uses parameter caching to avoid redundant Optuna searches. Only 4 tests run
full hyperparameter optimisation; the remaining 16 use pre-computed parameters
via fit_with_known_params(). Total runtime: ~4 minutes (vs ~15 without caching).

Usage:
    cd 20260423_Interp_Engine
    /home/davidncu/miniconda3/envs/env_ds/bin/python test_engine.py
"""
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import warnings

sys.path.insert(0, str(Path(__file__).parent))

from src.engines.kriging import AnisotropicKriging
from src.preprocessor import TrendProcessor, NormalScoreTransform, check_normality

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent / "test_data"
CACHE_DIR = DATA_DIR / "params_cache"
PYTHON = "/home/davidncu/miniconda3/envs/env_ds/bin/python"
EPS = 1e-10

# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_scenario(name: str):
    f_samples = DATA_DIR / f"{name}.csv"
    f_truth   = DATA_DIR / f"{name}_ground_truth.csv"
    if not f_samples.exists():
        return None, None, None
    df_s = pd.read_csv(f_samples)
    X = df_s[["X", "Y"]].values.astype(np.float64)
    y = df_s["Value"].values.astype(np.float64)
    if f_truth.exists():
        df_t = pd.read_csv(f_truth)
        X_gt = df_t[["X", "Y"]].values.astype(np.float64)
        y_gt = df_t["Value"].values.astype(np.float64)
        return X, y, (X_gt, y_gt)
    return X, y, None

def get_cached_params(name: str, X, y, max_anisotropy=3.0, n_trials=100):
    """Load cached kriging parameters, or fit once and cache them."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache_path = CACHE_DIR / f"{name}.json"
    if cache_path.exists():
        data = json.loads(cache_path.read_text())
        return data["model_name"], data["params"]
    model = AnisotropicKriging(n_trials=n_trials, n_splits=5, max_anisotropy=max_anisotropy)
    model.fit(X, y)
    cache_path.write_text(json.dumps({
        "model_name": model.best_model_name_,
        "params": {k: v for k, v in model.best_params_.items()},
    }, indent=2, default=str))
    return model.best_model_name_, dict(model.best_params_)

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > EPS else 0.0

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

PASS = 0
FAIL = 0

def check(condition, msg, fatal=False):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {msg}")
    else:
        FAIL += 1
        print(f"  ✗ FAIL: {msg}")
        if fatal:
            raise AssertionError(msg)

def header(title):
    print(f"\n{'─'*60}\n  {title}\n{'─'*60}")

# ═══════════════════════════════════════════════════════════════════════════
# 1. Cached-param tests — use fit_with_known_params (fast)
# ═══════════════════════════════════════════════════════════════════════════

def test_isotropic_fit():
    header("1. Isotropic Kriging (S1 — 500 pts)")
    X, y, gt = load_scenario("S1_Isotropic")
    check(len(X) == 500, f"Loaded 500 points (got {len(X)})")

    name, params = get_cached_params("S1_Isotropic", X, y, max_anisotropy=3.0)
    model = AnisotropicKriging()
    model.fit_with_known_params(X, y, name, params)

    p = model.get_kernel_params()
    check(p["best_model"] is not None, f"Model: {p['best_model']}")
    check(p["range"] > 0, f"Range = {p['range']:.1f} > 0")
    check(p["psill"] > 0, f"Sill = {p['psill']:.3f} > 0")
    check(p["nugget"] >= 0, f"Nugget = {p['nugget']:.3f} >= 0")

    if gt is not None:
        X_gt, y_gt = gt
        y_pred, y_std = model.predict(X_gt, return_std=True)
        r2 = r2_score(y_gt, y_pred)
        check(r2 > 0.7, f"R² = {r2:.4f} > 0.7")
        check(np.all(np.isfinite(y_pred)), "All predictions finite")
        check(np.all(y_std >= 0), "All std >= 0")
        print(f"    RMSE={rmse(y_gt, y_pred):.4f}  R²={r2:.4f}")

def test_anisotropic_recovery():
    header("2. Anisotropic Recovery (S2 — angle=45°, ratio=2)")
    X, y, gt = load_scenario("S2_Anisotropic_45deg")
    check(len(X) == 500, f"Loaded 500 points (got {len(X)})")

    name, params = get_cached_params("S2_Anisotropic_45deg", X, y, max_anisotropy=5.0)
    model = AnisotropicKriging()
    model.fit_with_known_params(X, y, name, params)

    p = model.get_kernel_params()
    angle, ratio = p["rotation_angle_deg"], p["anisotropy_ratio"]
    check(15.0 <= angle <= 75.0, f"Angle = {angle:.1f}° ∈ [15,75]")
    check(ratio >= 1.3, f"Ratio = {ratio:.2f} ≥ 1.3")

def test_sparse_data():
    header("3. Sparse Data (S3 — 100 pts)")
    X, y, gt = load_scenario("S3_Sparse_Aniso_120deg")
    check(len(X) == 100, f"Loaded 100 points (got {len(X)})")

    name, params = get_cached_params("S3_Sparse_Aniso_120deg", X, y, max_anisotropy=5.0)
    model = AnisotropicKriging()
    model.fit_with_known_params(X, y, name, params)

    p = model.get_kernel_params()
    check(p["range"] > 0, f"Range = {p['range']:.1f} > 0")
    if gt is not None:
        X_gt, y_gt = gt
        y_pred, _ = model.predict(X_gt, return_std=True)
        check(r2_score(y_gt, y_pred) > 0.5, f"R² = {r2_score(y_gt, y_pred):.4f} > 0.5")

def test_high_nugget():
    header("4. High Nugget (S4)")
    X, y, _ = load_scenario("S4_HighNugget_Isotropic")
    check(len(X) == 500, f"Loaded 500 points (got {len(X)})")

    name, params = get_cached_params("S4_HighNugget_Isotropic", X, y, max_anisotropy=3.0)
    model = AnisotropicKriging()
    model.fit_with_known_params(X, y, name, params)

    p = model.get_kernel_params()
    check(p["nugget"] > 0, f"Nugget = {p['nugget']:.3f} > 0")
    check(p["psill"] > 0, f"Sill = {p['psill']:.3f} > 0")
    frac = p["nugget"] / (p["psill"] + p["nugget"] + EPS)
    print(f"    nugget={p['nugget']:.3f}  psill={p['psill']:.3f}  frac={frac:.3f}")

def test_extreme_anisotropy():
    header("5. Extreme Anisotropy (S5 — 10:1)")
    X, y, _ = load_scenario("S5_SGS_Extreme_Aniso")
    check(len(X) == 300, f"Loaded 300 points (got {len(X)})")

    name, params = get_cached_params("S5_SGS_Extreme_Aniso", X, y, max_anisotropy=15.0, n_trials=150)
    model = AnisotropicKriging()
    model.fit_with_known_params(X, y, name, params)

    p = model.get_kernel_params()
    check(p["anisotropy_ratio"] >= 3.0, f"Ratio = {p['anisotropy_ratio']:.2f} ≥ 3.0")
    print(f"    angle={p['rotation_angle_deg']:.1f}°  ratio={p['anisotropy_ratio']:.2f}")

def test_nested_structures():
    header("6. Nested Structures (S6)")
    X, y, gt = load_scenario("S6_SGS_Nested")
    check(len(X) == 400, f"Loaded 400 points (got {len(X)})")

    name, params = get_cached_params("S6_SGS_Nested", X, y, max_anisotropy=10.0, n_trials=150)
    model = AnisotropicKriging()
    model.fit_with_known_params(X, y, name, params)

    p = model.get_kernel_params()
    check(p["range"] > 0, f"Range = {p['range']:.1f} > 0")
    if gt is not None:
        X_gt, y_gt = gt
        y_pred, _ = model.predict(X_gt, return_std=True)
        check(r2_score(y_gt, y_pred) > 0.2, f"R² = {r2_score(y_gt, y_pred):.4f} > 0.2")

def test_clustered_sampling():
    header("7. Clustered Sampling (S7 — 200 pts, 2 clusters)")
    X, y, gt = load_scenario("S7_SGS_Clustered")
    check(len(X) == 200, f"Loaded 200 points (got {len(X)})")

    name, params = get_cached_params("S7_SGS_Clustered", X, y, max_anisotropy=5.0)
    model = AnisotropicKriging()
    model.fit_with_known_params(X, y, name, params)

    if gt is not None:
        X_gt, y_gt = gt
        y_pred, y_std = model.predict(X_gt, return_std=True)
        r2 = r2_score(y_gt, y_pred)
        check(r2 > 0.4, f"R² = {r2:.4f} > 0.4")
        check(y_pred.max() - y_pred.min() < (y.max() - y.min()) * 5.0,
              "Predictions bounded")
        check(np.all(np.isfinite(y_pred)), "All predictions finite")
        check(np.all(y_std > 0), "All std > 0")

# ═══════════════════════════════════════════════════════════════════════════
# 2. API correctness (some need full fits)
# ═══════════════════════════════════════════════════════════════════════════

def test_fit_with_known_params():
    header("8. fit_with_known_params — produces same predictions as fit()")
    X, y, _ = load_scenario("S1_Isotropic")

    # Full fit (this test specifically validates the full Optuna path)
    model1 = AnisotropicKriging(n_trials=80, n_splits=3, max_anisotropy=3.0)
    model1.fit(X, y)
    pred1, std1 = model1.predict(X[:10], return_std=True)

    # Shortcut fit
    model2 = AnisotropicKriging()
    model2.fit_with_known_params(X, y, model1.best_model_name_, dict(model1.best_params_))
    pred2, std2 = model2.predict(X[:10], return_std=True)

    check(model1.best_model_name_ == model2.best_model_name_,
          f"Models match: {model1.best_model_name_}")
    check(np.allclose(pred1, pred2, rtol=1e-5),
          "Predictions match between fit() and fit_with_known_params()")

def test_predict_shapes():
    header("9. Predict output shapes")
    X, y, _ = load_scenario("S1_Isotropic")
    name, params = get_cached_params("S1_Isotropic", X, y, max_anisotropy=3.0)
    model = AnisotropicKriging()
    model.fit_with_known_params(X, y, name, params)

    p1 = model.predict(X[:1])
    check(p1.shape == (1,), f"Single point shape: {p1.shape}")
    check(np.issubdtype(p1.dtype, np.floating), "Float dtype")

    p2, s2 = model.predict(X[:10], return_std=True)
    check(p2.shape == (10,), f"10-point shape: {p2.shape}")
    check(s2.shape == (10,), f"Std shape: {s2.shape}")

def test_get_kernel_params_keys():
    header("10. get_kernel_params returns all keys")
    X, y, _ = load_scenario("S1_Isotropic")
    name, params = get_cached_params("S1_Isotropic", X, y, max_anisotropy=3.0)
    model = AnisotropicKriging()
    model.fit_with_known_params(X, y, name, params)

    p = model.get_kernel_params()
    for key in ["model_type", "best_model", "rotation_angle_deg",
                "anisotropy_ratio", "psill", "range", "nugget"]:
        check(key in p, f"'{key}' present")
    check(p["model_type"] == "Kriging", "model_type is 'Kriging'")

# ═══════════════════════════════════════════════════════════════════════════
# 3. Edge Cases — use cached params
# ═══════════════════════════════════════════════════════════════════════════

def test_few_points():
    header("11. Very Few Points (S9 — n=15)")
    X, y, _ = load_scenario("S9_FewPoints")
    check(len(X) == 15, f"Loaded 15 points (got {len(X)})")

    name, params = get_cached_params("S9_FewPoints", X, y, max_anisotropy=3.0, n_trials=80)
    model = AnisotropicKriging()
    model.fit_with_known_params(X, y, name, params)

    X_test = np.column_stack([
        np.random.default_rng(99).uniform(X[:, 0].min(), X[:, 0].max(), 50),
        np.random.default_rng(99).uniform(X[:, 1].min(), X[:, 1].max(), 50),
    ])
    pred, _ = model.predict(X_test, return_std=True)
    check(np.all(np.isfinite(pred)), "All predictions finite with n=15")

def test_duplicate_coordinates():
    header("12. Duplicate Coordinates (S10)")
    X, y, _ = load_scenario("S10_Duplicates")
    from scipy.spatial.distance import pdist, squareform
    dist_mat = squareform(pdist(X))
    n_exact = 0
    visited = np.zeros(len(X), dtype=bool)
    for i in range(len(X)):
        if visited[i]: continue
        exact_group = np.where(dist_mat[i] == 0.0)[0]
        if len(exact_group) > 1: n_exact += len(exact_group) - 1
        visited[exact_group] = True
    check(n_exact > 0, f"Detected {n_exact} exact duplicates")

    # Clean duplicates
    visited = np.zeros(len(X), dtype=bool)
    keep = np.ones(len(X), dtype=bool)
    y_c = y.copy()
    for i in range(len(X)):
        if visited[i]: continue
        exact_group = np.where(dist_mat[i] == 0.0)[0]
        if len(exact_group) > 1:
            y_c[exact_group[0]] = np.mean(y_c[exact_group])
            keep[exact_group[1:]] = False
        visited[exact_group] = True
    X_c, y_c = X[keep], y_c[keep]
    check(len(X_c) < len(X), f"Cleaned: {len(X_c)} of {len(X)} pts remain")

    name, params = get_cached_params("S10_Duplicates", X_c, y_c, max_anisotropy=3.0, n_trials=80)
    model = AnisotropicKriging()
    model.fit_with_known_params(X_c, y_c, name, params)
    check(True, "Fit succeeded on cleaned duplicate data")

def test_colinear_points():
    header("13. Nearly Colinear Points (S11)")
    X, y, _ = load_scenario("S11_Colinear")
    check(len(X) == 30, f"Loaded 30 points (got {len(X)})")

    X_ctr = X - X.mean(axis=0)
    eigvals = np.linalg.eigvalsh(X_ctr.T @ X_ctr / len(X))
    spread_ratio = eigvals[0] / max(eigvals[1], EPS)
    check(spread_ratio < 0.01, f"Near-colinear: spread_ratio={spread_ratio:.6f}")

    name, params = get_cached_params("S11_Colinear", X, y, max_anisotropy=3.0)
    model = AnisotropicKriging()
    model.fit_with_known_params(X, y, name, params)
    check(True, "Fit succeeded on nearly colinear data")

def test_lognormal_nst():
    header("14. Log-Normal Data (S12 — NST test)")
    X, y, gt = load_scenario("S12_LogNormal")
    norm_stats = check_normality(y)
    check(not norm_stats["is_normal"], f"SW rejects normality: p={norm_stats['shapiro_p']:.2e}")
    check(norm_stats["recommend_nst"], "NST recommended")

    # Without NST
    name_no, params_no = get_cached_params("S12_LogNormal", X, y, max_anisotropy=3.0)
    model_no = AnisotropicKriging()
    model_no.fit_with_known_params(X, y, name_no, params_no)

    # With NST
    nst = NormalScoreTransform(tail_extrapolation=True)
    y_norm = nst.fit_transform(y)
    name_nst, params_nst = get_cached_params("S12_LogNormal_nst", X, y_norm, max_anisotropy=3.0)
    model_nst = AnisotropicKriging()
    model_nst.fit_with_known_params(X, y_norm, name_nst, params_nst)

    if gt is not None:
        X_gt, y_gt = gt
        pred_no, _ = model_no.predict(X_gt, return_std=True)
        pred_nst_n, _ = model_nst.predict(X_gt, return_std=True)
        pred_nst = nst.inverse_transform(pred_nst_n)
        r2_no, r2_nst = r2_score(y_gt, pred_no), r2_score(y_gt, pred_nst)
        print(f"    Without NST: R²={r2_no:.4f}  With NST: R²={r2_nst:.4f}")
        check(r2_nst > 0.3, f"With NST: R²={r2_nst:.4f} > 0.3")

def test_strong_trend():
    header("15. Strong Trend (S13)")
    X, y, gt = load_scenario("S13_StrongTrend")
    from src.preprocessor import analyze_trend
    trend_stats = analyze_trend(X[:, 0], X[:, 1], y, order=1)
    check(trend_stats["recommend_detrend"],
          f"Trend: F-p={trend_stats['f_pvalue']:.2e}, R²={trend_stats['r2']:.3f}")

    tp = TrendProcessor(order=1)
    tp.fit(X[:, 0], X[:, 1], y)
    y_res = tp.detrend(X[:, 0], X[:, 1], y)

    name, params = get_cached_params("S13_StrongTrend", X, y_res, max_anisotropy=3.0)
    model = AnisotropicKriging()
    model.fit_with_known_params(X, y_res, name, params)

    if gt is not None:
        X_gt, y_gt = gt
        y_pred_res, _ = model.predict(X_gt, return_std=True)
        y_pred = tp.retrend(X_gt[:, 0], X_gt[:, 1], y_pred_res)
        r2 = r2_score(y_gt, y_pred)
        check(r2 > 0.5, f"R² = {r2:.4f} > 0.5 (detrended kriging)")
        print(f"    R²={r2:.4f}  RMSE={rmse(y_gt, y_pred):.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# 4. Pure NST tests (no fitting needed)
# ═══════════════════════════════════════════════════════════════════════════

def test_nst_roundtrip():
    header("16. NST Roundtrip Fidelity")
    rng = np.random.default_rng(42)
    x_orig = rng.lognormal(mean=0.0, sigma=1.5, size=500)
    nst = NormalScoreTransform(tail_extrapolation=True)
    z = nst.fit_transform(x_orig)
    x_r = nst.inverse_transform(z)
    check(np.all(np.diff(z[np.argsort(x_orig)]) >= -1e-10), "NST monotone")
    check(np.allclose(np.argsort(np.argsort(x_orig)), np.argsort(np.argsort(x_r))),
          "NST rank-preserving")
    check(abs(np.mean(z)) < 0.1, f"z mean ≈ 0 (got {np.mean(z):.4f})")
    check(abs(np.std(z) - 1.0) < 0.15, f"z std ≈ 1 (got {np.std(z):.4f})")

def test_nst_small_sample():
    header("17. NST with Minimum Sample Size")
    nst = NormalScoreTransform()
    try:
        nst.fit(np.array([1.0, 2.0]))
        check(False, "NST should reject n=2")
    except ValueError as e:
        check("at least 3" in str(e), f"NST correctly rejects n=2")
    nst.fit(np.array([1.0, 2.0, 3.0]))
    check(nst.is_fitted, "NST fits with n=3")

# ═══════════════════════════════════════════════════════════════════════════
# 5. Tests that need full Optuna search
# ═══════════════════════════════════════════════════════════════════════════

def test_max_anisotropy_bound():
    header("18. max_anisotropy Bound Enforced")
    X, y, _ = load_scenario("S14_ExtremeAniso")

    model_t = AnisotropicKriging(n_trials=80, n_splits=3, max_anisotropy=3.0)
    model_t.fit(X, y)
    check(model_t.get_kernel_params()["anisotropy_ratio"] <= 3.0 + EPS,
          "Tight bound (3.0) enforced")

    name_l, params_l = get_cached_params("S14_ExtremeAniso", X, y, max_anisotropy=20.0, n_trials=150)
    model_l = AnisotropicKriging()
    model_l.fit_with_known_params(X, y, name_l, params_l)
    check(model_l.get_kernel_params()["anisotropy_ratio"] >= 3.0,
          f"Loose bound allows high ratio: {model_l.get_kernel_params()['anisotropy_ratio']:.2f}")
    print(f"    tight max=3.0 → ratio={model_t.get_kernel_params()['anisotropy_ratio']:.2f}")
    print(f"    loose max=20.0 → ratio={model_l.get_kernel_params()['anisotropy_ratio']:.2f}")

def test_all_models_trial():
    header("19. All Variogram Models Explored (full fit)")
    X, y, _ = load_scenario("S1_Isotropic")
    from src.engines.kriging import AnisotropicKriging as AK
    all_models = AK.NATIVE_MODELS + list(AK.CUSTOM_MODELS.keys())
    print(f"    Available: {all_models}")

    model = AnisotropicKriging(n_trials=100, n_splits=5, max_anisotropy=3.0)
    model.fit(X, y)
    check(model.best_model_name_ in all_models,
          f"Chose '{model.best_model_name_}' from known models")

def test_full_pipeline_kriging():
    header("20. Full Pipeline — Kriging End-to-End")
    import subprocess, tempfile, yaml, shutil

    tmp_dir = Path(tempfile.mkdtemp(prefix="krg_test_"))
    try:
        cfg = {
            "input": {"filepath": str(DATA_DIR / "S1_Isotropic.csv"), "format": "csv",
                       "columns": {"x": "X", "y": "Y", "value": "Value"},
                       "ground_truth_filepath": str(DATA_DIR / "S1_Isotropic_ground_truth.csv")},
            "geometry": {"resolution_m": 50.0, "convex_hull_buffer_percent": 10.0},
            "preprocessing": {"detrend": {"auto_detect": False, "enabled": False, "order": 1},
                              "nst": {"enabled": False},
                              "duplicates": {"min_separation": None}},
            "engine": {"mode": "kriging",
                        "kriging": {"max_anisotropy": 3.0, "n_splits": 3, "n_trials": 50}},
            "output": {"base_directory": str(tmp_dir / "output"),
                        "save_diagnostics": True, "formats": ["csv"]},
        }
        cfg_path = tmp_dir / "test_config.yaml"
        with open(cfg_path, "w") as f: yaml.dump(cfg, f)

        result = subprocess.run(
            [PYTHON, "main.py", str(cfg_path)],
            capture_output=True, text=True, timeout=360,
            cwd=str(Path(__file__).parent),
        )
        check(result.returncode == 0, f"Pipeline exit code {result.returncode}")
        if result.returncode != 0:
            for l in (result.stdout + result.stderr).split("\n")[-15:]:
                if l.strip(): print(f"    [pipe] {l}")

        out_dir = tmp_dir / "output" / "S1_Isotropic"
        check((out_dir / "parameters_kriging.json").exists(), "parameters_kriging.json exists")
        check((out_dir / "cv_results_kriging.csv").exists(), "cv_results_kriging.csv exists")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t_start = time.time()
    print("=" * 60)
    print("  Kriging Engine Test Suite (with param caching)")
    print(f"  Python: {PYTHON}")
    print("=" * 60)

    tests = [
        test_isotropic_fit, test_anisotropic_recovery, test_sparse_data,
        test_high_nugget, test_extreme_anisotropy, test_nested_structures,
        test_clustered_sampling,
        test_fit_with_known_params, test_predict_shapes, test_get_kernel_params_keys,
        test_few_points, test_duplicate_coordinates, test_colinear_points,
        test_lognormal_nst, test_strong_trend,
        test_nst_roundtrip, test_nst_small_sample,
        test_max_anisotropy_bound, test_all_models_trial,
        test_full_pipeline_kriging,
    ]

    for test in tests:
        t0 = time.time()
        try:
            test()
        except Exception as e:
            FAIL += 1
            print(f"  ✗ CRASH: {test.__name__}: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
        dt = time.time() - t0
        print(f"    [{dt:.1f}s]")

    total = PASS + FAIL
    elapsed = time.time() - t_start
    print(f"\n{'═'*60}")
    print(f"  Results: {PASS}/{total} passed", end="")
    if FAIL > 0: print(f", {FAIL} FAILED", end="")
    print(f"  |  {elapsed:.0f}s total")
    print(f"{'═'*60}")
    sys.exit(0 if FAIL == 0 else 1)
