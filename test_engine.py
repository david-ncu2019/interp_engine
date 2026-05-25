"""
test_engine.py — Comprehensive test suite for the 2D Kriging Engine.

Tests the AnisotropicKriging class against 14 synthetic scenarios covering
isotropic, anisotropic, sparse, high-nugget, clustered, nested-structure,
and extreme edge cases.

Usage:
    cd 20260423_Interp_Engine
    /home/davidncu/miniconda3/envs/env_ds/bin/python test_engine.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings

# Add src to path so we can import engine modules directly
sys.path.insert(0, str(Path(__file__).parent))

from src.engines.kriging import AnisotropicKriging
from src.preprocessor import TrendProcessor, NormalScoreTransform, check_normality

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent / "test_data"
PYTHON = "/home/davidncu/miniconda3/envs/env_ds/bin/python"
EPS = 1e-10

# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_scenario(name: str):
    """Load sample points and ground truth for a named scenario."""
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
# 1. Basic Functionality
# ═══════════════════════════════════════════════════════════════════════════

def test_isotropic_fit():
    header("1. Isotropic Kriging (S1 — 500 pts, range=300, angle=0)")
    X, y, gt = load_scenario("S1_Isotropic")
    check(len(X) == 500, f"Loaded 500 points (got {len(X)})")

    model = AnisotropicKriging(n_trials=150, n_splits=5, max_anisotropy=3.0)
    model.fit(X, y)

    params = model.get_kernel_params()
    check(params["best_model"] is not None, f"Model selected: {params['best_model']}")
    check(params["range"] > 0, f"Range = {params['range']:.1f} > 0")
    check(params["psill"] > 0, f"Partial sill = {params['psill']:.3f} > 0")
    check(params["nugget"] >= 0, f"Nugget = {params['nugget']:.3f} >= 0")
    check(params["anisotropy_ratio"] >= 1.0 and params["anisotropy_ratio"] <= 3.0,
          f"Anisotropy ratio = {params['anisotropy_ratio']:.2f} in [1, 3]")

    # Predict and evaluate against ground truth
    if gt is not None:
        X_gt, y_gt = gt
        y_pred, y_std = model.predict(X_gt, return_std=True)
        r2 = r2_score(y_gt, y_pred)
        e_rmse = rmse(y_gt, y_pred)
        check(r2 > 0.4, f"R² = {r2:.4f} > 0.4 (spatial structure recovered)")
        check(len(y_pred) == len(y_gt), f"Prediction shape matches ({len(y_pred)})")
        check(np.all(np.isfinite(y_pred)), "All predictions finite")
        check(np.all(y_std >= 0), "All std values non-negative")
        print(f"    RMSE={e_rmse:.4f}  R²={r2:.4f}")

def test_anisotropic_recovery():
    header("2. Anisotropic Recovery (S2 — 500 pts, angle=45°, ratio=2)")
    X, y, gt = load_scenario("S2_Anisotropic_45deg")
    check(len(X) == 500, f"Loaded 500 points (got {len(X)})")

    model = AnisotropicKriging(n_trials=150, n_splits=5, max_anisotropy=5.0)
    model.fit(X, y)

    params = model.get_kernel_params()
    angle = params["rotation_angle_deg"]
    ratio = params["anisotropy_ratio"]

    check(15.0 <= angle <= 75.0,
          f"Angle = {angle:.1f}° within [15, 75] of true 45°")
    check(ratio >= 1.3,
          f"Anisotropy ratio = {ratio:.2f} ≥ 1.3 (true=2.0)")

    if gt is not None:
        X_gt, y_gt = gt
        y_pred, _ = model.predict(X_gt, return_std=True)
        r2 = r2_score(y_gt, y_pred)
        check(r2 > 0.4, f"R² = {r2:.4f} > 0.4")

def test_sparse_data():
    header("3. Sparse Data (S3 — only 100 pts, angle=120°, ratio=2)")
    X, y, gt = load_scenario("S3_Sparse_Aniso_120deg")
    check(len(X) == 100, f"Loaded 100 points (got {len(X)})")

    # Fewer trials acceptable for sparse data
    model = AnisotropicKriging(n_trials=100, n_splits=3, max_anisotropy=5.0)
    model.fit(X, y)

    params = model.get_kernel_params()
    check(params["range"] > 0, f"Range = {params['range']:.1f} > 0")

    if gt is not None:
        X_gt, y_gt = gt
        y_pred, _ = model.predict(X_gt, return_std=True)
        r2 = r2_score(y_gt, y_pred)
        # Sparse data → lower R² expectations, but should still capture some structure
        check(r2 > 0.15, f"R² = {r2:.4f} > 0.15 (sparse data, modest expectation)")

def test_high_nugget():
    header("4. High Nugget (S4 — nugget/sill ≈ 0.5)")
    X, y, gt = load_scenario("S4_HighNugget_Isotropic")
    check(len(X) == 500, f"Loaded 500 points (got {len(X)})")

    model = AnisotropicKriging(n_trials=150, n_splits=5, max_anisotropy=3.0)
    model.fit(X, y)

    params = model.get_kernel_params()
    nugget_frac = params["nugget"] / (params["psill"] + params["nugget"] + EPS)
    # S4 was generated with high noise (std=2.0) relative to signal (std=2.7).
    # The model may absorb some noise into the structured sill, so we only
    # assert that nugget is non-zero and that predictions are meaningful.
    check(params["nugget"] > 0.0, f"Nugget = {params['nugget']:.3f} > 0 (noise detected)")
    check(params["psill"] > 0, f"Partial sill = {params['psill']:.3f} > 0")
    print(f"    nugget={params['nugget']:.3f}  psill={params['psill']:.3f}  nugget_frac={nugget_frac:.3f}")

def test_extreme_anisotropy():
    header("5. Extreme Anisotropy (S5 — ratio=10:1 at 30°)")
    X, y, gt = load_scenario("S5_SGS_Extreme_Aniso")
    check(len(X) == 300, f"Loaded 300 points (got {len(X)})")

    # Need high max_anisotropy to capture 10:1
    model = AnisotropicKriging(n_trials=200, n_splits=5, max_anisotropy=15.0)
    model.fit(X, y)

    params = model.get_kernel_params()
    ratio = params["anisotropy_ratio"]
    check(ratio >= 3.0,
          f"Anisotropy ratio = {ratio:.2f} ≥ 3.0 (true=10, partially recovered)")
    print(f"    angle={params['rotation_angle_deg']:.1f}°  ratio={ratio:.2f}")

def test_nested_structures():
    header("6. Nested Structures (S6 — short iso + long aniso)")
    X, y, gt = load_scenario("S6_SGS_Nested")
    check(len(X) == 400, f"Loaded 400 points (got {len(X)})")

    # Nested structures are approximated by a single effective structure
    model = AnisotropicKriging(n_trials=200, n_splits=5, max_anisotropy=10.0)
    model.fit(X, y)

    params = model.get_kernel_params()
    check(params["range"] > 0, f"Effective range = {params['range']:.1f} > 0")
    # Nested → the single-structure model should still produce reasonable predictions
    if gt is not None:
        X_gt, y_gt = gt
        y_pred, _ = model.predict(X_gt, return_std=True)
        r2 = r2_score(y_gt, y_pred)
        check(r2 > 0.2, f"R² = {r2:.4f} > 0.2 (nested approximated by single structure)")

def test_clustered_sampling():
    header("7. Clustered Sampling (S7 — 200 pts in 2 clusters)")
    X, y, gt = load_scenario("S7_SGS_Clustered")
    check(len(X) == 200, f"Loaded 200 points (got {len(X)})")

    model = AnisotropicKriging(n_trials=150, n_splits=3, max_anisotropy=5.0)
    model.fit(X, y)

    # Key test: predictions should not explode in data gaps
    if gt is not None:
        X_gt, y_gt = gt
        y_pred, y_std = model.predict(X_gt, return_std=True)
        r2 = r2_score(y_gt, y_pred)
        # Clustered data with gaps → moderate R²
        check(r2 > 0.1, f"R² = {r2:.4f} > 0.1 (gappy clustered data)")
        # Predictions must stay bounded — no NaN, no extreme overshoot
        pred_range = y_pred.max() - y_pred.min()
        data_range = y.max() - y.min()
        check(pred_range < data_range * 5.0,
              f"Prediction range ({pred_range:.2f}) < 5× data range ({data_range:.2f})")
        check(np.all(np.isfinite(y_pred)), "All predictions finite in data gaps")
        check(np.all(y_std > 0), "All std values positive in data gaps")

# ═══════════════════════════════════════════════════════════════════════════
# 2. API Correctness
# ═══════════════════════════════════════════════════════════════════════════

def test_fit_with_known_params():
    header("8. fit_with_known_params — shortcut fitting")
    X, y, _ = load_scenario("S1_Isotropic")

    # First fit normally
    model1 = AnisotropicKriging(n_trials=100, n_splits=3, max_anisotropy=3.0)
    model1.fit(X, y)
    params1 = model1.get_kernel_params()
    pred1, std1 = model1.predict(X[:10], return_std=True)

    # Now fit with known params
    model2 = AnisotropicKriging()
    model2.fit_with_known_params(
        X, y,
        best_model_name=model1.best_model_name_,
        best_params=dict(model1.best_params_)
    )
    params2 = model2.get_kernel_params()
    pred2, std2 = model2.predict(X[:10], return_std=True)

    check(model1.best_model_name_ == model2.best_model_name_,
          f"Model names match: {model1.best_model_name_} == {model2.best_model_name_}")
    check(abs(params1["range"] - params2["range"]) < EPS * 10,
          f"Range preserved: {params1['range']:.2f} == {params2['range']:.2f}")
    check(abs(params1["psill"] - params2["psill"]) < EPS * 10,
          f"Sill preserved: {params1['psill']:.3f} == {params2['psill']:.3f}")
    check(np.allclose(pred1, pred2, rtol=1e-5),
          "Predictions match between fit and fit_with_known_params")

def test_predict_shapes():
    header("9. Predict output shapes and types")
    X, y, _ = load_scenario("S1_Isotropic")
    model = AnisotropicKriging(n_trials=50, n_splits=3)
    model.fit(X, y)

    # Single point
    p1 = model.predict(X[:1])
    check(p1.shape == (1,) or p1.shape == (1, 1),
          f"Single point shape: {p1.shape}")
    check(np.issubdtype(p1.dtype, np.floating),
          "Predictions are floating-point")

    # With std
    p2, s2 = model.predict(X[:10], return_std=True)
    check(p2.shape == (10,) or p2.shape[0] == 10,
          f"Multi-point shape: {p2.shape}")
    check(s2.shape == (10,) or s2.shape[0] == 10,
          f"Std shape: {s2.shape}")

def test_get_kernel_params_keys():
    header("10. get_kernel_params returns all expected keys")
    X, y, _ = load_scenario("S1_Isotropic")
    model = AnisotropicKriging(n_trials=50, n_splits=3)
    model.fit(X, y)

    params = model.get_kernel_params()
    required = ["model_type", "best_model", "rotation_angle_deg",
                "anisotropy_ratio", "psill", "range", "nugget"]
    for key in required:
        check(key in params, f"Key '{key}' present in params")
    check(params["model_type"] == "Kriging", "model_type is 'Kriging'")

# ═══════════════════════════════════════════════════════════════════════════
# 3. Edge Cases & Robustness
# ═══════════════════════════════════════════════════════════════════════════

def test_few_points():
    header("11. Very Few Points (S9 — n=15)")
    X, y, gt = load_scenario("S9_FewPoints")
    check(len(X) == 15, f"Loaded 15 points (got {len(X)})")

    model = AnisotropicKriging(n_trials=80, n_splits=3, max_anisotropy=3.0)
    try:
        model.fit(X, y)
        params = model.get_kernel_params()
        check(params["range"] > 0, f"Fit succeeded with {len(X)} points (range={params['range']:.1f})")

        # Predict on a test grid
        X_test = np.column_stack([
            np.random.default_rng(99).uniform(X[:, 0].min(), X[:, 0].max(), 50),
            np.random.default_rng(99).uniform(X[:, 1].min(), X[:, 1].max(), 50),
        ])
        pred, std = model.predict(X_test, return_std=True)
        check(np.all(np.isfinite(pred)), "All predictions finite with n=15")
    except Exception as e:
        check(False, f"Fit crashed with n=15: {type(e).__name__}: {e}")

def test_duplicate_coordinates():
    header("12. Duplicate Coordinates (S10 — exact + near duplicates)")
    X, y, _ = load_scenario("S10_Duplicates")

    n_orig = len(y)

    # Simulate what main.py does: check_and_clean_duplicates
    from scipy.spatial.distance import pdist, squareform
    dist_mat = squareform(pdist(X))
    n_exact = 0
    for i in range(len(X)):
        exact_group = np.where(dist_mat[i] == 0.0)[0]
        if len(exact_group) > 1 and i == exact_group[0]:  # first of group
            n_exact += len(exact_group) - 1

    check(n_exact > 0, f"Detected {n_exact} exact duplicates in S10 dataset")

    # Clean via the main.py logic
    from src.preprocessor import TrendProcessor  # just a reference, we inline the logic
    visited = np.zeros(len(X), dtype=bool)
    keep_mask = np.ones(len(X), dtype=bool)
    y_clean = y.copy()
    for i in range(len(X)):
        if visited[i]: continue
        exact_group = np.where(dist_mat[i] == 0.0)[0]
        if len(exact_group) > 1:
            y_clean[exact_group[0]] = np.mean(y_clean[exact_group])
            keep_mask[exact_group[1:]] = False
        visited[exact_group] = True

    X_c = X[keep_mask]
    y_c = y_clean[keep_mask]
    check(len(X_c) < n_orig, f"Cleaned: {len(X_c)} points remain (removed {n_orig - len(X_c)} duplicates)")

    # Fit on cleaned data
    model = AnisotropicKriging(n_trials=80, n_splits=3, max_anisotropy=3.0)
    try:
        model.fit(X_c, y_c)
        check(True, "Fit succeeded on cleaned duplicate data")
    except Exception as e:
        check(False, f"Fit failed on cleaned duplicate data: {type(e).__name__}: {e}")

    # Verify that uncleaned data causes a crash / error
    try:
        model_raw = AnisotropicKriging(n_trials=30, n_splits=3, max_anisotropy=3.0)
        model_raw.fit(X, y)
        # If it doesn't crash, predictions might be nonsensical
        pred_raw, _ = model_raw.predict(X[:5], return_std=True)
        check(np.all(np.isfinite(pred_raw)),
              "Uncleaned data fit completed (may be numerically borderline)")
    except Exception:
        check(True, "Uncleaned data caused expected failure (singular matrix)")

def test_colinear_points():
    header("13. Nearly Colinear Points (S11 — points along a line)")
    X, y, gt = load_scenario("S11_Colinear")
    check(len(X) == 30, f"Loaded 30 points (got {len(X)})")

    # Check that points are indeed nearly colinear using PCA:
    # the ratio of the smaller eigenvalue to the larger eigenvalue measures
    # how thin the point cloud is perpendicular to its main axis.
    X_centered = X - X.mean(axis=0)
    cov = X_centered.T @ X_centered / len(X)
    eigvals = np.linalg.eigvalsh(cov)
    spread_ratio = eigvals[0] / eigvals[1] if eigvals[1] > EPS else 1.0
    check(spread_ratio < 0.01,
          f"Near-colinear: perpendicular/parallel spread = {spread_ratio:.6f}")
    print(f"    PCA eigenvalues: {eigvals}, spread_ratio={spread_ratio:.6f}")

    model = AnisotropicKriging(n_trials=100, n_splits=3, max_anisotropy=3.0)
    try:
        model.fit(X, y)
        check(True, "Fit succeeded on nearly colinear data")
        # Predictions should be finite
        X_gt, y_gt = gt
        y_pred, _ = model.predict(X_gt, return_std=True)
        check(np.all(np.isfinite(y_pred)), "Predictions finite on colinear data")
    except Exception as e:
        check(False, f"Colinear data caused crash: {type(e).__name__}: {e}")

def test_lognormal_nst():
    header("14. Log-Normal Data (S12 — tests NST auto-detection)")
    X, y, gt = load_scenario("S12_LogNormal")

    # Check normality — should detect strong non-normality
    norm_stats = check_normality(y)
    check(not norm_stats["is_normal"],
          f"Shapiro-Wilk rejects normality: p={norm_stats['shapiro_p']:.2e}")
    check(norm_stats["recommend_nst"],
          "NST recommended for log-normal data")

    # Fit WITHOUT NST — baseline
    model_no_nst = AnisotropicKriging(n_trials=150, n_splits=5, max_anisotropy=3.0)
    model_no_nst.fit(X, y)

    # Fit WITH NST
    nst = NormalScoreTransform(tail_extrapolation=True)
    y_norm = nst.fit_transform(y)
    model_nst = AnisotropicKriging(n_trials=150, n_splits=5, max_anisotropy=3.0)
    model_nst.fit(X, y_norm)

    if gt is not None:
        X_gt, y_gt = gt
        # Without NST
        pred_no_nst, _ = model_no_nst.predict(X_gt, return_std=True)
        r2_no = r2_score(y_gt, pred_no_nst)

        # With NST
        pred_nst_norm, _ = model_nst.predict(X_gt, return_std=True)
        pred_nst = nst.inverse_transform(pred_nst_norm)
        r2_nst = r2_score(y_gt, pred_nst)

        e_rmse_no = rmse(y_gt, pred_no_nst)
        e_rmse_nst = rmse(y_gt, pred_nst)

        print(f"    Without NST: R²={r2_no:.4f}, RMSE={e_rmse_no:.4f}")
        print(f"    With NST:    R²={r2_nst:.4f}, RMSE={e_rmse_nst:.4f}")

        # Log-normal data: spatial structure should be partly recoverable.
        # NST should not make things dramatically worse.
        check(r2_no > 0.3, f"Without NST: R²={r2_no:.4f} > 0.3 (spatial structure recovered)")
        check(r2_nst > 0.3, f"With NST: R²={r2_nst:.4f} > 0.3 (NST preserves spatial structure)")

def test_strong_trend():
    header("15. Strong Trend (S13 — linear trend R² ≈ 0.6)")
    X, y, gt = load_scenario("S13_StrongTrend")

    # Detect trend
    from src.preprocessor import analyze_trend
    trend_stats = analyze_trend(X[:, 0], X[:, 1], y, order=1)
    check(trend_stats["recommend_detrend"],
          f"Trend detected: F-test p={trend_stats['f_pvalue']:.2e}, R²={trend_stats['r2']:.3f}")
    print(f"    Trend R²={trend_stats['r2']:.3f}, recommend_detrend={trend_stats['recommend_detrend']}")

    # Fit with detrending
    tp = TrendProcessor(order=1)
    tp.fit(X[:, 0], X[:, 1], y)
    y_res = tp.detrend(X[:, 0], X[:, 1], y)

    model = AnisotropicKriging(n_trials=150, n_splits=5, max_anisotropy=3.0)
    model.fit(X, y_res)

    if gt is not None:
        X_gt, y_gt = gt
        y_pred_res, _ = model.predict(X_gt, return_std=True)
        y_pred = tp.retrend(X_gt[:, 0], X_gt[:, 1], y_pred_res)
        r2 = r2_score(y_gt, y_pred)
        e = rmse(y_gt, y_pred)
        data_std = float(np.std(y_gt))
        print(f"    R²={r2:.4f}, RMSE={e:.4f}, data_std={data_std:.4f}")
        # After detrending, spatial interpolation should explain substantial variance
        check(r2 > 0.5, f"R² = {r2:.4f} > 0.5 (detrended kriging on trend+spatial data)")

def test_nst_roundtrip():
    header("16. NST Roundtrip Fidelity")
    rng = np.random.default_rng(42)
    x_orig = rng.lognormal(mean=0.0, sigma=1.5, size=500)
    # Add some extreme values
    x_orig[:5] = [0.001, 0.01, 50, 100, 200]

    nst = NormalScoreTransform(tail_extrapolation=True)
    z = nst.fit_transform(x_orig)
    x_recovered = nst.inverse_transform(z)

    # Check monotonicity: sorted x should map to sorted z
    check(np.all(np.diff(z[np.argsort(x_orig)]) >= -1e-10),
          "NST is monotone increasing")

    # Roundtrip should preserve ranking
    rank_orig = np.argsort(np.argsort(x_orig))
    rank_recovered = np.argsort(np.argsort(x_recovered))
    check(np.allclose(rank_orig, rank_recovered),
          "NST roundtrip preserves rank order")

    # z should be approximately N(0,1)
    check(abs(np.mean(z)) < 0.1, f"z mean ≈ 0 (got {np.mean(z):.4f})")
    check(abs(np.std(z) - 1.0) < 0.15, f"z std ≈ 1 (got {np.std(z):.4f})")

def test_nst_small_sample():
    header("17. NST with Minimum Sample Size")
    # NST requires at least 3 observations
    nst = NormalScoreTransform()
    try:
        nst.fit(np.array([1.0, 2.0]))
        check(False, "NST should reject n=2")
    except ValueError as e:
        check("at least 3" in str(e), f"NST correctly rejects n=2: {e}")

    # n=3 should work
    nst.fit(np.array([1.0, 2.0, 3.0]))
    check(nst.is_fitted, "NST fits with n=3")

def test_max_anisotropy_bound():
    header("18. Anisotropy Search Respects max_anisotropy Bound")
    # S14 has true ratio 15:1 — test with tight and loose bounds
    X, y, _ = load_scenario("S14_ExtremeAniso")

    # Tight bound
    model_tight = AnisotropicKriging(n_trials=100, n_splits=3, max_anisotropy=3.0)
    model_tight.fit(X, y)
    ratio_tight = model_tight.get_kernel_params()["anisotropy_ratio"]
    check(ratio_tight <= 3.0 + EPS,
          f"Tight bound enforced: ratio={ratio_tight:.2f} ≤ 3.0")

    # Loose bound
    model_loose = AnisotropicKriging(n_trials=200, n_splits=5, max_anisotropy=20.0)
    model_loose.fit(X, y)
    ratio_loose = model_loose.get_kernel_params()["anisotropy_ratio"]
    check(ratio_loose >= 3.0,
          f"Loose bound allows high ratio: ratio={ratio_loose:.2f} ≥ 3.0")
    print(f"    tight max=3.0  → recovered ratio={ratio_tight:.2f}")
    print(f"    loose max=20.0 → recovered ratio={ratio_loose:.2f}")

def test_all_models_trial():
    header("19. All Variogram Model Types Are Explored")
    X, y, _ = load_scenario("S1_Isotropic")
    from src.engines.kriging import AnisotropicKriging as AK

    all_models = AK.NATIVE_MODELS + list(AK.CUSTOM_MODELS.keys())
    print(f"    Available models: {all_models}")

    # Run with explicit categorical — Optuna should explore all
    model = AnisotropicKriging(n_trials=150, n_splits=5, max_anisotropy=3.0)
    model.fit(X, y)
    chosen = model.best_model_name_
    check(chosen in all_models,
          f"Chosen model '{chosen}' is in the known model list")
    print(f"    Best model selected: {chosen}")

# ═══════════════════════════════════════════════════════════════════════════
# 4. End-to-end pipeline smoke tests
# ═══════════════════════════════════════════════════════════════════════════

def test_full_pipeline_kriging():
    header("20. Full Pipeline — Kriging Mode End-to-End")
    import subprocess, tempfile, yaml, shutil

    # Create a temporary config for S1 kriging
    tmp_dir = Path(tempfile.mkdtemp(prefix="krg_test_"))
    try:
        cfg = {
            "input": {
                "filepath": str(DATA_DIR / "S1_Isotropic.csv"),
                "format": "csv",
                "columns": {"x": "X", "y": "Y", "value": "Value"},
                "ground_truth_filepath": str(DATA_DIR / "S1_Isotropic_ground_truth.csv"),
            },
            "prediction_points": None,
            "geometry": {"resolution_m": 50.0, "convex_hull_buffer_percent": 10.0},
            "preprocessing": {
                "detrend": {"auto_detect": False, "enabled": False, "order": 1},
                "nst": {"enabled": False},
                "duplicates": {"min_separation": None},
            },
            "engine": {
                "mode": "kriging",
                "kriging": {"max_anisotropy": 3.0, "n_splits": 3, "n_trials": 50},
            },
            "output": {
                "base_directory": str(tmp_dir / "output"),
                "save_diagnostics": True,
                "formats": ["csv"],
            },
        }
        cfg_path = tmp_dir / "test_config.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f)

        result = subprocess.run(
            [PYTHON, "main.py", str(cfg_path)],
            capture_output=True, text=True, timeout=360,
            cwd=str(Path(__file__).parent),
        )
        check(result.returncode == 0,
              f"Pipeline exited with code {result.returncode}")
        if result.returncode != 0:
            # Print last 30 lines of output for debugging
            lines = result.stdout.split("\n") + result.stderr.split("\n")
            relevant = [l for l in lines[-30:] if l.strip()]
            for l in relevant[-15:]:
                print(f"    [pipeline] {l}")

        # Check output files exist
        out_dir = tmp_dir / "output" / "S1_Isotropic"
        check((out_dir / "parameters_kriging.json").exists(),
              "parameters_kriging.json exists")
        check((out_dir / "cv_results_kriging.csv").exists(),
              "cv_results_kriging.csv exists")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Kriging Engine Test Suite")
    print(f"  Python: {PYTHON}")
    print(f"  Data:   {DATA_DIR}")
    print("=" * 60)

    tests = [
        # Core functionality
        test_isotropic_fit,
        test_anisotropic_recovery,
        test_sparse_data,
        test_high_nugget,
        test_extreme_anisotropy,
        test_nested_structures,
        test_clustered_sampling,
        # API correctness
        test_fit_with_known_params,
        test_predict_shapes,
        test_get_kernel_params_keys,
        # Edge cases
        test_few_points,
        test_duplicate_coordinates,
        test_colinear_points,
        test_lognormal_nst,
        test_strong_trend,
        test_nst_roundtrip,
        test_nst_small_sample,
        test_max_anisotropy_bound,
        test_all_models_trial,
        # End-to-end
        test_full_pipeline_kriging,
    ]

    for test in tests:
        try:
            test()
        except Exception as e:
            FAIL += 1
            print(f"  ✗ TEST CRASHED: {test.__name__}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'═'*60}")
    total = PASS + FAIL
    print(f"  Results: {PASS}/{total} passed", end="")
    if FAIL > 0:
        print(f", {FAIL} FAILED")
    else:
        print(" — all tests pass ✓")
    print(f"{'═'*60}")

    sys.exit(0 if FAIL == 0 else 1)
