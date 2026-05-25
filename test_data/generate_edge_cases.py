"""
Generate edge-case test datasets where samples and ground truth are drawn from
the SAME underlying Gaussian random field realization, so ground truth comparison
is meaningful.

Key design: we draw one joint Gaussian vector covering all locations (sample
points + dense grid), then partition it. This guarantees the sample values and
ground truth values come from the same realization.
"""
import numpy as np
import pandas as pd
from pathlib import Path

OUT_DIR = Path(".")


def joint_gaussian_field(X_all, nugget=0.1, sill=1.0, range_val=300,
                         angle=0, ratio=1.0, seed=42):
    """
    Generate a Gaussian random field at all locations in X_all (N,2).
    Returns the noise-free field (nugget excluded from covariance).
    """
    rng = np.random.default_rng(seed)
    n = len(X_all)
    x, y = X_all[:, 0], X_all[:, 1]

    cos_a = np.cos(np.radians(angle))
    sin_a = np.sin(np.radians(angle))
    dx = np.subtract.outer(x, x)
    dy = np.subtract.outer(y, y)
    dx_rot = dx * cos_a + dy * sin_a
    dy_rot = -dx * sin_a + dy * cos_a
    h = np.sqrt(dx_rot ** 2 + (dy_rot * ratio) ** 2)

    # Spherical covariance — structured component only (no nugget on off-diagonals)
    C = np.where(h < range_val,
                 sill * (1.0 - 1.5 * h / range_val + 0.5 * (h / range_val) ** 3),
                 0.0)
    np.fill_diagonal(C, sill)  # diag = sill (noise-free variance at a point)

    L = np.linalg.cholesky(C + np.eye(n) * 1e-8)
    z = L @ rng.normal(0, 1, n)
    return z


def make_paired_dataset(sample_xy, grid_xy, field_kwargs, noise_std=0.0,
                        sample_seed=456, trend_fn=None):
    """
    Create paired (samples, ground_truth) from one joint realization.

    Parameters
    ----------
    sample_xy : (n,2) array — sample point locations
    grid_xy : (m,2) array — dense grid locations (ground truth)
    field_kwargs : dict — passed to joint_gaussian_field
    noise_std : float — Gaussian noise added to sample values ONLY
    sample_seed : int — seed for the noise
    trend_fn : callable or None — if given, add trend(X,Y) to all values

    Returns
    -------
    df_samples, df_truth
    """
    # One joint draw
    all_xy = np.vstack([sample_xy, grid_xy])
    z_all = joint_gaussian_field(all_xy, **field_kwargs)
    z_samples = z_all[:len(sample_xy)]
    z_grid = z_all[len(sample_xy):]

    # Apply trend if requested
    if trend_fn is not None:
        t_samples = trend_fn(sample_xy[:, 0], sample_xy[:, 1])
        t_grid = trend_fn(grid_xy[:, 0], grid_xy[:, 1])
        z_samples += t_samples
        z_grid += t_grid

    # Add noise to samples only
    if noise_std > 0:
        rng = np.random.default_rng(sample_seed)
        z_samples += rng.normal(0, noise_std, len(z_samples))

    df_s = pd.DataFrame({"X": sample_xy[:, 0], "Y": sample_xy[:, 1], "Value": z_samples})
    df_t = pd.DataFrame({"X": grid_xy[:, 0], "Y": grid_xy[:, 1], "Value": z_grid})
    return df_s, df_t


def dense_grid(x_min, x_max, y_min, y_max, nx, ny):
    gx = np.linspace(x_min, x_max, nx)
    gy = np.linspace(y_min, y_max, ny)
    GX, GY = np.meshgrid(gx, gy)
    return np.column_stack([GX.ravel(), GY.ravel()])


# ═══════════════════════════════════════════════════════════════════════════
# S9: Very few points (n=15) — isotropic, noise-free
# ═══════════════════════════════════════════════════════════════════════════
print("S9: Very few points (n=15)")
rng = np.random.default_rng(42)
sample_xy = rng.uniform(0, 1000, (15, 2))
grid_xy = dense_grid(0, 1000, 0, 1000, 100, 100)
df9, df9_gt = make_paired_dataset(
    sample_xy, grid_xy,
    field_kwargs=dict(nugget=0.1, sill=1.0, range_val=300, seed=42),
)
df9.to_csv(OUT_DIR / "S9_FewPoints.csv", index=False)
df9_gt.to_csv(OUT_DIR / "S9_FewPoints_ground_truth.csv", index=False)
print(f"  Samples: {len(df9)}, Ground truth: {len(df9_gt)}")

# ═══════════════════════════════════════════════════════════════════════════
# S10: Data with exact and near-duplicate coordinates
# ═══════════════════════════════════════════════════════════════════════════
print("S10: Duplicate coordinates")
X10 = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 300, 301, 302,
                100, 100, 200, 200, 300, 300, 400, 400], dtype=float)
Y10 = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 300, 301, 302,
                100, 100, 200, 200, 300, 300, 400, 400], dtype=float)
# Generate from a clean field, then overwrite with conflicting values at duplicates
sample_xy10 = np.column_stack([X10, Y10])
grid_xy10 = dense_grid(0, 1000, 0, 1000, 80, 80)
df10, df10_gt = make_paired_dataset(
    sample_xy10, grid_xy10,
    field_kwargs=dict(nugget=0.1, sill=1.0, range_val=300, seed=42),
    noise_std=0.05,
)
# Overwrite duplicate-group values with deliberately conflicting values
# This simulates measurement error at co-located points
z_clean = df10["Value"].values.copy()
z_clean[12] = z_clean[13] + 3.0   # (100,100) second copy: +3 offset
z_clean[14] = z_clean[15] + 3.0   # (200,200) second copy
z_clean[16] = z_clean[17] + 3.0   # (300,300) second copy
z_clean[18] = z_clean[19] + 3.0   # (400,400) second copy
df10["Value"] = z_clean
df10.to_csv(OUT_DIR / "S10_Duplicates.csv", index=False)
df10_gt.to_csv(OUT_DIR / "S10_Duplicates_ground_truth.csv", index=False)
print(f"  Samples: {len(df10)}, Ground truth: {len(df10_gt)}")

# ═══════════════════════════════════════════════════════════════════════════
# S11: Nearly colinear points (along a diagonal) — degenerate geometry
# ═══════════════════════════════════════════════════════════════════════════
print("S11: Nearly colinear points")
X11 = np.linspace(0, 1000, 30)
Y11 = 0.7 * X11 + np.random.default_rng(42).normal(0, 5, 30)  # tight scatter around a line
sample_xy11 = np.column_stack([X11, Y11])
grid_xy11 = dense_grid(0, 1000, 0, 1000, 80, 80)
df11, df11_gt = make_paired_dataset(
    sample_xy11, grid_xy11,
    field_kwargs=dict(nugget=0.05, sill=1.0, range_val=300, seed=42),
)
df11.to_csv(OUT_DIR / "S11_Colinear.csv", index=False)
df11_gt.to_csv(OUT_DIR / "S11_Colinear_ground_truth.csv", index=False)
print(f"  Samples: {len(df11)}, Ground truth: {len(df11_gt)}")

# ═══════════════════════════════════════════════════════════════════════════
# S12: Log-normal values (tests NST auto-detection)
# ═══════════════════════════════════════════════════════════════════════════
print("S12: Log-normal values")
rng12 = np.random.default_rng(42)
sample_xy12 = rng12.uniform(0, 1000, (200, 2))
grid_xy12 = dense_grid(0, 1000, 0, 1000, 80, 80)
df12, df12_gt = make_paired_dataset(
    sample_xy12, grid_xy12,
    field_kwargs=dict(nugget=0.05, sill=0.5, range_val=300, seed=42),
    noise_std=0.02,
)
# Apply exponential transform AFTER generation (on the same realization)
df12["Value"] = np.exp(df12["Value"] * 1.2)
df12_gt["Value"] = np.exp(df12_gt["Value"] * 1.2)
df12.to_csv(OUT_DIR / "S12_LogNormal.csv", index=False)
df12_gt.to_csv(OUT_DIR / "S12_LogNormal_ground_truth.csv", index=False)
print(f"  Samples: {len(df12)}, Ground truth: {len(df12_gt)}")

# ═══════════════════════════════════════════════════════════════════════════
# S13: Strong linear trend + spatial correlation
# ═══════════════════════════════════════════════════════════════════════════
print("S13: Strong trend + spatial correlation")
rng13 = np.random.default_rng(42)
sample_xy13 = rng13.uniform(0, 1000, (200, 2))
grid_xy13 = dense_grid(0, 1000, 0, 1000, 80, 80)
trend_fn = lambda x, y: 3.0 * (x / 1000.0) + 2.0 * (y / 1000.0) - 2.5
df13, df13_gt = make_paired_dataset(
    sample_xy13, grid_xy13,
    field_kwargs=dict(nugget=0.1, sill=1.0, range_val=300, seed=42),
    noise_std=0.1,
    trend_fn=trend_fn,
)
df13.to_csv(OUT_DIR / "S13_StrongTrend.csv", index=False)
df13_gt.to_csv(OUT_DIR / "S13_StrongTrend_ground_truth.csv", index=False)
print(f"  Samples: {len(df13)}, Ground truth: {len(df13_gt)}")

# ═══════════════════════════════════════════════════════════════════════════
# S14: Extreme anisotropy (15:1) — push beyond default max_anisotropy=3
# ═══════════════════════════════════════════════════════════════════════════
print("S14: Extreme anisotropy (15:1)")
rng14 = np.random.default_rng(42)
sample_xy14 = rng14.uniform(0, 1000, (200, 2))
grid_xy14 = dense_grid(0, 1000, 0, 1000, 80, 80)
df14, df14_gt = make_paired_dataset(
    sample_xy14, grid_xy14,
    field_kwargs=dict(nugget=0.05, sill=1.0, range_val=300, angle=60, ratio=15.0, seed=42),
    noise_std=0.05,
)
df14.to_csv(OUT_DIR / "S14_ExtremeAniso.csv", index=False)
df14_gt.to_csv(OUT_DIR / "S14_ExtremeAniso_ground_truth.csv", index=False)
print(f"  Samples: {len(df14)}, Ground truth: {len(df14_gt)}")

print("\nAll edge-case datasets generated (paired samples + ground truth).")
