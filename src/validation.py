"""
validation.py — Shared input validation for spatial interpolation engines.

Pure functions that raise specific ValueError messages so users know exactly
what's wrong with their data, rather than getting opaque errors from
PyKrige / Optuna / ConvexHull deep in the stack.
"""
import numpy as np

EPS = 1e-12


def validate_finite(X, y=None, name="coordinates"):
    """
    Check that X (and optionally y) contain only finite values.

    Raises ValueError with counts of NaN/Inf values found.
    """
    X = np.asarray(X)
    n_bad = int((~np.isfinite(X)).sum())
    if n_bad > 0:
        raise ValueError(
            f"Input {name} contains {n_bad} non-finite value(s) "
            f"(NaN or Inf). Please remove or impute them before fitting."
        )
    if y is not None:
        y = np.asarray(y)
        n_bad_y = int((~np.isfinite(y)).sum())
        if n_bad_y > 0:
            raise ValueError(
                f"Input values contain {n_bad_y} non-finite value(s) "
                f"(NaN or Inf). Please remove or impute them before fitting."
            )


def validate_min_samples(X, y, min_samples=5):
    """
    Check that there are enough data points for meaningful spatial modelling.

    Args:
        X: coordinate array (n, d)
        y: value array (n,)
        min_samples: minimum required samples (default 5)

    Raises ValueError if n < min_samples.
    """
    X = np.asarray(X)
    n = len(X)
    if n < min_samples:
        raise ValueError(
            f"Only {n} data point(s) provided. At least {min_samples} are "
            f"required for variogram fitting and kriging. "
            f"Please provide more samples or use a simpler interpolation method."
        )


def validate_not_constant(y):
    """
    Check that the target values have non-zero variance.

    Raises ValueError if all values are identical (within machine precision).
    """
    y = np.asarray(y, dtype=np.float64)
    data_var = float(np.var(y))
    if data_var < EPS:
        raise ValueError(
            f"All target values are identical (mean = {y[0]:.6g}). "
            f"Kriging cannot model a spatially constant field — there is "
            f"no spatial structure to learn. If this is intentional, "
            f"use the constant value directly as your prediction."
        )


def validate_2d_coordinates(X):
    """
    Check that X is a 2D coordinate array with non-degenerate geometry.

    Checks:
      1. X has shape (n, 2)
      2. Points are not all coincident (max pairwise distance > 0)
      3. Points are not perfectly colinear (condition number check)

    Raises ValueError for any degenerate case.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError(
            f"Expected coordinate array of shape (n, 2), got {X.shape}. "
            f"This engine only supports 2D spatial interpolation."
        )
    n = X.shape[0]
    if n < 2:
        return  # validate_min_samples catches this separately

    # Check for all-coincident points
    extent = np.max(X, axis=0) - np.min(X, axis=0)
    if np.all(extent < EPS):
        raise ValueError(
            f"All data points are at the same location "
            f"({X[0]}). Cannot build a spatial model from coincident points."
        )

    # Check for perfectly colinear points via condition number of centred coords
    X_c = X - X.mean(axis=0)
    cov = X_c.T @ X_c / n
    eigvals = np.linalg.eigvalsh(cov)
    if eigvals[0] < EPS and eigvals[1] > EPS:
        raise ValueError(
            f"All data points are perfectly colinear (lie on a single line). "
            f"Kriging requires points with 2D spatial spread to model anisotropy. "
            f"Consider projecting to 1D or adding a small perpendicular jitter."
        )


def validate_shape_match(X, y):
    """
    Check that coordinates and values have matching first dimensions.

    Raises ValueError if len(X) != len(y).
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if len(X) != len(y):
        raise ValueError(
            f"Mismatch: {len(X)} coordinate rows but {len(y)} values. "
            f"Each point must have exactly one target value."
        )
