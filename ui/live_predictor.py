"""
live_predictor.py — fast in-process coarse-grid prediction for the live
Workspace preview. NOT the full pipeline (that stays in main.py/EngineRunner).
Uses preset variogram/kernel params directly; no optimization, no NST.

Kept headless-testable: no tkinter import at module level.
"""
from __future__ import annotations
import os
import sys
import threading
from typing import Optional, Callable
import numpy as np

# Ensure the project root (parent of ui/) is importable for src.* / utils.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _coarse_config(x: np.ndarray, y: np.ndarray, n_cells: int) -> dict:
    """Config for generate_prediction_grid with roughly n_cells along the long axis."""
    x_rng = float(np.ptp(x)) or 1.0
    y_rng = float(np.ptp(y)) or 1.0
    resolution = max(x_rng, y_rng) / max(int(n_cells), 4)
    return {"geometry": {"resolution_m": resolution,
                         "convex_hull_buffer_percent": 5.0}}


def _kriging_params_from_preset(preset: dict) -> dict:
    """Map the UI preset (angle_deg / anisotropy_ratio) to the keys
    AnisotropicKriging._get_ok_instance expects (angle / scaling)."""
    params = {
        "psill":   float(preset.get("psill", 1.0)),
        "range":   float(preset.get("range", 1.0)),
        "nugget":  float(preset.get("nugget", 0.0)),
        "angle":   float(preset.get("angle_deg", preset.get("angle", 0.0))),
        "scaling": float(preset.get("anisotropy_ratio",
                                    preset.get("scaling", 1.0))),
    }
    if "alpha" in preset:
        params["alpha"] = float(preset["alpha"])
    return params


def compute_preview(engine: str, X: np.ndarray, y: np.ndarray,
                    preset: dict, n_cells: int = 40) -> dict:
    """
    Predict on a coarse grid using preset params. Returns a dict with
    X_grid, Y_grid (2D), mean, std (2D, NaN outside hull), X_obs, Y_obs, hull.
    """
    from src.geometry import generate_prediction_grid

    X = np.asarray(X, float)
    y = np.asarray(y, float)
    coords = np.column_stack([X[:, 0], X[:, 1]]) if X.ndim == 2 else np.asarray(X, float)

    cfg = _coarse_config(coords[:, 0], coords[:, 1], n_cells)
    X_grid, Y_grid, mask, grid_shape, hull = generate_prediction_grid(
        coords[:, 0], coords[:, 1], cfg)
    pts = np.column_stack([X_grid.ravel(), Y_grid.ravel()])

    if engine == "gp":
        from src.engines.gp import RotatedGPR
        model = RotatedGPR()
        model.fit_with_known_params(coords, y, preset)
        mean, std = model.predict(pts, return_std=True)
    else:
        # Direct lu_solve kriging — ~3-4x faster than going through PyKrige
        # via AnisotropicKriging.predict, and the slider-drag UX hinges on it.
        from src.engines.fast_kriging import ok_predict
        kparams = _kriging_params_from_preset(preset)
        mean, std = ok_predict(
            coords, y, pts,
            preset.get("model", "spherical"),
            kparams, return_std=True)
    mean = np.asarray(mean, float).copy()
    std = np.asarray(std, float).copy()
    mean[~mask] = np.nan        # blank outside the buffered hull
    std[~mask] = np.nan
    return {
        "X_grid": X_grid, "Y_grid": Y_grid,
        "mean": mean.reshape(grid_shape), "std": std.reshape(grid_shape),
        "X_obs": coords[:, 0], "Y_obs": coords[:, 1],
        "hull": hull,
    }


class LivePreviewWorker:
    """Debounced, cancel-stale background runner for compute_preview.

    Call request(...) from the Tk thread; when a (non-stale) result is ready,
    on_done(result) is invoked through the supplied marshal callback (which
    must hop back onto the Tk thread, e.g. ``lambda fn: widget.after(0, fn)``).
    """

    def __init__(self, marshal: Callable[[Callable], None]):
        self._marshal = marshal
        self._lock = threading.Lock()
        self._seq = 0
        self._thread: Optional[threading.Thread] = None

    def request(self, engine, X, y, preset, n_cells,
                on_done, on_error=None):
        with self._lock:
            self._seq += 1
            my_seq = self._seq

        def _work():
            try:
                res = compute_preview(engine, X, y, preset, n_cells)
                with self._lock:
                    stale = my_seq != self._seq
                if not stale:
                    self._marshal(lambda: on_done(res))
            except Exception as exc:  # noqa: BLE001 — surfaced to the UI
                with self._lock:
                    stale = my_seq != self._seq
                if not stale and on_error is not None:
                    self._marshal(lambda e=exc: on_error(e))

        self._thread = threading.Thread(target=_work, daemon=True)
        self._thread.start()
