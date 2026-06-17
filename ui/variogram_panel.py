"""
variogram_panel.py  –  Interactive variogram fitting panel (Tab 3).

Contains:
  - Variogram model math functions (computed in the UI, no engine call needed)
  - KrigingPanel  : empirical dots + live fitted-curve canvas + slider controls
  - GPPanel       : empirical variogram reference + kernel/angle/scale controls
"""

import tkinter as tk
from tkinter import ttk
from typing import Optional, Callable
import numpy as np

# Matplotlib embedded in Tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from tkinter import filedialog
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Variogram model functions  (γ(h) formulas)
# ─────────────────────────────────────────────────────────────────────────────

def _spherical(h, range_, sill, nugget):
    h = np.asarray(h, dtype=float)
    g = np.where(
        h <= range_,
        nugget + sill * (1.5 * (h / range_) - 0.5 * (h / range_) ** 3),
        nugget + sill,
    )
    return np.where(h == 0, 0.0, g)

def _exponential(h, range_, sill, nugget):
    h = np.asarray(h, dtype=float)
    return np.where(h == 0, 0.0, nugget + sill * (1.0 - np.exp(-h / range_)))

def _gaussian(h, range_, sill, nugget):
    h = np.asarray(h, dtype=float)
    return np.where(h == 0, 0.0, nugget + sill * (1.0 - np.exp(-(h / range_) ** 2)))

def _matern32(h, range_, sill, nugget):
    h = np.asarray(h, dtype=float)
    u = np.sqrt(3) * h / range_
    return np.where(h == 0, 0.0, nugget + sill * (1.0 - (1.0 + u) * np.exp(-u)))

def _matern52(h, range_, sill, nugget):
    h = np.asarray(h, dtype=float)
    u = np.sqrt(5) * h / range_
    return np.where(h == 0, 0.0, nugget + sill * (1.0 - (1.0 + u + u**2 / 3.0) * np.exp(-u)))

def _power(h, range_, sill, nugget):
    h = np.asarray(h, dtype=float)
    return np.where(h == 0, 0.0, nugget + sill * (h / range_))

def _linear(h, range_, sill, nugget):
    h = np.asarray(h, dtype=float)
    return np.where(h == 0, 0.0, nugget + np.minimum(sill * h / range_, sill))

def _hole_effect(h, range_, sill, nugget):
    h = np.asarray(h, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        g = nugget + sill * (1.0 - np.sinc(h / range_))
    return np.where(h == 0, 0.0, g)

def _stable(h, range_, sill, nugget, alpha=1.5):
    h = np.asarray(h, dtype=float)
    return np.where(h == 0, 0.0, nugget + sill * (1.0 - np.exp(-(h / range_) ** alpha)))

def _circular(h, range_, sill, nugget):
    h = np.asarray(h, dtype=float)
    mask = h < range_
    g = np.full_like(h, nugget + sill)
    hr = np.where(mask, h / range_, 1.0)
    g[mask] = (nugget + sill * (1.0 - (2.0 / np.pi) *
               (np.arccos(hr[mask]) - hr[mask] * np.sqrt(1.0 - hr[mask] ** 2))))
    return np.where(h == 0, 0.0, g)

def _rational_quadratic(h, range_, sill, nugget, alpha=1.0):
    h = np.asarray(h, dtype=float)
    return np.where(
        h == 0, 0.0,
        nugget + sill * (1.0 - (1.0 + h**2 / (2.0 * alpha * range_**2)) ** (-alpha))
    )

VARIOGRAM_MODELS = {
    "spherical":         (_spherical,         False),
    "exponential":       (_exponential,       False),
    "gaussian":          (_gaussian,          False),
    "matern_32":         (_matern32,          False),
    "matern_52":         (_matern52,          False),
    "linear":            (_linear,            False),
    "power":             (_power,             False),
    "hole-effect":       (_hole_effect,       False),
    "stable":            (_stable,            True),   # True = has alpha param
    "circular":          (_circular,          False),
    "rational-quadratic":(_rational_quadratic,True),
}

def compute_model_curve(model_name: str, h: np.ndarray, range_: float,
                        sill: float, nugget: float, alpha: float = 1.0) -> np.ndarray:
    fn, has_alpha = VARIOGRAM_MODELS.get(model_name, (_spherical, False))
    if has_alpha:
        return fn(h, max(range_, 1e-6), max(sill, 0.0), max(nugget, 0.0), alpha)
    return fn(h, max(range_, 1e-6), max(sill, 0.0), max(nugget, 0.0))


# ─────────────────────────────────────────────────────────────────────────────
# Empirical variogram computation (omnidirectional)
# ─────────────────────────────────────────────────────────────────────────────

def compute_empirical_variogram(X: np.ndarray, y: np.ndarray,
                                 n_lags: int = 15) -> tuple:
    """Return (lag_centers, semivariance) for the omnidirectional variogram."""
    n = len(y)
    if n < 4:
        return np.array([]), np.array([])

    coords = np.column_stack([X[:, 0], X[:, 1]])
    from scipy.spatial.distance import pdist, squareform
    dist_mat = squareform(pdist(coords))
    val_diff_sq = squareform(pdist(y.reshape(-1, 1))) ** 2

    max_lag = np.percentile(dist_mat[dist_mat > 0], 50)
    lag_edges = np.linspace(0, max_lag, n_lags + 1)
    lag_centers = 0.5 * (lag_edges[:-1] + lag_edges[1:])

    sv = np.full(n_lags, np.nan)
    for i in range(n_lags):
        mask = (dist_mat > lag_edges[i]) & (dist_mat <= lag_edges[i + 1])
        pairs = val_diff_sq[mask]
        if len(pairs) > 0:
            sv[i] = 0.5 * np.mean(pairs)

    valid = ~np.isnan(sv)
    return lag_centers[valid], sv[valid]


# ─────────────────────────────────────────────────────────────────────────────
# Labeled slider widget
# ─────────────────────────────────────────────────────────────────────────────

class LabeledSlider(ttk.Frame):
    """A slider + numeric entry that stay in sync."""

    def __init__(self, parent, label: str, from_: float, to: float,
                 initial: float, resolution: float = 0.01,
                 on_change: Optional[Callable] = None, **kwargs):
        super().__init__(parent, **kwargs)
        self._on_change = on_change
        self._updating  = False

        ttk.Label(self, text=label, width=20, anchor="w").grid(
            row=0, column=0, sticky="w", padx=(0, 4))

        self._var = tk.DoubleVar(value=initial)
        self._slider = ttk.Scale(
            self, from_=from_, to=to, variable=self._var,
            orient="horizontal", length=180,
            command=self._on_slider,
        )
        self._slider.grid(row=0, column=1, sticky="ew", padx=4)

        self._entry_var = tk.StringVar(value=f"{initial:.3g}")
        self._entry = ttk.Entry(self, textvariable=self._entry_var, width=9)
        self._entry.grid(row=0, column=2, padx=(4, 0))
        self._entry.bind("<Return>",   self._on_entry)
        self._entry.bind("<FocusOut>", self._on_entry)

        self.columnconfigure(1, weight=1)

    def _on_slider(self, _=None):
        if self._updating:
            return
        self._updating = True
        val = self._var.get()
        self._entry_var.set(f"{val:.3g}")
        self._updating = False
        if self._on_change:
            self._on_change()

    def _on_entry(self, _=None):
        if self._updating:
            return
        try:
            val = float(self._entry_var.get())
            lo  = self._slider.cget("from")
            hi  = self._slider.cget("to")
            val = max(lo, min(hi, val))
            self._updating = True
            self._var.set(val)
            self._entry_var.set(f"{val:.3g}")
            self._updating = False
            if self._on_change:
                self._on_change()
        except ValueError:
            pass

    def get(self) -> float:
        return self._var.get()

    def set(self, value: float):
        self._updating = True
        self._var.set(value)
        self._entry_var.set(f"{value:.3g}")
        self._updating = False

    def configure_range(self, from_: float, to: float):
        self._slider.configure(from_=from_, to=to)


# ─────────────────────────────────────────────────────────────────────────────
# Sub-tab canvas: a Figure + canvas + Export button, for one dashboard tab
# ─────────────────────────────────────────────────────────────────────────────

class SubTabCanvas(ttk.Frame):
    """One dashboard sub-tab: a matplotlib Figure, its canvas, and an Export button."""

    def __init__(self, parent, figsize=(6.0, 4.2), dpi=96, **kwargs):
        super().__init__(parent, **kwargs)
        self.fig = Figure(figsize=figsize, dpi=dpi, tight_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", pady=(2, 0))
        ttk.Button(btn_frame, text="Export…", command=self._export).pack(
            side="right", padx=4)

    def _export(self):
        path = filedialog.asksaveasfilename(
            title="Export plot",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("SVG vector", "*.svg"),
                       ("PDF document", "*.pdf")],
        )
        if path:
            self.fig.savefig(path, dpi=150, bbox_inches="tight")


# ─────────────────────────────────────────────────────────────────────────────
# Kriging variogram panel
# ─────────────────────────────────────────────────────────────────────────────

class KrigingPanel(ttk.Frame):
    """
    Left: matplotlib canvas with empirical variogram dots + live model curve.
    Right: model type dropdown + n_lags spinbox + Range / Sill / Nugget / Angle / Anisotropy sliders.
    """

    def __init__(self, parent, state: dict, **kwargs):
        super().__init__(parent, **kwargs)
        self.state = state
        self._lags: Optional[np.ndarray] = None
        self._sv:   Optional[np.ndarray] = None
        self._X_data: Optional[np.ndarray] = None
        self._y_data: Optional[np.ndarray] = None
        self._build_layout()

    def _build_layout(self):
        # ── Left: canvas ──────────────────────────────────────────────────────
        left = ttk.Frame(self)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        self.fig    = Figure(figsize=(5, 3.5), dpi=96, tight_layout=True)
        self.ax     = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self._init_plot()

        # ── Right: controls ───────────────────────────────────────────────────
        right = ttk.Frame(self)
        right.grid(row=0, column=1, sticky="nsew")

        row = 0
        ttk.Label(right, text="Model type:").grid(
            row=row, column=0, sticky="w", pady=(0, 4))
        self._model_var = tk.StringVar(value="spherical")
        model_cb = ttk.Combobox(
            right, textvariable=self._model_var,
            values=list(VARIOGRAM_MODELS.keys()), state="readonly", width=18,
        )
        model_cb.grid(row=row, column=1, sticky="w", pady=(0, 4))
        model_cb.bind("<<ComboboxSelected>>", lambda _: self._on_param_change())
        row += 1

        # Number of lags (ArcMap default = 12)
        ttk.Label(right, text="Number of lags:").grid(
            row=row, column=0, sticky="w", pady=(0, 4))
        self._nlags_var = tk.IntVar(value=12)
        nlags_sb = ttk.Spinbox(
            right, from_=4, to=50, textvariable=self._nlags_var,
            width=6, command=self._on_nlags_change,
        )
        nlags_sb.grid(row=row, column=1, sticky="w", pady=(0, 4))
        nlags_sb.bind("<Return>", lambda _: self._on_nlags_change())
        row += 1

        # Sliders — ranges filled in after data is loaded
        self._range_sl  = LabeledSlider(right, "Range:",        1, 1000, 300,
                                         on_change=self._on_param_change)
        self._sill_sl   = LabeledSlider(right, "Sill (psill):", 0.001, 20, 2.0,
                                         on_change=self._on_param_change)
        self._nugget_sl = LabeledSlider(right, "Nugget:",       0, 10,   0.5,
                                         on_change=self._on_param_change)
        self._angle_sl  = LabeledSlider(right, "Angle (°):",    0, 180,  0,
                                         on_change=self._on_param_change)
        self._aniso_sl  = LabeledSlider(right, "Anisotropy ×:", 1, 10,   1,
                                         on_change=self._on_param_change)

        # Alpha (only for stable / rational-quadratic)
        self._alpha_sl  = LabeledSlider(right, "Alpha:",        0.1, 2.0, 1.0,
                                         on_change=self._on_param_change)

        for i, sl in enumerate([self._range_sl, self._sill_sl, self._nugget_sl,
                                  self._angle_sl, self._aniso_sl, self._alpha_sl],
                                 start=row):
            sl.grid(row=i, column=0, columnspan=2, sticky="ew", pady=2)
        row += 6

        # Alpha row hidden by default (only for models that need it)
        self._alpha_row = row - 1
        self._alpha_sl.grid_remove()

        # Buttons
        btn_frame = ttk.Frame(right)
        btn_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        ttk.Button(btn_frame, text="Auto Optimize ▸",
                   command=self._on_auto_optimize).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Reset",
                   command=self._reset_defaults).pack(side="left", padx=4)

        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=2)
        self.rowconfigure(0, weight=1)

    def _init_plot(self):
        self.ax.clear()
        self.ax.set_xlabel("Lag distance", fontsize=9)
        self.ax.set_ylabel("Semivariance", fontsize=9)
        self.ax.set_title("Variogram", fontsize=10)
        self.ax.tick_params(labelsize=8)
        self.canvas.draw_idle()

    def load_data(self, X: np.ndarray, y: np.ndarray):
        """Compute empirical variogram from data and update plot."""
        self._X_data = np.asarray(X, dtype=float)
        self._y_data = np.asarray(y, dtype=float)
        self._recompute_variogram()

    def _recompute_variogram(self):
        """(Re)compute empirical variogram with current n_lags and refresh sliders."""
        if self._X_data is None:
            return
        n_lags = self._nlags_var.get()
        self._lags, self._sv = compute_empirical_variogram(
            self._X_data, self._y_data, n_lags=n_lags
        )
        if len(self._lags) == 0:
            return

        max_lag  = float(self._lags[-1]) * 1.5
        data_var = float(np.var(self._y_data))
        self._range_sl.configure_range(max_lag * 0.01, max_lag)
        self._sill_sl.configure_range(0.001, data_var * 4)
        self._nugget_sl.configure_range(0, data_var * 2)
        self._aniso_sl.configure_range(1, float(self.state.get("kriging_max_anisotropy", 10)))

        self._range_sl.set(max_lag * 0.5)
        self._sill_sl.set(data_var)
        self._nugget_sl.set(0.0)
        self._on_param_change()

    def _on_nlags_change(self):
        """Recompute empirical variogram when user changes the lag count."""
        self._recompute_variogram()

    def _on_param_change(self):
        """Redraw only the model curve overlay — empirical dots stay fixed."""
        model = self._model_var.get()
        _, has_alpha = VARIOGRAM_MODELS.get(model, (_spherical, False))
        if has_alpha:
            self._alpha_sl.grid()
        else:
            self._alpha_sl.grid_remove()

        self._redraw_curve()
        self._push_to_state()

    def _redraw_curve(self):
        self.ax.clear()
        self.ax.set_xlabel("Lag distance", fontsize=9)
        self.ax.set_ylabel("Semivariance", fontsize=9)
        self.ax.set_title("Variogram (live)", fontsize=10)
        self.ax.tick_params(labelsize=8)

        if self._lags is not None and len(self._lags) > 0:
            self.ax.scatter(self._lags, self._sv, color="#1f77b4",
                            zorder=3, s=30, label="Empirical")
            # Fitted model curve
            h = np.linspace(0, self._lags[-1] * 1.2, 200)
            gam = compute_model_curve(
                self._model_var.get(), h,
                self._range_sl.get(), self._sill_sl.get(), self._nugget_sl.get(),
                self._alpha_sl.get(),
            )
            self.ax.plot(h, gam, color="#d62728", linewidth=2,
                         label=self._model_var.get())
            self.ax.axhline(self._sill_sl.get() + self._nugget_sl.get(),
                            color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
            self.ax.legend(fontsize=8)

        self.canvas.draw_idle()

    def _push_to_state(self):
        model = self._model_var.get()
        preset = {
            "model":            model,
            "psill":            self._sill_sl.get(),
            "range":            self._range_sl.get(),
            "nugget":           self._nugget_sl.get(),
            "angle_deg":        self._angle_sl.get(),
            "anisotropy_ratio": self._aniso_sl.get(),
        }
        _, has_alpha = VARIOGRAM_MODELS.get(model, (None, False))
        if has_alpha:
            preset["alpha"] = self._alpha_sl.get()
        self.state["kriging_preset"] = preset
        self.state["kriging_n_lags"] = self._nlags_var.get()

    def populate_from_params(self, params: dict):
        """Fill sliders from a parameters dict (returned by Auto Optimize)."""
        model = params.get("best_model", "spherical")
        self._model_var.set(model)
        if "range"  in params: self._range_sl.set(params["range"])
        if "psill"  in params: self._sill_sl.set(params["psill"])
        if "nugget" in params: self._nugget_sl.set(params["nugget"])
        if "rotation_angle_deg" in params:
            self._angle_sl.set(params["rotation_angle_deg"])
        if "anisotropy_ratio" in params:
            self._aniso_sl.set(params["anisotropy_ratio"])
        if "alpha" in params:
            self._alpha_sl.set(params["alpha"])
        self._on_param_change()

    def _reset_defaults(self):
        if self._lags is not None and len(self._lags) > 0:
            data_var = float(self._sv.mean()) * 2 if len(self._sv) else 1.0
            self._range_sl.set(self._lags[-1] * 0.5)
            self._sill_sl.set(data_var)
            self._nugget_sl.set(0.0)
        self._angle_sl.set(0.0)
        self._aniso_sl.set(1.0)
        self._alpha_sl.set(1.0)
        self._on_param_change()

    def _on_auto_optimize(self):
        # Signal the app to launch AutoOptimizeRunner — handled by app.py callback
        self.event_generate("<<AutoOptimize>>")


# ─────────────────────────────────────────────────────────────────────────────
# GP kernel panel
# ─────────────────────────────────────────────────────────────────────────────

class GPPanel(ttk.Frame):
    """
    Left: empirical variogram for reference (no fitted curve — GP uses LML).
    Right: kernel type radio + length scale / anisotropy / angle sliders.
    """

    def __init__(self, parent, state: dict, **kwargs):
        super().__init__(parent, **kwargs)
        self.state = state
        self._lags: Optional[np.ndarray] = None
        self._sv:   Optional[np.ndarray] = None
        self._build_layout()

    def _build_layout(self):
        # ── Left: reference variogram ──────────────────────────────────────────
        left = ttk.Frame(self)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        self.fig    = Figure(figsize=(5, 3.5), dpi=96, tight_layout=True)
        self.ax     = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self._init_plot()

        # ── Right: controls ───────────────────────────────────────────────────
        right = ttk.Frame(self)
        right.grid(row=0, column=1, sticky="nsew")

        row = 0
        ttk.Label(right, text="Kernel type:").grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(0, 2))
        row += 1

        self._kernel_var = tk.StringVar(value="matern_52")
        for kname, klabel in [
            ("matern_32", "Matérn-3/2  (rough, C¹)"),
            ("matern_52", "Matérn-5/2  (moderate, C²)"),
            ("rbf",       "RBF  (smooth, C∞)"),
        ]:
            ttk.Radiobutton(
                right, text=klabel, variable=self._kernel_var, value=kname,
                command=self._push_to_state,
            ).grid(row=row, column=0, columnspan=2, sticky="w")
            row += 1

        ttk.Separator(right, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=6)
        row += 1

        self._ls_sl    = LabeledSlider(right, "Length scale:", 1, 5000, 500,
                                        on_change=self._push_to_state)
        self._aniso_sl = LabeledSlider(right, "Anisotropy ×:", 1, 15, 1,
                                        on_change=self._push_to_state)
        self._angle_sl = LabeledSlider(right, "Angle (°):",    0, 180, 0,
                                        on_change=self._push_to_state)

        for sl in [self._ls_sl, self._aniso_sl, self._angle_sl]:
            sl.grid(row=row, column=0, columnspan=2, sticky="ew", pady=2)
            row += 1

        ttk.Label(right, text="(nugget & signal variance are\nauto-scaled from data)",
                  font=("TkDefaultFont", 8), foreground="gray").grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(4, 0))
        row += 1

        btn_frame = ttk.Frame(right)
        btn_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        ttk.Button(btn_frame, text="Auto Optimize ▸",
                   command=self._on_auto_optimize).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Reset",
                   command=self._reset_defaults).pack(side="left", padx=4)

        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=2)
        self.rowconfigure(0, weight=1)

    def _init_plot(self):
        self.ax.clear()
        self.ax.set_xlabel("Lag distance", fontsize=9)
        self.ax.set_ylabel("Semivariance", fontsize=9)
        self.ax.set_title("Empirical variogram (reference)", fontsize=10)
        self.ax.tick_params(labelsize=8)
        self.canvas.draw_idle()

    def load_data(self, X: np.ndarray, y: np.ndarray):
        self._lags, self._sv = compute_empirical_variogram(X, y)
        if len(self._lags) == 0:
            return

        max_lag  = float(self._lags[-1]) * 1.5
        self._ls_sl.configure_range(max_lag * 0.01, max_lag * 2)
        self._ls_sl.set(max_lag * 0.5)
        self._aniso_sl.configure_range(1, float(self.state.get("gp_max_anisotropy", 15)))
        self._redraw_empirical()
        self._push_to_state()

    def _redraw_empirical(self):
        self.ax.clear()
        self.ax.set_xlabel("Lag distance", fontsize=9)
        self.ax.set_ylabel("Semivariance", fontsize=9)
        self.ax.set_title("Empirical variogram (reference)", fontsize=10)
        self.ax.tick_params(labelsize=8)
        if self._lags is not None and len(self._lags) > 0:
            self.ax.scatter(self._lags, self._sv, color="#1f77b4", s=30, zorder=3)
            self.ax.axhline(float(np.max(self._sv)), color="gray",
                            linestyle="--", linewidth=0.8, alpha=0.5, label="max SV")
        self.canvas.draw_idle()

    def _push_to_state(self, *_):
        ls = self._ls_sl.get()
        ratio = self._aniso_sl.get()
        self.state["gp_preset"] = {
            "kernel_type":        self._kernel_var.get(),
            "length_scale_major": ls,
            "anisotropy_ratio":   ratio,
            "angle_deg":          self._angle_sl.get(),
        }

    def populate_from_params(self, params: dict):
        """Fill controls from Auto Optimize output (parameters_gp.json)."""
        kt = params.get("kernel_type", "matern_52")
        self._kernel_var.set(kt)
        ls_list = params.get("length_scale", [500, 500])
        if isinstance(ls_list, list) and len(ls_list) >= 1:
            self._ls_sl.set(max(ls_list))
            if len(ls_list) >= 2:
                ratio = max(ls_list) / max(min(ls_list), 1e-6)
                self._aniso_sl.set(min(ratio, 15))
        if "rotation_angle_deg" in params:
            self._angle_sl.set(params["rotation_angle_deg"])
        self._push_to_state()

    def _reset_defaults(self):
        self._kernel_var.set("matern_52")
        if self._lags is not None and len(self._lags) > 0:
            self._ls_sl.set(self._lags[-1] * 0.5)
        self._aniso_sl.set(1.0)
        self._angle_sl.set(0.0)
        self._push_to_state()

    def _on_auto_optimize(self):
        self.event_generate("<<AutoOptimize>>")
