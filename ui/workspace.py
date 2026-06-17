"""
workspace.py — the unified Workspace tab.

Left: variogram (or GP kernel) controls + a Live-update toggle + Run/Export.
Right: a 2x2 grid of interactive plots (prediction surface, uncertainty,
variogram fit, CV dashboard) + a metrics bar.

Live mode: moving a control re-renders surface/uncertainty/variogram-fit in
real time via the in-process coarse-grid predictor (ui/live_predictor.py),
debounced and run off-thread. The CV dashboard + metrics (incl. mean_SSPE/
RMSS) refresh only on a full Run/Refresh (the subprocess engine). Files are
written only when the user clicks Export.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ui.variogram_panel import (
    LabeledSlider, SubTabCanvas, VARIOGRAM_MODELS, compute_model_curve,
    compute_empirical_variogram,
)
from ui.live_predictor import compute_preview


# ─────────────────────────────────────────────────────────────────────────────
# Variogram / kernel controls (reusable; built fresh, not carved from the old
# Model-tab panels which are being removed)
# ─────────────────────────────────────────────────────────────────────────────

class VariogramControls(ttk.Frame):
    """Control column for the active engine. Emits get_preset()/set ranges,
    and calls on_change() whenever any control moves."""

    def __init__(self, parent, state: dict, on_change, **kwargs):
        super().__init__(parent, **kwargs)
        self.state = state
        self._on_change = on_change
        self._build()

    def _build(self):
        row = 0
        ttk.Label(self, text="Engine:").grid(row=row, column=0, sticky="w")
        self._engine_var = tk.StringVar(value=self.state.get("engine_mode", "kriging"))
        eng = ttk.Combobox(self, textvariable=self._engine_var, state="readonly",
                           width=16, values=["kriging", "gp"])
        eng.grid(row=row, column=1, sticky="w", pady=(0, 6))
        eng.bind("<<ComboboxSelected>>", lambda _: self._on_engine_change())
        row += 1

        # ── Kriging model dropdown ──
        self._model_lbl = ttk.Label(self, text="Model:")
        self._model_lbl.grid(row=row, column=0, sticky="w")
        self._model_var = tk.StringVar(value="spherical")
        self._model_cb = ttk.Combobox(
            self, textvariable=self._model_var, state="readonly", width=16,
            values=list(VARIOGRAM_MODELS.keys()))
        self._model_cb.grid(row=row, column=1, sticky="w", pady=(0, 4))
        self._model_cb.bind("<<ComboboxSelected>>", lambda _: self._fire())
        row += 1

        # ── Kriging kernel for GP ──
        self._kernel_var = tk.StringVar(value="matern_52")

        # ── Sliders (shared widgets; meaning differs per engine) ──
        self._range_sl = LabeledSlider(self, "Range / Length:", 1, 1000, 300,
                                       on_change=self._fire)
        self._sill_sl  = LabeledSlider(self, "Sill (psill):", 0.001, 20, 2.0,
                                       on_change=self._fire)
        self._nugget_sl = LabeledSlider(self, "Nugget:", 0, 10, 0.5,
                                        on_change=self._fire)
        self._angle_sl = LabeledSlider(self, "Angle (°):", 0, 180, 0,
                                       on_change=self._fire)
        self._aniso_sl = LabeledSlider(self, "Anisotropy ×:", 1, 10, 1,
                                       on_change=self._fire)
        self._alpha_sl = LabeledSlider(self, "Alpha:", 0.1, 2.0, 1.0,
                                       on_change=self._fire)
        self._slider_start = row
        for sl in (self._range_sl, self._sill_sl, self._nugget_sl,
                   self._angle_sl, self._aniso_sl, self._alpha_sl):
            sl.grid(row=row, column=0, columnspan=2, sticky="ew", pady=2)
            row += 1
        self._alpha_sl.grid_remove()
        self.columnconfigure(1, weight=1)
        self._apply_engine_visibility()

    # -- engine switching --
    def _on_engine_change(self):
        self.state["engine_mode"] = self._engine_var.get()
        self._apply_engine_visibility()
        self._fire()

    def _apply_engine_visibility(self):
        is_gp = self._engine_var.get() == "gp"
        if is_gp:
            self._model_lbl.grid_remove()
            self._model_cb.grid_remove()
            self._sill_sl.grid_remove()
            self._nugget_sl.grid_remove()
            self._alpha_sl.grid_remove()
        else:
            self._model_lbl.grid()
            self._model_cb.grid()
            self._sill_sl.grid()
            self._nugget_sl.grid()
            model = self._model_var.get()
            _, has_alpha = VARIOGRAM_MODELS.get(model, (None, False))
            (self._alpha_sl.grid() if has_alpha else self._alpha_sl.grid_remove())

    def _fire(self, *_):
        # alpha visibility tracks the kriging model
        if self._engine_var.get() != "gp":
            _, has_alpha = VARIOGRAM_MODELS.get(self._model_var.get(), (None, False))
            (self._alpha_sl.grid() if has_alpha else self._alpha_sl.grid_remove())
        if self._on_change:
            self._on_change()

    @property
    def engine(self) -> str:
        return self._engine_var.get()

    def get_preset(self) -> dict:
        if self._engine_var.get() == "gp":
            return {
                "kernel_type":        self._kernel_var.get(),
                "length_scale_major": self._range_sl.get(),
                "anisotropy_ratio":   self._aniso_sl.get(),
                "angle_deg":          self._angle_sl.get(),
            }
        preset = {
            "model":            self._model_var.get(),
            "psill":            self._sill_sl.get(),
            "range":            self._range_sl.get(),
            "nugget":           self._nugget_sl.get(),
            "angle_deg":        self._angle_sl.get(),
            "anisotropy_ratio": self._aniso_sl.get(),
        }
        _, has_alpha = VARIOGRAM_MODELS.get(preset["model"], (None, False))
        if has_alpha:
            preset["alpha"] = self._alpha_sl.get()
        return preset

    def configure_ranges(self, X, y):
        """Adjust slider ranges/defaults from loaded data."""
        lags, sv = compute_empirical_variogram(X, y)
        if len(lags) == 0:
            return
        max_lag = float(lags[-1]) * 1.5
        data_var = float(np.var(y))
        self._range_sl.configure_range(max_lag * 0.01, max_lag * 2)
        self._range_sl.set(max_lag * 0.5)
        self._sill_sl.configure_range(0.001, data_var * 4)
        self._sill_sl.set(data_var)
        self._nugget_sl.configure_range(0, data_var * 2)
        self._nugget_sl.set(0.0)

    def set_engine(self, engine: str):
        self._engine_var.set(engine)
        self._apply_engine_visibility()

    def populate_from_params(self, params: dict):
        """Fill controls from an Auto-Optimize parameters dict."""
        if self._engine_var.get() == "gp":
            kt = params.get("kernel_type", "matern_52")
            self._kernel_var.set(kt)
            ls = params.get("length_scale", None)
            if isinstance(ls, (list, tuple)) and ls:
                self._range_sl.set(max(ls))
                if len(ls) >= 2:
                    self._aniso_sl.set(min(max(ls) / max(min(ls), 1e-6), 15))
            if "rotation_angle_deg" in params:
                self._angle_sl.set(params["rotation_angle_deg"])
        else:
            if "best_model" in params:
                self._model_var.set(params["best_model"])
            if "range" in params:
                self._range_sl.set(params["range"])
            if "psill" in params:
                self._sill_sl.set(params["psill"])
            if "nugget" in params:
                self._nugget_sl.set(params["nugget"])
            if "rotation_angle_deg" in params:
                self._angle_sl.set(params["rotation_angle_deg"])
            if "anisotropy_ratio" in params:
                self._aniso_sl.set(params["anisotropy_ratio"])
            if "alpha" in params:
                self._alpha_sl.set(params["alpha"])
        self._apply_engine_visibility()


# ─────────────────────────────────────────────────────────────────────────────
# Workspace panel
# ─────────────────────────────────────────────────────────────────────────────

class WorkspacePanel(ttk.Frame):
    def __init__(self, parent, state: dict, on_run_full, on_auto_fit, **kwargs):
        super().__init__(parent, **kwargs)
        self.state = state
        self._on_run_full = on_run_full
        self._on_auto_fit = on_auto_fit
        self._X = None
        self._y = None
        self._last_full = None         # last full-run result dict (for Export)
        self._debounce_id = None
        self._build()

    def _build(self):
        # ── Left controls ──
        left = ttk.Frame(self)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        self._controls = VariogramControls(left, self.state,
                                           on_change=self._on_param_change)
        self._controls.pack(fill="x")

        opts = ttk.Frame(left)
        opts.pack(fill="x", pady=(8, 0))
        self._live_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="Live update", variable=self._live_var,
                        command=self._on_param_change).pack(anchor="w")
        ttk.Button(opts, text="Auto-fit ▸", command=self._auto_fit).pack(
            fill="x", pady=2)
        ttk.Button(opts, text="▶  Run full-res", command=self._run_full).pack(
            fill="x", pady=2)
        ttk.Button(opts, text="💾  Export…", command=self._export).pack(
            fill="x", pady=2)

        self._status_var = tk.StringVar(value="Load data, then adjust the variogram.")
        ttk.Label(left, textvariable=self._status_var, foreground="gray",
                  wraplength=240, justify="left").pack(anchor="w", pady=(8, 0))

        # ── Right results: 2x2 interactive canvases + metrics bar ──
        right = ttk.Frame(self)
        right.grid(row=0, column=1, sticky="nsew")

        self._surf_canvas  = SubTabCanvas(right, figsize=(5.0, 3.4))
        self._unc_canvas   = SubTabCanvas(right, figsize=(5.0, 3.4))
        self._vario_canvas = SubTabCanvas(right, figsize=(5.0, 3.4))
        self._cv_canvas    = SubTabCanvas(right, figsize=(5.0, 3.4))
        self._surf_canvas.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        self._unc_canvas.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)
        self._vario_canvas.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)
        self._cv_canvas.grid(row=1, column=1, sticky="nsew", padx=2, pady=2)
        right.columnconfigure(0, weight=1)
        right.columnconfigure(1, weight=1)
        right.rowconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        self._metrics_var = tk.StringVar(value="Run full-res for CV metrics "
                                               "(MAE · RMSE · R² · mean_SSPE · RMSS).")
        mbar = ttk.Label(right, textvariable=self._metrics_var, relief="groove",
                         padding=4, anchor="w")
        mbar.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(4, 0))

        self.columnconfigure(0, weight=2)
        self.columnconfigure(1, weight=5)
        self.rowconfigure(0, weight=1)

    # ── data ──
    def load_data(self, X, y):
        self._X = np.asarray(X, float)
        self._y = np.asarray(y, float)
        self.state["_Xy"] = (self._X, self._y)
        self._controls.set_engine(self.state.get("engine_mode", "kriging"))
        self._controls.configure_ranges(self._X, self._y)
        self._redraw_variogram_fit()
        if self._live_var.get():
            self._schedule_preview()

    # ── live flow ──
    def _on_param_change(self):
        self.state["engine_mode"] = self._controls.engine
        self._redraw_variogram_fit()
        if self._live_var.get():
            self._schedule_preview()
        else:
            self._status_var.set("Live off — click Run full-res to update the surface.")

    def _schedule_preview(self):
        if self._X is None:
            return
        if self._debounce_id:
            self.after_cancel(self._debounce_id)
        self._debounce_id = self.after(300, self._launch_preview)

    def _launch_preview(self):
        # Compute the coarse preview INLINE on the Tk main thread. matplotlib-
        # TkAgg + Tcl/Tk are not thread-safe; driving a redraw from a worker
        # thread hard-crashes the process on Windows. The coarse grid keeps a
        # kriging solve at ~0.2–0.5 s, which is fine after the 300 ms debounce.
        self._debounce_id = None
        if self._X is None:
            return
        self._status_var.set("Computing live preview…")
        try:
            self.update_idletasks()   # paint the status before the solve
        except Exception:
            pass
        try:
            res = compute_preview(self._controls.engine, self._X, self._y,
                                  self._controls.get_preset(), n_cells=40)
        except Exception as exc:  # noqa: BLE001
            self._status_var.set(f"Preview failed: {exc}")
            return
        self._draw_surface(res["X_grid"], res["Y_grid"], res["mean"], res["std"],
                           res["X_obs"], res["Y_obs"], res.get("hull"),
                           title_suffix="(live preview)")
        self._status_var.set("Live preview updated.")

    # ── drawing ──
    def _draw_surface(self, X_grid, Y_grid, mean, std, X_obs, Y_obs, hull,
                      title_suffix=""):
        for canvas, data, label, cmap in (
            (self._surf_canvas, mean, "Predicted mean", "viridis"),
            (self._unc_canvas, std, "Uncertainty (std)", "magma_r"),
        ):
            fig = canvas.fig
            fig.clear()
            ax = fig.add_subplot(111)
            try:
                cf = ax.contourf(X_grid, Y_grid, data, levels=30, cmap=cmap,
                                 extend="both")
                fig.colorbar(cf, ax=ax, fraction=0.046, label=label)
            except Exception:
                ax.text(0.5, 0.5, "no surface", ha="center", va="center",
                        transform=ax.transAxes)
            if hull is not None and len(hull):
                ax.plot(hull[:, 0], hull[:, 1], "w-", lw=1.0, alpha=0.7)
            if X_obs is not None:
                ax.scatter(X_obs, Y_obs, c="red", s=6, marker="x", alpha=0.6)
            ax.set_aspect("equal")
            ax.set_title(f"{label} {title_suffix}", fontsize=11)
            ax.tick_params(labelsize=9)
            canvas.canvas.draw_idle()

    def _redraw_variogram_fit(self):
        fig = self._vario_canvas.fig
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_xlabel("Lag distance", fontsize=11)
        ax.set_ylabel("Semivariance", fontsize=11)
        ax.set_title("Variogram fit", fontsize=11)
        ax.tick_params(labelsize=9)
        if self._X is not None and self._controls.engine != "gp":
            lags, sv = compute_empirical_variogram(self._X, self._y)
            if len(lags):
                ax.scatter(lags, sv, color="#1f77b4", s=28, zorder=3,
                           label="Empirical")
                p = self._controls.get_preset()
                h = np.linspace(0, lags[-1] * 1.2, 200)
                gam = compute_model_curve(p["model"], h, p["range"], p["psill"],
                                          p["nugget"], p.get("alpha", 1.0))
                ax.plot(h, gam, color="#d62728", lw=2, label=p["model"])
                ax.legend(fontsize=9)
        elif self._controls.engine == "gp":
            ax.text(0.5, 0.5, "GP fits by marginal likelihood\n(no variogram curve)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=10,
                    color="gray")
        self._vario_canvas.canvas.draw_idle()

    def _draw_cv(self, cv_df):
        from utils import plot_cv_dashboard
        fig = self._cv_canvas.fig
        try:
            plot_cv_dashboard(cv_df, engine_name=self._controls.engine.upper(),
                              scenario_name="", fig=fig)
        except Exception as exc:  # noqa: BLE001
            fig.clear()
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f"CV plot failed:\n{exc}", ha="center",
                    va="center", transform=ax.transAxes, fontsize=9)
        self._cv_canvas.canvas.draw_idle()

    # ── full run + export ──
    def _auto_fit(self):
        if self._on_auto_fit:
            self._on_auto_fit()

    def _run_full(self):
        if self._X is None:
            messagebox.showinfo("No data", "Load an input file first.")
            return
        self._status_var.set("Running full-resolution interpolation…")
        if self._on_run_full:
            self._on_run_full()

    def show_full_result(self, result: dict):
        """Called by the app when the full subprocess run completes."""
        self._last_full = result
        grid = result.get("grid")
        if grid is not None and "mean" in grid:
            hull = grid.get("hull")
            self._draw_surface(grid["xv"], grid["yv"], grid["mean"], grid["std"],
                               grid.get("X_obs"), grid.get("Y_obs"),
                               hull if hull is not None and len(hull) else None,
                               title_suffix="(full-res)")
        cv_df = result.get("cv_df")
        if cv_df is not None:
            self._draw_cv(cv_df)

        def _g(k):
            v = result.get(k)
            return f"{v:.4g}" if isinstance(v, (int, float)) and v == v else "—"
        self._metrics_var.set(
            f"MAE {_g('mae')}    RMSE {_g('rmse')}    R² {_g('r2')}    "
            f"mean_SSPE {_g('mean_sspe')}    RMSS {_g('rmss')}")
        self._status_var.set("Full-resolution result ready. Export to save.")

    def populate_from_params(self, params: dict, cv: dict = None):
        """Called when Auto-Optimize finishes: load fitted params, show CV."""
        self._controls.populate_from_params(params)
        if cv:
            parts = "    ".join(f"{k} {v:.4g}" if isinstance(v, (int, float))
                                else f"{k} {v}" for k, v in cv.items())
            self._metrics_var.set(f"Auto-fit CV:   {parts}")
        self._on_param_change()   # redraw fit + (if live) preview with new params

    def _export(self):
        if self._last_full is None:
            messagebox.showinfo(
                "Nothing to export",
                "Run a full-resolution interpolation first, then export.")
            return
        dlg = _ExportDialog(self)
        self.wait_window(dlg)
        if not dlg.confirmed:
            return
        folder = dlg.folder
        if not folder:
            return
        saved = []
        try:
            if dlg.want_figures:
                for name, canvas in (("prediction_surface", self._surf_canvas),
                                     ("uncertainty", self._unc_canvas),
                                     ("variogram_fit", self._vario_canvas),
                                     ("cv_dashboard", self._cv_canvas)):
                    p = Path(folder) / f"{name}.png"
                    canvas.fig.savefig(p, dpi=150, bbox_inches="tight")
                    saved.append(p.name)
            if dlg.want_grid and self._last_full.get("grid") is not None:
                g = self._last_full["grid"]
                np.savez_compressed(Path(folder) / "predicted_grid.npz", **g)
                saved.append("predicted_grid.npz")
            if dlg.want_cv and self._last_full.get("cv_df") is not None:
                self._last_full["cv_df"].to_csv(
                    Path(folder) / "cv_results.csv", index=False)
                saved.append("cv_results.csv")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Export failed", str(exc))
            return
        messagebox.showinfo("Export complete",
                            "Saved:\n" + "\n".join(saved) if saved else "Nothing selected.")


class _ExportDialog(tk.Toplevel):
    """Small modal: choose what to export + target folder."""

    def __init__(self, parent):
        super().__init__(parent)
        self.title("Export results")
        self.transient(parent)
        self.grab_set()
        self.confirmed = False
        self.folder = None
        self.want_figures = tk.BooleanVar(value=True)
        self.want_grid = tk.BooleanVar(value=True)
        self.want_cv = tk.BooleanVar(value=True)

        ttk.Label(self, text="Export to a folder:").pack(anchor="w", padx=12, pady=(12, 4))
        ttk.Checkbutton(self, text="Figures (PNG)", variable=self.want_figures).pack(anchor="w", padx=20)
        ttk.Checkbutton(self, text="Predicted grid (.npz)", variable=self.want_grid).pack(anchor="w", padx=20)
        ttk.Checkbutton(self, text="CV results (.csv)", variable=self.want_cv).pack(anchor="w", padx=20)

        btns = ttk.Frame(self)
        btns.pack(fill="x", padx=12, pady=12)
        ttk.Button(btns, text="Choose folder & export", command=self._choose).pack(side="left")
        ttk.Button(btns, text="Cancel", command=self.destroy).pack(side="right")

    def _choose(self):
        folder = filedialog.askdirectory(title="Export to folder")
        if folder:
            self.folder = folder
            self.confirmed = True
            self.destroy()
