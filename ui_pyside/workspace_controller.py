"""WorkspaceController — pure signal/slot mediator, no widgets."""
import os, sys, json, tempfile, csv
from pathlib import Path
import numpy as np

from PySide6.QtCore import QObject, QProcess, QTimer, Signal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ui.engine_runner import build_config


class WorkspaceController(QObject):
    logLine = Signal(str)
    statusMessage = Signal(str)
    metricsUpdated = Signal(dict)       # {mae, rmse, r2, mean_sspe, rmss}
    resultReady = Signal(dict)          # full result dict with grid/cv_df/params
    paramsReady = Signal(dict)           # optimized parameters from auto-fit
    dataLoaded = Signal()               # X/y are now available

    def __init__(self, parent=None):
        super().__init__(parent)
        self._X = None
        self._y = None
        self._engine = "kriging"
        self._live = True
        self._last_full = None
        self._auto_fit_dir = None        # temp dir for auto-fit (read params after)
        self._on_slider_preset = {
            "model": "spherical", "psill": 5.0, "range": 300,
            "nugget": 0.5, "angle_deg": 0.0, "anisotropy_ratio": 1.0}

        self._state = {"engine_mode": "kriging", "output_dir": "output",
                       "save_diagnostics": True, "export_formats": ["nc"]}

        # Debounce timer for live preview (single-shot, 300ms)
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(300)
        self._debounce.timeout.connect(self._compute_preview)

        # Full-run subprocess
        self._proc = QProcess(self)
        self._proc.setProcessChannelMode(QProcess.SeparateChannels)
        self._proc.readyReadStandardOutput.connect(self._on_stdout)
        self._proc.finished.connect(self._on_run_complete)

        # DLL-safe env for subprocess
        self._proc_env = os.environ.copy()
        self._proc_env["PYTHONIOENCODING"] = "utf-8"
        _p = sys.prefix
        _path_extra = ";".join(
            d for d in (os.path.join(_p, "Library", "bin"),
                        os.path.join(_p, "Library", "lib"),
                        os.path.join(_p, "DLLs"), _p)
            if os.path.isdir(d))
        if _path_extra:
            self._proc_env["PATH"] = _path_extra + ";" + self._proc_env.get("PATH", "")

    # ------------------------------------------------------------------
    def load_data(self, filepath, col_x="X", col_y="Y", col_val="Value"):
        import pandas as pd
        df = pd.read_csv(filepath)
        self._X = df[[col_x, col_y]].to_numpy(dtype=float)
        self._y = df[col_val].to_numpy(dtype=float)
        self._state["input_filepath"] = filepath
        self._state["col_x"] = col_x
        self._state["col_y"] = col_y
        self._state["col_value"] = col_val
        self.dataLoaded.emit()
        self.statusMessage.emit(f"Loaded {len(self._y)} points.")
        self._compute_preview()

    def set_engine(self, mode: str):
        self._engine = mode
        self._state["engine_mode"] = mode

    def set_live(self, enabled: bool):
        self._live = enabled

    def on_slider_change(self, preset: dict):
        self._on_slider_preset = preset
        self._debounce.start()

    # ------------------------------------------------------------------
    # Live preview (inline, main thread)
    def _compute_preview(self):
        if self._X is None or not self._live:
            return
        from ui.live_predictor import compute_preview
        self.statusMessage.emit("Computing live preview…")
        try:
            # Suppress engine diagnostic prints (noise in the UI)
            import io, contextlib
            with contextlib.redirect_stdout(io.StringIO()):
                res = compute_preview(self._engine, self._X, self._y,
                                      self._on_slider_preset, n_cells=40)
            self.resultReady.emit({"preview": True, "grid": res})
            self.statusMessage.emit("Live preview updated.")
        except Exception as exc:
            self.statusMessage.emit(f"Preview failed: {exc}")

    # ------------------------------------------------------------------
    # Full run (subprocess via QProcess)
    def run_full(self):
        if self._X is None:
            return
        self._proc.kill()
        self._proc.waitForFinished(500)
        self._bundle_dir = tempfile.mkdtemp(prefix="interp_uirun_")
        state = dict(self._state)
        state["ui_mode"] = True
        state["bundle_dir"] = self._bundle_dir
        state["engine_mode"] = self._engine
        preset = self._on_slider_preset
        if self._engine == "kriging" and preset:
            state["kriging"] = state.get("kriging", {})
            state["kriging"]["preset_params"] = preset
        elif self._engine == "gp" and preset:
            state["gp"] = state.get("gp", {})
            state["gp"]["preset_params"] = preset
        cfg_path = _write_temp_config(state)
        self.statusMessage.emit("Running full-resolution interpolation…")
        self._proc.start(sys.executable, [str(PROJECT_ROOT / "main.py"), cfg_path])
        self._last_cfg_path = cfg_path

    def auto_fit(self):
        if self._X is None:
            return
        self._proc.kill()
        self._proc.waitForFinished(500)
        self._auto_fit_dir = tempfile.mkdtemp(prefix="interp_autoopt_")
        state = dict(self._state)
        state["output_dir"] = self._auto_fit_dir
        state["save_diagnostics"] = False
        state["resolution_m"] = 200.0
        state["export_formats"] = ["nc"]
        state.pop("kriging_preset", None)
        state.pop("gp_preset", None)
        preset = self._on_slider_preset
        if self._engine == "kriging" and preset:
            state["kriging_model"] = preset.get("model", "spherical")
        state["engine_mode"] = self._engine
        cfg_path = _write_temp_config(state)
        self.statusMessage.emit("Auto-fitting parameters…")
        self._proc.start(sys.executable, [str(PROJECT_ROOT / "main.py"), cfg_path])
        self._last_cfg_path = cfg_path

    def _on_stdout(self):
        data = self._proc.readAllStandardOutput().data().decode("utf-8", errors="replace")
        for line in data.splitlines():
            if line.strip():
                self.logLine.emit(line.strip())

    def _on_run_complete(self, exit_code):
        try:
            os.unlink(getattr(self, "_last_cfg_path", ""))
        except OSError:
            pass
        if exit_code != 0:
            self.statusMessage.emit(f"Engine exited with code {exit_code}")
            self._auto_fit_dir = None
            return
        # ── Auto-fit: read optimized parameters from the temp output dir ──
        af_dir = getattr(self, "_auto_fit_dir", None)
        if af_dir:
            self._auto_fit_dir = None
            # derive_output_dir creates {base}/{input_stem}/ — read from there
            in_stem = Path(self._state.get("input_filepath", "data")).stem
            params_file = Path(af_dir) / in_stem / f"parameters_{self._engine}.json"
            if params_file.exists():
                import json as _json
                with open(params_file) as f:
                    params = _json.load(f)
                self.paramsReady.emit(params)
                self.statusMessage.emit("Auto-fit complete — controls updated.")
            else:
                self.statusMessage.emit("Auto-fit finished — no parameters found.")
            return
        # ── Full run: load the bundle ──
        bd = getattr(self, "_bundle_dir", None)
        if not bd:
            return
        result = {"mode": self._engine}
        import pandas as pd
        grid_file = Path(bd) / "grid.npz"
        if grid_file.exists():
            with np.load(grid_file) as gz:
                result["grid"] = {k: gz[k] for k in gz.files}
        cv_file = Path(bd) / f"cv_results_{self._engine}.csv"
        if cv_file.exists():
            cv_df = pd.read_csv(cv_file)
            result["cv_df"] = cv_df
            resid = cv_df["Observed"] - cv_df["Predicted"]
            result["mae"] = float(resid.abs().mean())
            result["rmse"] = float((resid ** 2).mean() ** 0.5)
            ss_res = float((resid ** 2).sum())
            ss_tot = float(((cv_df["Observed"] - cv_df["Observed"].mean()) ** 2).sum())
            result["r2"] = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            z = cv_df["Z_Score"].to_numpy()
            result["mean_sspe"] = float(np.mean(z ** 2)) if len(z) else float("nan")
            result["rmss"] = float(np.sqrt(result["mean_sspe"]))
        self._last_full = result
        self.resultReady.emit(result)
        self.metricsUpdated.emit(result)
        self.statusMessage.emit("Full-resolution result ready.")

    def export(self, folder, want_grid=True, want_cv=True):
        if self._last_full is None:
            return []
        saved = []
        if want_grid and "grid" in self._last_full:
            np.savez_compressed(Path(folder) / "predicted_grid.npz",
                                **self._last_full["grid"])
            saved.append("predicted_grid.npz")
        if want_cv and self._last_full.get("cv_df") is not None:
            self._last_full["cv_df"].to_csv(Path(folder) / "cv_results.csv", index=False)
            saved.append("cv_results.csv")
        return saved


def _write_temp_config(state: dict) -> str:
    import yaml
    cfg = build_config(state)
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="interp_ui_")
    with os.fdopen(fd, "w") as f:
        yaml.dump(cfg, f)
    return path
