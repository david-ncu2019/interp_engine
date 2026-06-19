"""WorkspaceController — pure signal/slot mediator, no widgets."""
import os, sys, json, re, tempfile, csv, shutil
from pathlib import Path
import numpy as np

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_TEMP_ROOT = PROJECT_ROOT / ".temp"

from ui.engine_runner import build_config


class WorkspaceController(QObject):
    logLine = Signal(str)
    statusMessage = Signal(str)
    metricsUpdated = Signal(dict)       # {mae, rmse, r2, mean_sspe, rmss}
    resultReady = Signal(dict)          # full result dict with grid/cv_df/params
    paramsReady = Signal(dict)           # optimized parameters from auto-fit
    dataLoaded = Signal()               # X/y are now available
    progressChanged = Signal(int, str)  # (percent 0-100, label) — determinate subprocess progress
    busyStarted = Signal(str)           # (label) — indeterminate long op started
    busyFinished = Signal()             # long op finished (success or failure)

    def __init__(self, parent=None):
        super().__init__(parent)
        # Clean up stale temp dirs from previous sessions
        if _TEMP_ROOT.exists():
            for _d in list(_TEMP_ROOT.iterdir()):
                try:
                    if _d.is_dir():
                        shutil.rmtree(_d, ignore_errors=True)
                    else:
                        _d.unlink()
                except OSError:
                    pass
        else:
            _TEMP_ROOT.mkdir(parents=True, exist_ok=True)

        self._X = None
        self._y = None
        self._engine = "kriging"
        self._live = True
        self._last_full = None
        self._auto_fit_dir = None        # temp dir for auto-fit (read params after)
        self._on_slider_preset = {
            "model": "spherical", "psill": 5.0, "range": 300,
            "nugget": 0.5, "angle_deg": 0.0, "anisotropy_ratio": 1.0}

        # Live preview and auto-fit run in raw space (detrend/NST forced off)
        # so the variogram plot, sliders, and optimized params are always in
        # consistent units. The full "Run Interpolation" respects the user's
        # preprocessing choices from the sidebar — the subprocess applies
        # transform → fit → back-transform so results stay in original units.
        self._state = {"engine_mode": "kriging", "output_dir": "output",
                       "save_diagnostics": True, "export_formats": ["nc"],
                       "detrend_auto": False, "detrend_enabled": False,
                       "detrend_order": 1,
                       "nst_enabled": False}

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
        # Do NOT auto-render a prediction on bare load. The user hasn't fit or
        # run anything yet; main_window draws the empirical variogram instead.
        # Live preview is preserved for genuine interaction (slider / Run / Optimize).
        self.statusMessage.emit(f"Loaded {len(self._y)} points.")

    def set_engine(self, mode: str):
        self._engine = mode
        self._state["engine_mode"] = mode

    def set_live(self, enabled: bool):
        self._live = enabled

    def set_preprocessing(self, detrend_enabled: bool, detrend_order: int,
                          detrend_auto: bool, nst_enabled):
        """Update preprocessing flags. None for nst_enabled means auto-detect."""
        self._state["detrend_enabled"] = detrend_enabled
        self._state["detrend_order"] = detrend_order
        self._state["detrend_auto"] = detrend_auto
        self._state["nst_enabled"] = nst_enabled  # True/False/None

    def on_slider_change(self, preset: dict):
        self._on_slider_preset = preset
        self._debounce.start()

    # ------------------------------------------------------------------
    # Live preview (inline, main thread)
    def _compute_preview(self):
        if self._X is None or not self._live:
            return
        from ui.live_predictor import compute_preview
        # Inline preview blocks the main thread; flip the busy indicator on so it
        # paints before the (brief) compute, then off when done.
        self.busyStarted.emit("Computing live preview…")
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
            msg = str(exc).split("\n")[0]  # first line only (Qhull diagnostics are noisy)
            self.statusMessage.emit(f"Preview failed: {msg}")
        finally:
            self.busyFinished.emit()

    # ------------------------------------------------------------------
    # Full run (subprocess via QProcess)
    def run_full(self):
        if self._X is None:
            return
        self._proc.kill()
        self._proc.waitForFinished(500)
        self._bundle_dir = tempfile.mkdtemp(prefix="uirun_", dir=str(_TEMP_ROOT))
        state = dict(self._state)
        state["ui_mode"] = True
        state["bundle_dir"] = self._bundle_dir
        state["engine_mode"] = self._engine
        # Run uses the current slider values directly (instant, no optimization).
        # build_config() reads the FLAT keys kriging_preset / gp_preset — writing the
        # nested kriging.preset_params here would be silently dropped, leaving the
        # config with no model and no preset → main.py falls through to the slow
        # legacy Optuna fit().
        preset = self._on_slider_preset
        if self._engine == "kriging" and preset:
            state["kriging_preset"] = preset
        elif self._engine == "gp" and preset:
            state["gp_preset"] = preset
        cfg_path = _write_temp_config(state)
        self.statusMessage.emit("Running full-resolution interpolation…")
        self.busyStarted.emit("Running full-resolution interpolation…")
        self.progressChanged.emit(0, "Starting…")
        # Apply DLL-safe subprocess environment
        env = QProcessEnvironment()
        for k, v in self._proc_env.items():
            env.insert(k, v)
        self._proc.setProcessEnvironment(env)
        self._proc.start(sys.executable, [str(PROJECT_ROOT / "main.py"), cfg_path])
        self._last_cfg_path = cfg_path

    def auto_fit(self):
        if self._X is None:
            return
        self._proc.kill()
        self._proc.waitForFinished(500)
        self._auto_fit_dir = tempfile.mkdtemp(prefix="autoopt_", dir=str(_TEMP_ROOT))
        state = dict(self._state)
        # Auto-fit MUST run in raw space so returned params match the
        # raw empirical variogram and sliders (NST/detrend transforms
        # would change the units of psill/nugget/range).
        state["detrend_enabled"] = False
        state["detrend_auto"] = False
        state["nst_enabled"] = False
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
        self.busyStarted.emit("Auto-fitting parameters…")
        self.progressChanged.emit(0, "Starting…")
        # Apply DLL-safe subprocess environment
        env = QProcessEnvironment()
        for k, v in self._proc_env.items():
            env.insert(k, v)
        self._proc.setProcessEnvironment(env)
        self._proc.start(sys.executable, [str(PROJECT_ROOT / "main.py"), cfg_path])
        self._last_cfg_path = cfg_path

    # Stage markers main.py prints, e.g. "[5/7] Fitting model ...".
    _STAGE_RE = re.compile(r"\[(\d+)\s*/\s*(\d+)\]\s*(.*)")

    def _on_stdout(self):
        data = self._proc.readAllStandardOutput().data().decode("utf-8", errors="replace")
        for line in data.splitlines():
            s = line.strip()
            if not s:
                continue
            self.logLine.emit(s)
            m = self._STAGE_RE.search(s)
            if m:
                k, total = int(m.group(1)), int(m.group(2))
                pct = int(round(100.0 * k / max(total, 1)))
                label = m.group(3).strip() or f"Step {k}/{total}"
                self.progressChanged.emit(min(pct, 100), f"[{k}/{total}] {label}")

    def _on_run_complete(self, exit_code):
        # Reset/hide the progress indicator no matter how this run ends.
        self.progressChanged.emit(0, "")
        self.busyFinished.emit()
        try:
            os.unlink(getattr(self, "_last_cfg_path", ""))
        except OSError:
            pass

        # ── Auto-fit: params are saved BEFORE fragile downstream steps ──
        # (prediction / NetCDF export may fail after params are on disk).
        # Read them regardless of exit code.
        af_dir = getattr(self, "_auto_fit_dir", None)
        if af_dir is not None:
            self._auto_fit_dir = None
            in_stem = Path(self._state.get("input_filepath", "data")).stem
            params_file = Path(af_dir) / in_stem / f"parameters_{self._engine}.json"
            if params_file.exists():
                import json as _json
                with open(params_file) as f:
                    params = _json.load(f)
                self.paramsReady.emit(params)
                self.statusMessage.emit("Auto-fit complete — controls updated.")
            else:
                self.statusMessage.emit(
                    f"Auto-fit failed (exit {exit_code}) — no parameters found.")
            # Clean up the temp output dir
            shutil.rmtree(af_dir, ignore_errors=True)
            return

        # ── Full run: require success ──
        if exit_code != 0:
            self.statusMessage.emit(f"Engine exited with code {exit_code}")
            return
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
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="ui_", dir=str(_TEMP_ROOT))
    with os.fdopen(fd, "w") as f:
        yaml.dump(cfg, f)
    return path
