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
        self._gt_result = None           # cached ground truth comparison result
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

    def set_grid_mode(self, use_custom: bool):
        """Toggle between auto-grid (False) and custom prediction points (True)."""
        self._state["_grid_mode_custom"] = use_custom
        if not use_custom:
            # Clear custom points when switching back to auto-grid
            self._state.pop("prediction_points_filepath", None)
            self._state.pop("prediction_points_col_x", None)
            self._state.pop("prediction_points_col_y", None)

    def set_prediction_points_file(self, filepath: str, col_x: str, col_y: str):
        """Set the custom prediction points CSV and its X/Y columns."""
        self._state["prediction_points_filepath"] = filepath
        self._state["prediction_points_col_x"] = col_x
        self._state["prediction_points_col_y"] = col_y

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


    # ------------------------------------------------------------------
    # Ground truth comparison (in-process, no subprocess)
    def compare_ground_truth(self, gt_filepath: str, gt_col: str) -> dict:
        """Fit model with current slider params, predict at GT points,
        compute validation metrics. Returns dict for GroundTruthWindow."""
        import pandas as pd
        gt_df = pd.read_csv(gt_filepath)
        # Use same X/Y column names as the training data
        col_x = self._state.get("col_x", "X")
        col_y = self._state.get("col_y", "Y")
        gt_X = gt_df[col_x].to_numpy(dtype=float)
        gt_Y = gt_df[col_y].to_numpy(dtype=float)
        gt_obs = gt_df[gt_col].to_numpy(dtype=float)

        preset = self._on_slider_preset
        if self._engine == "kriging":
            from src.engines.kriging import AnisotropicKriging
            from ui.live_predictor import _kriging_params_from_preset
            model_name = preset.get("model", "spherical")
            model = AnisotropicKriging()
            # _kriging_params_from_preset maps angle_deg→angle and
            # anisotropy_ratio→scaling — the keys _get_ok_instance requires
            # (it indexes params['angle']/['scaling'] directly, no defaults).
            model.fit_with_known_params(
                self._X, self._y, model_name,
                _kriging_params_from_preset(preset))
        else:
            from src.engines.gp import RotatedGPR
            model = RotatedGPR()
            model.fit_with_known_params(self._X, self._y, preset)

        # predict() takes a single (N, 2) array of points + return_std.
        gt_pts = np.column_stack([gt_X, gt_Y])
        gt_pred, gt_std = model.predict(gt_pts, return_std=True)
        residuals = gt_pred - gt_obs

        n = len(gt_obs)
        mae = float(np.mean(np.abs(residuals)))
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        bias = float(np.mean(residuals))
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((gt_obs - gt_obs.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        # SSPE/RMSS (standardized squared prediction error)
        z_scores = residuals / np.maximum(gt_std, 1e-10)
        mean_sspe = float(np.mean(z_scores ** 2)) if len(z_scores) else float("nan")
        rmss = float(np.sqrt(mean_sspe))

        metrics = {
            "mae": mae, "rmse": rmse, "r2": r2, "bias": bias,
            "mean_sspe": mean_sspe, "rmss": rmss, "n": n,
        }
        self._gt_result = {
            "metrics": metrics, "residuals": residuals,
            "gt_obs": gt_obs, "gt_pred": gt_pred,
            "gt_X": gt_X, "gt_Y": gt_Y, "gt_col": gt_col,
        }
        return self._gt_result

    # ------------------------------------------------------------------
    # Save / Load config
    def build_config_from_current_state(self) -> dict:
        """Build a full config YAML dict from the current UI state.
        Compatible with main.py CLI and Save/Load Config."""
        import yaml as _yaml
        preset = self._on_slider_preset
        cfg = {
            "input": {
                "filepath": self._state.get("input_filepath", ""),
                "columns": {
                    "x": self._state.get("col_x", "X"),
                    "y": self._state.get("col_y", "Y"),
                    "value": self._state.get("col_value", "Value"),
                },
            },
            "geometry": {
                "resolution_m": float(self._state.get("resolution_m", 50.0)),
                "convex_hull_buffer_percent": float(
                    self._state.get("convex_hull_buffer", 10.0)),
            },
            "preprocessing": {
                "detrend": {
                    "auto_detect": self._state.get("detrend_auto", False),
                    "enabled": self._state.get("detrend_enabled", False),
                    "order": int(self._state.get("detrend_order", 1)),
                },
                "nst": {
                    "enabled": self._state.get("nst_enabled", False),
                },
            },
            "engine": {"mode": self._engine},
            "output": {"save_diagnostics": True,
                       "netcdf_z_dim_name": "Depth",
                       "formats": ["nc"]},
        }
        if self._engine == "kriging":
            n_lags = int(self._state.get("kriging_n_lags", 12))
            cfg["engine"]["kriging"] = {
                "n_lags": n_lags,
                "preset_params": preset.copy(),
            }
        else:
            cfg["engine"]["gp"] = {
                "preset_params": preset.copy(),
            }
        return cfg

    def apply_config(self, config: dict):
        """Apply a loaded config YAML dict to the controller state and
        return UI update instructions for main_window."""
        engine_mode = config.get("engine", {}).get("mode", "kriging")
        self._engine = engine_mode
        self._state["engine_mode"] = engine_mode

        updates = {"engine_mode": engine_mode}

        # Preprocessing
        pp = config.get("preprocessing", {})
        detrend = pp.get("detrend", {})
        nst = pp.get("nst", {})
        updates["detrend_enabled"] = detrend.get("enabled", False)
        updates["detrend_order"] = detrend.get("order", 1)
        updates["detrend_auto"] = detrend.get("auto_detect", False)
        nst_val = nst.get("enabled", False)
        updates["nst_enabled"] = nst_val

        self._state["detrend_enabled"] = updates["detrend_enabled"]
        self._state["detrend_order"] = updates["detrend_order"]
        self._state["detrend_auto"] = updates["detrend_auto"]
        self._state["nst_enabled"] = nst_val

        # Engine params
        eng = config.get("engine", {})
        if engine_mode == "kriging":
            kc = eng.get("kriging", {})
            preset = kc.get("preset_params", {})
            updates["kriging_model"] = preset.get("model", "spherical")
            updates["n_lags"] = kc.get("n_lags", 12)
            updates["sliders"] = {
                "Range": preset.get("range", 300.0),
                "Sill (psill)": preset.get("psill", 5.0),
                "Nugget": preset.get("nugget", 0.5),
                "Angle (°)": preset.get("angle_deg", 0.0),
                "Anisotropy ×": preset.get("anisotropy_ratio", 1.0),
                "Alpha": preset.get("alpha", 1.0),
            }
        else:
            gc = eng.get("gp", {})
            preset = gc.get("preset_params", {})
            updates["gp_kernel"] = preset.get("kernel_type", "matern_52")
            updates["sliders"] = {
                "Range": preset.get("length_scale_major", 300.0),
                "Angle (°)": preset.get("angle_deg", 0.0),
                "Anisotropy ×": preset.get("anisotropy_ratio", 1.0),
            }

        # Update the controller's slider preset
        if engine_mode == "kriging" and "sliders" in updates:
            s = updates["sliders"]
            self._on_slider_preset = {
                "model": updates.get("kriging_model", "spherical"),
                "psill": s.get("Sill (psill)", 5.0),
                "range": s.get("Range", 300.0),
                "nugget": s.get("Nugget", 0.5),
                "angle_deg": s.get("Angle (°)", 0.0),
                "anisotropy_ratio": s.get("Anisotropy ×", 1.0),
                "alpha": s.get("Alpha", 1.0),
            }
        elif engine_mode == "gp" and "sliders" in updates:
            s = updates["sliders"]
            self._on_slider_preset = {
                "kernel_type": updates.get("gp_kernel", "matern_52"),
                "length_scale_major": s.get("Range", 300.0),
                "anisotropy_ratio": s.get("Anisotropy ×", 1.0),
                "angle_deg": s.get("Angle (°)", 0.0),
            }

        return updates


def _write_temp_config(state: dict) -> str:
    import yaml
    cfg = build_config(state)
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="ui_", dir=str(_TEMP_ROOT))
    with os.fdopen(fd, "w") as f:
        yaml.dump(cfg, f)
    return path
