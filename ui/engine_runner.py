"""
engine_runner.py  –  Subprocess bridge between the Tkinter UI and main.py.

Responsibilities
----------------
1. Write a temporary config YAML from the UI state dict.
2. Launch `python main.py <tmp_config>` as a child process.
3. Stream stdout line-by-line into a thread-safe queue.
4. Parse the completed output JSON / CSV for the results panel.
"""

import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import yaml


# ── Root of the project (parent of ui/) ───────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAIN_PY      = PROJECT_ROOT / "main.py"

# Matches the fast optimizer's single CV line, e.g.
#   [fast-opt] CV done in 2.13s: RMSE=0.4521, mean_SSPE=0.987
_CV_RE = re.compile(
    r"\[fast-opt\] CV done in [\d.]+s: RMSE=([\d.]+), mean_SSPE=([\d.]+)")

# Set True while debugging; flip to False once all errors are resolved
VERBOSE = True


# ─────────────────────────────────────────────────────────────────────────────
# Config builder
# ─────────────────────────────────────────────────────────────────────────────

def build_config(state: dict) -> dict:
    """
    Convert the UI state dict into a main.py-compatible config dict.

    State keys (set by the UI tabs):
      input_filepath, col_x, col_y, col_value,
      ground_truth_filepath (optional),
      output_dir,
      export_formats (list),
      resolution_m,
      convex_hull_buffer,
      detrend_enabled, detrend_order,
      nst_enabled,
      min_separation,
      engine_mode  ('kriging' | 'gp'),
      -- Kriging preset (optional, filled by slider values):
      kriging_preset  (dict with keys: model, psill, range, nugget, angle_deg,
                       anisotropy_ratio, [alpha])
      -- Kriging Optuna params:
      kriging_n_trials, kriging_n_splits, kriging_max_anisotropy,
      -- GP preset (optional):
      gp_preset  (dict with keys: kernel_type, length_scale_major,
                  anisotropy_ratio, angle_deg, signal_variance,
                  nugget_variance, jitter_alpha)
      -- GP Optuna params:
      gp_n_trials, gp_max_anisotropy, gp_angle_min, gp_angle_max,
      gp_random_state,
      save_diagnostics,
      netcdf_z_dim_name,
    """
    cfg: dict = {}

    # ── Input ──────────────────────────────────────────────────────────────────
    cfg["input"] = {
        "filepath": state["input_filepath"],
        "format":   "",
        "columns": {
            "x":     state.get("col_x", "X"),
            "y":     state.get("col_y", "Y"),
            "value": state.get("col_value", "Value"),
        },
        "ground_truth_filepath": state.get("ground_truth_filepath") or None,
    }

    # ── Geometry ───────────────────────────────────────────────────────────────
    cfg["geometry"] = {
        "resolution_m":               float(state.get("resolution_m", 50.0)),
        "convex_hull_buffer_percent": float(state.get("convex_hull_buffer", 10.0)),
    }

    # ── Preprocessing ──────────────────────────────────────────────────────────
    cfg["preprocessing"] = {
        "detrend": {
            "auto_detect": state.get("detrend_auto", True),
            "enabled":     state.get("detrend_enabled", True),
            "order":       int(state.get("detrend_order", 1)),
        },
        "nst": {
            "enabled": state.get("nst_enabled", None),
        },
        "duplicates": {
            "min_separation": state.get("min_separation", None),
        },
    }

    # ── Engine ─────────────────────────────────────────────────────────────────
    mode = state.get("engine_mode", "kriging")
    engine_cfg: dict = {"mode": mode}

    if mode == "kriging":
        k: dict = {
            "max_anisotropy": float(state.get("kriging_max_anisotropy", 10.0)),
            "n_splits":       int(state.get("kriging_n_splits", 3)),
            "n_trials":       int(state.get("kriging_n_trials", 300)),
            "n_lags":         int(state.get("kriging_n_lags", 12)),
        }
        if state.get("kriging_model"):
            k["model"] = state["kriging_model"]
        if state.get("kriging_preset"):
            k["preset_params"] = state["kriging_preset"]
        engine_cfg["kriging"] = k

    else:  # gp
        g: dict = {
            "max_anisotropy":  float(state.get("gp_max_anisotropy", 15.0)),
            "angle_min":       float(state.get("gp_angle_min", 0.0)),
            "angle_max":       float(state.get("gp_angle_max", 180.0)),
            "n_optuna_trials": int(state.get("gp_n_trials", 300)),
            "random_state":    state.get("gp_random_state", 42),
        }
        if state.get("gp_preset"):
            g["preset_params"] = state["gp_preset"]
        engine_cfg["gp"] = g

    cfg["engine"] = engine_cfg

    # ── Output ─────────────────────────────────────────────────────────────────
    cfg["output"] = {
        "base_directory":    state.get("output_dir", "output"),
        "netcdf_z_dim_name": state.get("netcdf_z_dim_name", "Depth"),
        "save_diagnostics":  state.get("save_diagnostics", True),
        "formats":           state.get("export_formats", ["nc"]),
    }

    return cfg


def write_temp_config(state: dict) -> str:
    """Write config to a temp YAML file; return the file path."""
    cfg = build_config(state)
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="interp_ui_")
    with os.fdopen(fd, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Subprocess runner
# ─────────────────────────────────────────────────────────────────────────────

class EngineRunner:
    """
    Launches main.py in a background thread and streams stdout to a queue.

    Usage
    -----
        runner = EngineRunner(state, log_queue)
        runner.start()
        # poll runner.is_alive(), drain log_queue, check runner.result
    """

    def __init__(self, state: dict, log_queue: "queue.Queue[str]"):
        self.state      = state
        self.log_queue  = log_queue
        self.result: Optional[dict] = None   # populated on success
        self.error:  Optional[str]  = None   # populated on failure
        self._proc:  Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self.start_time: Optional[float] = None
        self.elapsed: float = 0.0

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.start_time = time.time()
        self._thread.start()

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def cancel(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()

    def _run(self):
        config_path = write_temp_config(self.state)
        if VERBOSE:
            self.log_queue.put(f"[VERBOSE] config written → {config_path}")
            self.log_queue.put(f"[VERBOSE] subprocess: {sys.executable} {MAIN_PY}")
            self.log_queue.put(f"[VERBOSE] cwd: {PROJECT_ROOT}")
            try:
                with open(config_path) as _f:
                    for _line in _f:
                        self.log_queue.put(f"[VERBOSE cfg] {_line.rstrip()}")
            except Exception:
                pass
        try:
            _env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
            self._proc = subprocess.Popen(
                [sys.executable, str(MAIN_PY), config_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(PROJECT_ROOT),
                env=_env,
            )
            _tail = []
            for line in self._proc.stdout:
                stripped = line.rstrip("\n")
                self.log_queue.put(stripped)
                _tail.append(stripped)
                if len(_tail) > 30:
                    _tail.pop(0)
            self._proc.wait()
            self.elapsed = time.time() - self.start_time
            if VERBOSE:
                self.log_queue.put(f"[VERBOSE] returncode: {self._proc.returncode}")

            if self._proc.returncode == 0:
                self.result = self._parse_results()
            else:
                tail_text = "\n".join(_tail[-15:])
                self.error = (
                    f"Engine exited with code {self._proc.returncode}\n\n"
                    f"Last output:\n{tail_text}"
                )
        except Exception as exc:
            self.error = str(exc)
        finally:
            try:
                os.unlink(config_path)
            except OSError:
                pass
            self.log_queue.put(None)  # sentinel: run finished

    def _parse_results(self) -> dict:
        """Read parameters JSON and cv_results CSV from the output directory."""
        from pathlib import Path
        import csv

        state   = self.state
        mode    = state.get("engine_mode", "kriging")
        out_dir = Path(state.get("output_dir", "output"))

        # output is written to {base_dir}/{input_stem}/
        input_stem = Path(state["input_filepath"]).stem
        _base  = out_dir if out_dir.is_absolute() else PROJECT_ROOT / out_dir
        run_dir = _base / input_stem
        if VERBOSE:
            self.log_queue.put(f"[VERBOSE] looking for results in: {run_dir}")

        result: dict = {"mode": mode, "run_dir": str(run_dir), "elapsed": self.elapsed}

        # parameters JSON
        params_file = run_dir / f"parameters_{mode}.json"
        if params_file.exists():
            with open(params_file) as f:
                result["params"] = json.load(f)

        # cv_results CSV — compute mean metrics
        cv_file = run_dir / f"cv_results_{mode}.csv"
        if cv_file.exists():
            rows = []
            with open(cv_file, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(row)
            result["cv_rows"] = rows
            if rows:
                # try to extract aggregate metrics from the last row or compute mean
                def _mean(col):
                    vals = [float(r[col]) for r in rows if col in r and r[col] not in ("", "nan")]
                    return sum(vals) / len(vals) if vals else None

                result["rmse"] = _mean("rmse") or _mean("RMSE")
                result["mae"]  = _mean("mae")  or _mean("MAE")
                result["r2"]   = _mean("r2")   or _mean("R2") or _mean("r_squared")

        return result


# ─────────────────────────────────────────────────────────────────────────────
# Auto-Optimize runner (Optuna only — no diagnostics, no prediction)
# ─────────────────────────────────────────────────────────────────────────────

class AutoOptimizeRunner:
    """
    Runs Optuna optimization with save_diagnostics=False to quickly find
    best parameters, then reads parameters_*.json to populate the UI sliders.
    """

    def __init__(self, state: dict, log_queue: "queue.Queue[str]"):
        self.state      = state
        self.log_queue  = log_queue
        self.params:     Optional[dict] = None
        self.cv_summary: Optional[dict] = None
        self.error:      Optional[str]  = None
        self._thread:    Optional[threading.Thread] = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self):
        # Build a lightweight config: coarse grid, diagnostics off, isolated temp dir
        opt_state = dict(self.state)
        opt_state["save_diagnostics"] = False
        opt_state["resolution_m"]     = 200.0   # coarse grid for speed
        opt_state["export_formats"]   = ["nc"]  # must be non-empty; temp dir is cleaned up
        # Preserve the selected model for deterministic optimization,
        # but remove the full preset (no manual slider override during auto-opt)
        kriging_preset = opt_state.pop("kriging_preset", None)
        if kriging_preset and "model" in kriging_preset:
            opt_state["kriging_model"] = kriging_preset["model"]
        opt_state.pop("gp_preset", None)

        # Isolated output dir → no pollution of the user's real output folder
        _tmp_out = tempfile.mkdtemp(prefix="interp_autoopt_")
        opt_state["output_dir"] = _tmp_out

        config_path = write_temp_config(opt_state)
        if VERBOSE:
            self.log_queue.put(f"[VERBOSE] auto-opt config → {config_path}")
            self.log_queue.put(f"[VERBOSE] temp output dir → {_tmp_out}")
            self.log_queue.put(f"[VERBOSE] subprocess: {sys.executable} {MAIN_PY}")
            try:
                with open(config_path) as _f:
                    for _line in _f:
                        self.log_queue.put(f"[VERBOSE cfg] {_line.rstrip()}")
            except Exception:
                pass
        try:
            _env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
            proc = subprocess.Popen(
                [sys.executable, str(MAIN_PY), config_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(PROJECT_ROOT),
                env=_env,
            )
            _tail = []
            for line in proc.stdout:
                stripped = line.rstrip("\n")
                self.log_queue.put(stripped)
                _tail.append(stripped)
                if len(_tail) > 30:
                    _tail.pop(0)
                _m = _CV_RE.search(stripped)
                if _m:
                    self.cv_summary = {
                        "RMSE": float(_m.group(1)),
                        "Mean SSPE": float(_m.group(2)),
                    }
            proc.wait()
            if VERBOSE:
                self.log_queue.put(f"[VERBOSE] returncode: {proc.returncode}")

            if proc.returncode == 0:
                self.params = self._read_params(opt_state)
            else:
                tail_text = "\n".join(_tail[-15:])
                self.error = (
                    f"Auto-optimize exited with code {proc.returncode}\n\n"
                    f"Last output:\n{tail_text}"
                )
        except Exception as exc:
            self.error = str(exc)
        finally:
            try:
                os.unlink(config_path)
            except OSError:
                pass
            shutil.rmtree(_tmp_out, ignore_errors=True)
            self.log_queue.put(None)

    def _read_params(self, state: dict) -> dict:
        mode       = state.get("engine_mode", "kriging")
        out_dir    = Path(state.get("output_dir", "output"))
        input_stem = Path(state["input_filepath"]).stem
        _base      = out_dir if out_dir.is_absolute() else PROJECT_ROOT / out_dir
        run_dir    = _base / input_stem
        params_file = run_dir / f"parameters_{mode}.json"
        if VERBOSE:
            self.log_queue.put(f"[VERBOSE] params file → {params_file}  exists={params_file.exists()}")
        if params_file.exists():
            with open(params_file) as f:
                return json.load(f)
        return {}
