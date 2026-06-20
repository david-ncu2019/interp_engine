# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Geospatial interpolation engine supporting Ordinary Kriging (PyKrige) and Gaussian Process regression (scikit-learn). Takes scattered 2D point observations (X, Y, Value) and produces a predicted surface on a regular grid or at specified prediction points. Designed for geoscientific applications: groundwater levels, elevation, ore grades.

## Environment

- **Conda environment**: `fafalab2` — always use `conda run -n fafalab2 python ...` for execution
- **Python**: 3.12 (check with `conda run -n fafalab2 python --version`)
- **Key dependencies**: pykrige, numpy, scipy, scikit-learn, optuna, matplotlib, xarray, pandas, pyyaml, statsmodels, libpysal, esda
- **Platform**: Windows 10, paths use backslashes in shell but forward slashes work in Python

## Commands

```bash
# Run the CLI engine with a config file
conda run -n fafalab2 python main.py config.yaml

# Launch the Tkinter UI
conda run -n fafalab2 python ui/app.py

# Launch the PySide6 UI (primary UI) — dev/CLI form
conda run -n fafalab2 python kriging.py
# End-user one-click launch: double-click "Launch Kriging App.vbs" in the repo root

# Run the deterministic optimizer test (should complete in <2s)
conda run -n fafalab2 python _test_deterministic.py

# Run the Qhull edge-case fix test
conda run -n fafalab2 python _test_qhull_fix.py

# Syntax check a file
conda run -n fafalab2 python -c "import py_compile; py_compile.compile('src/engines/kriging.py', doraise=True)"

# Import check
conda run -n fafalab2 python -c "from src.engines.kriging import AnisotropicKriging; print('OK')"
```

No test suite or linter is configured. Verify changes with syntax check + import check + `_test_deterministic.py`.

## Architecture

### Two execution modes

1. **CLI** (`main.py config.yaml`): reads YAML config, runs full pipeline (load → preprocess → fit → predict → export → diagnostics). Config schema documented in `package_guidance.md`.
2. **PySide6 UI** (`python -m ui_pyside.main_window`): primary UI. Accordion sidebar + 2×2 Matplotlib plot grid. Full runs spawn `main.py` via QProcess; live preview runs inline on the main thread at 40-cell resolution. See `ui_pyside/` section below.
3. **Tkinter UI** (`ui/app.py`): legacy 4-tab wizard, still functional. Builds config dict, writes temp YAML, launches `main.py` via `ui/engine_runner.py`.

### Data pipeline (main.py)

`load_input_data` → `check_and_clean_duplicates` → `analyze_trend`/`TrendProcessor` → `NormalScoreTransform` → **fit engine** → **predict** → inverse-transform → `export_grid`/point export → diagnostic plots

The fit step has a three-way dispatch (main.py ~line 501):
- `kriging_preset` → `model.fit_with_known_params()` (UI slider values, no optimization)
- `kriging_model` → `model.fit_deterministic()` (fast WLS-only DE + directional anisotropy)
- Neither → `model.fit()` (legacy Optuna TPE search)

GP has a similar preset vs Optuna split.

### Engine classes

- **`src/engines/kriging.py`** — `AnisotropicKriging(BaseEstimator, RegressorMixin)`. Three fit paths: `fit_deterministic()` (fast 3-stage: WLS DE → directional anisotropy → single CV), `fit()` (Optuna), `fit_with_known_params()`. Wraps PyKrige `OrdinaryKriging`. Supports 11 variogram models (6 native PyKrige + 5 custom: stable, circular, rational-quadratic, matern_32, matern_52). Custom models defined as module-level functions. `VARIOGRAM_EVALUATORS` dict maps model name → callable. `HAS_ALPHA` set marks models needing an extra alpha parameter.

- **`src/engines/gp.py`** — `RotatedGPR(BaseEstimator, RegressorMixin)`. Coarse angle scan → Optuna TPE → L-BFGS-B polish. Supports Matern-3/2, Matern-5/2, RBF kernels with anisotropic rotation.

- **`src/engines/fast_kriging.py`** — `ok_predict(X_tr, y_tr, X_te, model_name, params)`. Stand-alone NumPy/`np.linalg.solve` ordinary-kriging predictor; byte-equivalent to PyKrige's `OrdinaryKriging.execute('points', …)` (verified to 1e-11 across spherical/exponential/gaussian/matern_32/stable, isotropic and anisotropic). Used by `_quick_cv_rmss` (search-loop CV) and `ui/live_predictor.compute_preview` (slider-drag live preview) — 20× faster than going through PyKrige for these hot paths. Final-Run prediction still uses PyKrige via `_get_ok_instance`.

### Utilities

- **`utils.py`** (project root, not in `src/`): empirical variogram computation (`compute_empirical_variogram` — GSLIB-style binning: `n_lags` + `lag_width` (separation) + `lag_tolerance` (± window). Centers sit at `lag_width*(k+0.5)`; default `lag_tolerance = lag_width/2` reproduces classic contiguous edges. Supports directional mode (`directions=` list) and reuse of precomputed pairwise data via `_dists`/`_diff_sq` kwargs. When BOTH `n_lags` and `lag_width` are passed, the GSLIB branch fires and `max_lag = n_lags × lag_width` overrides any caller-supplied `max_lag`), spatial block folds (`make_spatial_block_folds`), adaptive lag parameters (`auto_lag_params`), all diagnostic plot functions. Imported by both engines and main.py.

- **`src/preprocessor.py`**: `TrendProcessor` (polynomial detrending with F-test auto-detection), `NormalScoreTransform` (rank-based Gaussian transform). Uses statsmodels, libpysal/esda for Moran's I.

- **`src/geometry.py`**: `generate_prediction_grid(X, Y, config)` — builds 2D meshgrid bounded by a buffered convex hull. Uses `qhull_options="QJ"` (joggle mode) for collinear/degenerate data, with a bounding-box fallback. Returns `(X_grid, Y_grid, mask_flat, grid_shape, hull_vertices_closed)`.

### UI layers

#### `ui/` — Tkinter (legacy)

- **`ui/app.py`**: Main Tkinter window, 4-tab notebook. Manages shared `state` dict.
- **`ui/variogram_panel.py`**: `KrigingPanel` and `GPPanel` — live variogram plot with parameter sliders, n_lags spinbox, empirical variogram overlay. Both panels cache `_X_data`/`_y_data` and recompute variograms on parameter changes.
- **`ui/engine_runner.py`**: `EngineRunner` (full run) and `AutoOptimizeRunner` (optimization only). Converts UI state dict → config YAML via `build_config()`, launches `main.py` as subprocess, streams stdout to a queue.

#### `ui_pyside/` — PySide6 (primary)

- **`__init__.py`**: Hardens Windows PATH on import (removes rogue conda-env entries) before any Qt/NumPy load, preventing `0xc06d007f` access violations. Must be the first import.
- **`main_window.py`**: `GeospatialApp(QMainWindow)` — orchestrates QSplitter layout (sidebar | 2×2 plot grid). Creates `WorkspaceController`, wires all signals, restores geometry from `QSettings`.
- **`workspace_controller.py`**: `WorkspaceController(QObject)` — pure signal/slot mediator with no widget references. Key signals: `logLine`, `statusMessage`, `metricsUpdated`, `resultReady`, `paramsReady`. Full run: spawns `QProcess(conda run ... main.py <tmp.yaml>)`, streams stdout to `LogConsole`, on exit loads `grid.npz` + `cv_results.csv` and emits `resultReady`. Live preview: 300ms debounce timer → inline `_compute_preview()` (40-cell grid, main thread, no subprocess).
- **`accordion_sidebar.py`**: `AccordionSidebar` with 4 `CollapsibleSection` widgets (animated expand/collapse).
- **`animated_slider.py`**: `AnimatedSlider` — `QSlider` + `QDoubleSpinBox` pair with 300ms debounce on `valueChanged`.
- **`dockable_plot.py`** / **`mpl_canvas.py`**: `PlotPanel(QWidget)` wraps `MplCanvas(FigureCanvasQTAgg)` with NavigationToolbar and scroll-wheel zoom.
- Reuses `src/`, `utils.py`, `main.py`, and `ui/live_predictor.py` unchanged.

### Import conventions

- `main.py` does `from src.data_loader import ...`, `from src.engines.kriging import ...`, `from utils import ...`
- Engine files do `from utils import ...` (lazy, inside methods to avoid circular imports)
- UI files add `PROJECT_ROOT` to `sys.path` at import time

### Config flow

UI state dict → `build_config()` (engine_runner.py) → temp YAML → `main.py` reads YAML → pipeline. The config keys `kriging.model` and `kriging.preset_params` control the three-way fit dispatch. Kriging binning controls: `kriging.n_lags`, `kriging.lag_width`, `kriging.lag_tolerance` (0 = auto in UI / None in engine), and `kriging.lock_n_lags`, `kriging.lock_max_lag` (locks gate the Optimize-time search; default `True` and `False` respectively).

## Test data

`test_data/S1_Isotropic.csv` through `S8_SGS_HighNugget.csv` — synthetic datasets with known properties (isotropic, anisotropic, high-nugget, clustered, etc.). Each has a `_ground_truth.csv` companion. `gwl_drop_2015.csv` is real groundwater data from Taiwan.

## Performance considerations

- `fit_deterministic()` uses a 4-stage pipeline: optional lag-parameter search (only when `lock_n_lags=False` or `lock_max_lag=False`), Stage-1 empirical variogram, Stage-2 WLS multi-start `least_squares`, Stage-3 directional anisotropy from 4 directional variograms, Stage-4 optional CV (`compute_cv=True`, off by default). Completes in ~0.8–1.1 s on 500 points across all 11 models (was ~2.1 s before fast_kriging). Stage-2 bounds `psill` and `range` by what the empirical variogram can support (`psill_upper = max(data_var, max(emp_sv))*2`, `range_upper = min(max_dist*1.5, emp_lags[-1]*1.5)`) — prevents the Cressie-weighting degenerate-basin pathology when the user binning truncates the variogram below the structural range.
- The legacy `fit()` Optuna path runs CV inside every trial and is much slower — it exists for backward compatibility.
- Live preview (`ui/live_predictor.compute_preview`, 40-cell grid, N=500): ~75 ms, down from ~1.5 s by routing through `fast_kriging.ok_predict` instead of `AnisotropicKriging.predict` → PyKrige.
- PyKrige `OrdinaryKriging` solves O(N^3) per prediction call and is still used for the final Run (large-grid prediction). Keep N reasonable (<5000 points) or expect long prediction times.
- Windows stdout encoding: main.py reconfigures stdout to UTF-8 at startup. engine_runner.py passes `PYTHONIOENCODING=utf-8` to subprocesses. This prevents UnicodeEncodeError from matplotlib/Unicode print statements in piped mode.

## Pitfalls

- `conda run -n fafalab2 python -c "..."` does not support multiline Python strings well on Windows. Put everything on one line or use a script file.
- Both `KrigingPanel` and `GPPanel` in variogram_panel.py have similar `__init__` signatures. When editing one, include enough context (class name / docstring) to disambiguate.
- `utils.py` lives at project root, not in `src/`. Engine files import it with `from utils import ...` inside methods to avoid circular imports at module load time.
- Windows DLL contamination: if a conda env with MKL/Intel OpenMP (e.g. `gemini_env`) is on PATH when `fafalab2` Python starts, numpy/scipy crash with `0xc06d007f`. `ui_pyside/__init__.py` purges bad PATH entries before Qt import. For subprocess runs via QProcess the PATH is inherited clean from `conda run`. For dev/debug outside `conda run`, set `KMP_DUPLICATE_LIB_OK=TRUE` as a temporary workaround.
- **PyKrige's native variogram parameterization differs from ours.** PyKrige's `_make_variogram_parameter_list` treats the user-supplied `psill` as TOTAL sill and internally uses partial sill = `psill − nugget`. Its native `exponential` uses *effective* range and divides by 3 internally; `gaussian` uses `range·4/7`. Our `_eval_*` functions in `kriging.py` use partial sill + mathematical range directly. `fast_kriging._eval_gamma` mirrors PyKrige's conventions by dispatching to PyKrige's native funcs for native models (and our `VARIOGRAM_EVALUATORS` for custom models). When passing fitted params from our WLS path into PyKrige (in `_get_ok_instance`), the same convention conversion happens implicitly — be aware when comparing variogram curves.
- **Use `np.linalg.solve`, NOT `scipy.linalg.lu_factor` or `scipy.linalg.lapack.dgesv` for the kriging system.** On the bordered indefinite OK matrix in scipy 1.17, both scipy wrappers exhibit a progressive per-call slowdown (~22 ms first call → ~1700 ms by call 7) — pure memory-pollution pattern. `np.linalg.solve` stays flat at ~8 ms across thousands of calls. Same LAPACK underneath; only the wrappers differ. `fast_kriging.ok_predict` uses `np.linalg.solve` for this reason.
- **Stage-1 in `fit_deterministic` must gate `lag_width`/`lag_tolerance` on `lock_max_lag`.** When `lock_max_lag=False` the search loop picks `best_ml`; if the caller's stale `lag_width` is then passed to Stage-1, `compute_empirical_variogram`'s GSLIB branch silently overrides `max_lag` with `n_lags × stale_lag_width`, discarding the search result. The gate is: `lag_width=lag_width if lock_max_lag else None` (and same for `lag_tolerance`).
