# PySide6 Minimalist UI Redesign — Design Spec

_Date: 2026-06-19 · Project: interp_engine · Branch: feat/unified-workspace_

## Context

The current UI is a Tkinter application with a 3-tab wizard (Data | Method | Workspace). The Workspace tab embeds 4 matplotlib canvases with variogram controls in a 2-column layout. The user wants a **professional, minimalist** redesign using **PySide6 + QDarkStyle** with **detachable plots for multi-monitor teaching**, and architecture ready for **future 3D kriging visualization with PyVista**. The existing computation engines (kriging, GP, live predictor) and `fig=`-aware `utils.plot_*` functions are reused unchanged.

## Design Decisions (locked)

1. **Full PySide6 rewrite, 2D first** — rebuild all UI in PySide6 + `FigureCanvasQTAgg`. Backend engines + plotting functions reused as-is. PyVista/PyVistaQt installed but 3D panel is a placeholder; activated later.
2. **QMainWindow + QDockWidget** — dockable, movable, floatable panels with persistent layout via QSettings.
3. **QDarkStyle** — classic dark IDE theme, dark + light variants.
4. **Accordion sidebar** ("Option C") — fixed left sidebar (~320px) with collapsible sections. Only one section open at a time (QButtonGroup exclusive). Height animates 150ms.
5. **Detachable plot panels ("Option C′")** — each of the 4 matplotlib plots lives in a `DockablePlot(QDockWidget)` that can be undocked and dragged to a second monitor. When teaching: controls stay on Monitor 1 (laptop), plots fill Monitor 2 (projector). QSettings remembers layout.

## Window Architecture

```
QMainWindow (~1400×900, remembers geometry via QSettings)
├── QMenuBar
│   ├── File      (Open, Set Output Folder, Export…, Exit)
│   ├── View      (toggle actions for all docks, Reset Layout, Compact Mode)
│   ├── Analysis  (Run Full-res ⌘R, Auto-fit ⌘⇧R)
│   └── Help      (About)
│
├── CENTRAL: QSplitter (horizontal)
│   ├── LEFT: AccordionSidebar (QScrollArea, fixed ~320px)
│   │   ├── CollapsibleSection("Data & Setup")       ← default open
│   │   │   ├── FilePicker (QLineEdit + Browse QPushButton)
│   │   │   ├── ColumnSelector × 3 (QComboBox: X, Y, Value)
│   │   │   ├── EngineSelector (QButtonGroup + 2× QRadioButton)
│   │   │   └── ExportFormats (QCheckBox × 3: NetCDF, GeoTIFF, CSV)
│   │   │
│   │   ├── CollapsibleSection("Variogram Controls")  ← auto-expands after data load
│   │   │   ├── ModelDropdown (QComboBox, Kriging only)
│   │   │   ├── AnimatedSlider × 6 (Range, Sill, Nugget, Angle, Aniso, Alpha)
│   │   │   ├── LiveToggle (QCheckBox: "Live update")
│   │   │   └── ActionBar: [▶ Run full-res] [⟳ Auto-fit] [💾 Export…]
│   │   │
│   │   ├── CollapsibleSection("Engine Options")     ← collapsed by default
│   │   │   └── GP-specific: kernel type radios, trial count, LS bounds
│   │   │
│   │   └── CollapsibleSection("Log")                ← collapsed by default
│   │       └── LogConsole (QPlainTextEdit, read-only, dark bg, monospace)
│   │
│   └── RIGHT: PlotGrid (QGridLayout 2×2)
│       ├── [0,0] DockablePlot("Prediction Surface")  → FigureCanvasQTAgg
│       ├── [0,1] DockablePlot("Uncertainty (std)")   → FigureCanvasQTAgg
│       ├── [1,0] DockablePlot("Variogram Fit")        → FigureCanvasQTAgg
│       └── [1,1] DockablePlot("CV Dashboard")         → FigureCanvasQTAgg
│
└── QStatusBar
    └── MetricsLabel: "MAE 0.45 · RMSE 0.61 · R² 0.87 · mean_SSPE 1.04 · RMSS 1.02 · ✓ Ready"
```

## Component Details

### `CollapsibleSection(QWidget)`
- Header: QPushButton with ▸/▾ indicator + title text, flat style
- Content: QFrame with QVBoxLayout, hidden/shown via QPropertyAnimation (height, 150ms)
- Exclusive mode: QButtonGroup ensures only one section open at a time
- Signals: `toggled(bool)`, `expanded()`, `collapsed()`

### `AnimatedSlider(QWidget)` — port of `LabeledSlider`
- QLabel (fixed width 120px) + QSlider(Qt.Horizontal) + QDoubleSpinBox (80px)
- Slider drag → instant update to spinbox + signal emission via debounce timer (100ms)
- Spinbox edit → clamps to range, updates slider
- Signals: `valueChanged(float)`

### `DockablePlot(QDockWidget)`
- Contains MplCanvas + a ⤢/⤡ button in the title bar
- `setFeatures(DockWidgetMovable | DockWidgetClosable | DockWidgetFloatable)`
- On float: window resizes to fill 80% of available screen
- On dock: re-attaches to PlotGrid at original position
- Saves/restores geometry and dock state via QSettings
- Object names: "SurfaceDock", "UncertaintyDock", "VarioDock", "CVDock"

### `MplCanvas(QWidget)` — port of `SubTabCanvas`
- Contains FigureCanvasQTAgg + NavigationToolbar2QT + Export QPushButton
- Layout: QVBoxLayout(toolbar, canvas, button_row)
- Mouse-wheel zoom handler via `mpl_connect("scroll_event", …)`
- `draw_surface(fig, X_grid, Y_grid, mean, std, X_obs, Y_obs, hull, title)` → clears, draws contourf+scatter+colorbar, draw_idle()
- `draw_cv(fig, cv_df)` → calls utils.plot_cv_dashboard(cv_df, fig=fig), draw_idle()
- `draw_variogram(fig, lags, sv, model_name, preset)` → empirical + model curve

### `FilePicker(QWidget)`
- QLineEdit (read-only, shows selected path) + QPushButton("Browse…")
- Supports drag-and-drop of CSV files
- Signals: `fileSelected(str)`

### `ExportDialog(QDialog)`
- 3× QCheckBox: Figures (PNG/SVG/PDF), Grid (.npz), CV (.csv)
- QPushButton: "Choose Folder & Export"
- Uses saved last directory from QSettings

### `WorkspaceController(QObject)`
- Pure signal/slot mediator — owns no widgets
- Holds: `_X` (ndarray), `_y` (ndarray), `_last_result` (dict), `_debounce_timer` (QTimer, 300ms single-shot)
- Slots: `_on_param_change()`, `_load_data(path)`, `_run_full()`, `_auto_fit()`, `_export()`
- `_on_run_complete()` → loads bundle (grid.npz, cv csv, params json) → `show_full_result(result)`
- Subprocess management via `QProcess` (replaces threading.Thread + subprocess.Popen)

## Data flow (unchanged from Tkinter version)

```
UI widgets → WorkspaceController → [live: compute_preview()] OR [full: QProcess → main.py → temp bundle]
                                 → MplCanvas.draw_*()
                                 → StatusBar metrics
```

Signal-based navigation replaces Tk variable traces and `after()` polling. The `utils.plot_*` functions and `live_predictor.compute_preview` are reused with zero changes.

## Files

### New (`ui_pyside/`)

| File | Class(es) | Lines (est.) | Replaces |
|---|---|---|---|
| `__init__.py` | — (env setup) | 10 | — |
| `main_window.py` | `GeospatialApp(QMainWindow)` | ~120 | `ui/app.py` |
| `accordion_sidebar.py` | `CollapsibleSection`, `AccordionSidebar` | ~100 | sidebar portion of `workspace.py` |
| `animated_slider.py` | `AnimatedSlider(QWidget)` | ~70 | `LabeledSlider` |
| `mpl_canvas.py` | `MplCanvas(QWidget)` | ~80 | `SubTabCanvas` |
| `dockable_plot.py` | `DockablePlot(QDockWidget)` | ~60 | plot grid portion of `workspace.py` |
| `file_picker.py` | `FilePicker(QWidget)` | ~40 | Data tab file/column UI |
| `log_console.py` | `LogConsole(QPlainTextEdit)` | ~30 | `_log_text` |
| `export_dialog.py` | `ExportDialog(QDialog)` | ~50 | `_ExportDialog` |
| `workspace_controller.py` | `WorkspaceController(QObject)` | ~180 | app callback methods + `EngineRunner` |
| `theme.py` | `apply_theme(app)` | ~20 | — |
| `resources/icons/` | SVG icon set | — | — |

### Reused (unchanged)

| File | What |
|---|---|
| `ui/live_predictor.py` | `compute_preview(engine, X, y, preset, n_cells)` |
| `ui/engine_runner.py` | `build_config(state)`, `_CV_RE`, `_BUNDLE_RE` |
| `utils.py` | `plot_*` functions (already `fig=`-aware) |
| `src/engines/*.py` | Kriging, GP, geometry |
| `main.py` | `run_pipeline()` — subprocess entry point, ui_mode bundle |

### Legacy (preserved, not deleted)

`ui/app.py`, `ui/workspace.py`, `ui/variogram_panel.py` — preserved during migration for backward compatibility. Can be deleted after PySide6 app is stable.

## Dependencies to add

```
conda install -n fafalab2 conda-forge::pyvista conda-forge::pyvistaqt   # 3D future
pip install QDarkStyle     # theming
```

Already present: PySide6 6.11.1, matplotlib 3.10.9, numpy 1.26.4

**⚠️ DLL PATH hazard:** `ui_pyside/__init__.py` MUST harden DLL resolution (force `fafalab2`'s `Library/bin` to front of PATH + `os.add_dll_directory`) — see the diagnosed `gemini_env` contamination in `tk-import-verification-caveat` memory. This applies to both the GUI process and any QProcess subprocess.

## Interaction details

- **Accordion:** clicking a collapsed section header (▸) → currently-open section collapses (150ms height animation) → clicked section expands (150ms). Clicking the open section's header collapses it (leaves none open — sidebar shrinks).
- **Sliders:** drag → spinbox updates immediately, `valueChanged` signal fires. A 100ms QTimer debounce passes the final value to `_on_param_change()`. If another slider moves within 100ms, the timer resets.
- **Live preview:** `_on_param_change()` → if Live toggle ON, starts 300ms debounce timer → on timeout, calls `compute_preview()` inline on the main thread → draws surface + uncertainty + variogram canvases. CV dashboard + metrics unchanged until Run.
- **Run full-res:** validates file+columns → disables Run button, shows spinner in status bar → QProcess launches `main.py` in ui_mode → stdout lines appended to LogConsole → on finish, `show_full_result()` loads bundle → draws all 4 canvases + updates metrics.
- **Export:** opens ExportDialog → user picks folder + checkboxes → figures rendered via `plot_*(fig=canvas.fig, save_path=…)` at 150 dpi, grid copied from bundle, CV written as CSV.
- **Detach plot:** click ⤢ on a DockablePlot title bar → plot becomes floating QMainWindow (not QDialog — gets its own window frame, min/max/close buttons). Drag to Monitor 2. Resize freely. View menu → "Reset Layout" → all plots re-dock in the 2×2 grid.
- **Keyboard shortcuts:** ⌘R = Run, ⌘⇧R = Auto-fit, ⌘E = Export, ⌘\ = toggle compact mode (hide sidebar), ⌘1–4 = expand accordion sections 1–4.

## Verification (end-to-end)

1. Launch `python ui_pyside/main_window.py` → QDarkStyle themed window with accordion sidebar + 2×2 plot grid + status bar.
2. Load `test_data/S2_Anisotropic.csv` via FilePicker → column selectors auto-populate, Data section collapses, Variogram section expands.
3. Slider drag → variogram fit plot updates. With Live ON → surface + uncertainty update after ~300ms debounce. No freeze.
4. Click Run full-res → log streams to LogConsole, spinner in status bar. On completion: CV dashboard populates, metrics bar shows MAE/RMSE/R²/mean_SSPE/RMSS. Output folder stays empty.
5. Click ⤢ on Prediction Surface → plot floats. Drag to Monitor 2 → resizes to fill. Slider changes still update it live. Click ⤡ → re-docks to grid.
6. Export → choose folder, check Figures+Grid+CV → files written. Verify exported figure matches on-screen.
7. Switch to GP → Variogram Controls section shows kernel radios instead of model dropdown. Live preview still works (slower, acceptable).
8. CLI unchanged: `main.py config.yaml` still writes all A–I PNGs + grid + CV to `out_dir`. `_test_deterministic.py` PASS.
9. View menu → Compact Mode → sidebar collapses to 0, plots fill window. Toggle back → sidebar returns.
