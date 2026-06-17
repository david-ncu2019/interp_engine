# Unified Workspace with Live Interpolation Results — Design

_Date: 2026-06-17 · Project: interp_engine (Tkinter geospatial interpolation app)_

## Context

The app's current 4-tab flow (`Data | Method | Model | Run & Results`) separates the variogram controls (Model tab) from the prediction output (Run & Results tab), and the output is **static PNG files written to disk** during a subprocess run — the UI only shows a text log. The user wants:

1. **Interpolation results shown directly in the app**, interactive (zoom/pan/drag), not saved images.
2. **Most important:** the prediction surface should **update in real time as the variogram is modified**, with a **toggle** to enable/disable this live mode (off → manual refresh by button).
3. **Export-only output** — files are written only when the user clicks Export, never automatically.
4. **mean_SSPE + RMSS** surfaced in the results summary (they are computed in CV today but not shown on the full-run path).
5. License to **redesign the panel cleaner**.

This supersedes the earlier "Task 9" plan (which embedded the existing static plots into a read-only Run & Results tab). Work already completed and committed remains valid and is reused (see "Already done").

## Already done (committed, reused as-is)

- `4021997` — the 4 diagnostic `plot_*` in `utils.py` accept an optional `fig=` and draw onto a passed-in Figure without closing it (CLI behavior unchanged). **Reused** to render into the Workspace canvases.
- `e0b4d7d` — `SubTabCanvas` (in `ui/variogram_panel.py`) has a `NavigationToolbar2Tk` + mouse-wheel zoom. **Reused** as the interactive canvas for the Workspace plots.
- Tasks 1–8 (linear/power fix, dashboards, fonts, CV stdout parsing) — unaffected.

## Goal

Replace the Model + Run & Results tabs with a single **Workspace** tab: variogram controls on the left, a live, interactive results area on the right. Dragging a variogram slider updates the prediction surface, uncertainty map, and variogram-fit plot in real time (when live mode is on). Full-resolution results and all file output happen only on explicit user action.

## Architecture — two prediction paths

The current full pipeline runs `main.py` as a **subprocess** that writes files; it is far too slow for per-slider updates (kriging is O(N³) per grid solve; GP can take minutes). So the design has two distinct paths:

| | **Live preview** (toggle ON) | **Full run** (Run/Refresh button) |
|---|---|---|
| Where | **In-process** (UI imports the engine classes) | Subprocess `main.py` (existing `EngineRunner`) |
| Grid | Coarse (~40×40), inside the data's convex hull | Full resolution from config |
| Params | Current slider values (preset; **no optimization**) | Full config pipeline |
| Preprocessing | Skipped (raw X/y; optional detrend reused if cheap) | Full (duplicates, trend, NST) |
| Triggers | Debounced (~0.3 s) on slider/model change | Button click |
| Updates | Surface + uncertainty + variogram-fit | Everything incl. CV dashboard + metrics |
| Writes files | Never | Never automatically — only via Export |
| Speed target | ~0.2–0.5 s kriging; ~1–3 s GP | seconds–minutes |

**In-process predictor (new, small module `ui/live_predictor.py`):**
- Build a coarse grid over the data bounds (reuse the convex-hull/grid helpers from `main.py`/geometry utilities; coarse `resolution`).
- Kriging: construct `AnisotropicKriging`, call `fit_with_known_params(X, y, preset)` then `predict(grid_pts, return_std=True)` → `mean, std`. (`predict` already returns std — `kriging.py:641`.)
- GP: `RotatedGPR.fit_with_known_params(X, y, preset)` then `predict(..., return_std=True)`.
- Returns `mean`, `std` reshaped to the coarse meshgrid (+ obs X/Y for overlay).
- Runs on a **worker thread**; result marshalled back to the Tk main thread via the existing `after()`/queue pattern so the UI never blocks. A new request supersedes any in-flight one (debounce + cancel-stale).

**Live scope (confirmed):** on slider drag → re-render **surface, uncertainty, variogram-fit** only. **CV dashboard + metrics bar (MAE/RMSE/R²/mean_SSPE/RMSS) recompute only on Run/Refresh** (k-fold CV is too slow per drag).

## UI — the Workspace tab

Tabs become **`Data | Method | Workspace`** (Model + Run & Results removed; their content folds into Workspace). `Data` (file/columns) and `Method` (engine + options) are unchanged setup steps.

**`ui/workspace.py` — `WorkspacePanel(ttk.Frame)`**, two columns:

- **Left (controls, ~40%):** engine indicator; the existing variogram **`LabeledSlider`** controls (Model/Range/Sill/Nugget/Angle/Anisotropy/Alpha) for kriging, or kernel controls for GP — reuse the slider widgets/logic from `KrigingPanel`/`GPPanel`; **`☑ Live update`** checkbutton; **Auto-fit** button (existing Auto Optimize); **Run full-res** button; **Export…** button.
- **Right (results, ~60%):** a 2×2 grid of `SubTabCanvas`-style interactive canvases —
  - **Prediction surface (mean)** — live
  - **Uncertainty (std)** — live
  - **Variogram fit** (empirical + model curve) — live
  - **CV dashboard** (Z-score scatter + spatial Z-score map) — updates on Run/Refresh
  - plus a **metrics bar**: `MAE · RMSE · R² · mean_SSPE · RMSS` (updates on Run/Refresh).

Rendering reuses the `fig=`-aware `utils.plot_prediction_surface` (split into the two single-axis maps, or shown as its native side-by-side) and `utils.plot_cv_dashboard`; the variogram-fit reuses the Model-tab drawing. All canvases get the zoom/pan toolbar + wheel-zoom (already in `SubTabCanvas`).

**Live update flow:** slider `on_change` → mark surface/uncertainty/variogram "stale" → debounce timer (~300 ms) → if live on, kick the in-process predictor on a worker thread → on result, redraw the 3 live canvases. If live off, the slider still updates the variogram-fit curve (cheap, pure-numpy) but the surface shows a "Refresh to update" hint until the button is pressed.

## Export (on demand only)

An **Export…** dialog (checkboxes): **Figures** (the 4 plots → PNG/SVG/PDF), **Predicted grid** (full-res mean/std → nc/tif/csv), **CV results** (csv) → user picks a folder. Figures re-render via `plot_*(save_path=…)` from the last **full-res** result (so exports are publication-res, not the coarse preview). Grid/CV come from the last full run's data. **Nothing is written until Export is clicked** — the full-run path writes its outputs to a private temp dir (not the user's folder), mirroring the data back to the UI; Export copies/saves from there.

## SSPE / RMSS in the summary

`mean_SSPE = mean(Z²)` and `RMSS = √mean_SSPE` (Z = standardized CV residual; both already computed per-point in `cv_results`). Add them to:
- the engine's full-run printed summary in `main.py` (next to MAE/RMSE/R², ~line 669–675), computed from `cv_df["Z_Score"]`;
- the Workspace metrics bar (from the full-run CV data).
Interpretation shown in a tooltip/label: ≈1 well-calibrated, >1 over-confident (uncertainty too small), <1 under-confident.

## Components / files

| File | Responsibility | Change |
|---|---|---|
| `ui/live_predictor.py` (new) | In-process coarse-grid predict (kriging/GP) on a worker thread; debounce/cancel-stale | new |
| `ui/workspace.py` (new) | `WorkspacePanel`: left controls + right 2×2 interactive results + metrics bar + live toggle | new |
| `ui/variogram_panel.py` | Reuse `LabeledSlider`, `SubTabCanvas`, variogram-fit drawing; extract shared slider/kernel logic the Workspace imports | refactor for reuse |
| `ui/app.py` | Replace Model + Run&Results tabs with the single Workspace tab; wire Data/Method → Workspace | modify |
| `ui/engine_runner.py` | Full run writes to a temp bundle (no user-folder output); UI reads it for the full-res result + Export | modify |
| `main.py` | UI-mode: emit results to temp (grid arrays as `.npz`, cv csv, params json), suppress auto PNGs/grid; add mean_SSPE/RMSS to summary | modify |
| `utils.py` | already `fig=`-aware (done) | none |

## Performance & risks

- **Live responsiveness:** coarse grid + debounce + worker thread + cancel-stale keeps the UI fluid. If GP coarse-predict is too slow on a given dataset, the predictor reports timing and the UI can auto-suggest turning live off for GP. (Kriging is the primary live target.)
- **Thread safety:** prediction off the Tk thread; redraw marshalled back via `after()`. Never touch Tk widgets from the worker.
- **`matplotlib.use("Agg")` in `utils.py`** must not clobber the interactive TkAgg backend — the `plot_*` draw onto passed-in Figures (backend-agnostic); the UI owns TkAgg.
- **Coarse vs full divergence:** the preview is explicitly lower-res; the metrics/CV and Export always come from the full run, so nothing misleading is exported.
- **Determinism / correctness of the live preview:** uses the same `fit_with_known_params` + `predict` the engine uses, just on a coarse grid — so the preview is a faithful low-res version of the full result for the same params.

## Verification (end-to-end)

1. Launch UI → 3 tabs (`Data | Method | Workspace`).
2. Load data, pick Kriging → Workspace shows controls left, plots right.
3. Toggle **Live update** on; drag Range/Sill/Nugget → surface + uncertainty + variogram-fit re-render within ~0.3–0.5 s, smoothly, no freeze.
4. Toggle live off; drag a slider → surface shows "Refresh to update"; click **Run full-res** → all 4 plots + metrics (incl. mean_SSPE/RMSS) update.
5. Confirm the user's output folder stays **empty** after runs; **Export…** writes the chosen artifacts (figures/grid/CV) to a chosen folder; exported figures are full-res.
6. Zoom/pan/wheel work on every plot.
7. Switch to GP → same behavior (live may be slower; still functional).
8. CLI unaffected: `main.py config.yaml` still writes all A–I PNGs + grid + CV to `out_dir`.
9. `_test_deterministic.py` still PASS.
