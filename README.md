# Interpolation Engine — Geostatistical Interpolation Workbench

An interactive desktop app (PySide6) for spatial interpolation of scattered 2D point data.
Load `(X, Y, Value)` observations — groundwater levels, elevation, ore grades, etc. — fit a
variogram interactively, and produce a predicted surface with an uncertainty map. Supports
**Ordinary Kriging** (PyKrige) and **Gaussian Process regression** (scikit-learn).

---

## Features

- **Accordion sidebar** — Data, Preprocessing, Variogram Controls, Run Options, and Validation,
  all in one tidy panel.
- **Live variogram preview** — adjust sill / range / nugget / anisotropy with sliders and watch
  the model curve update against the empirical variogram in real time.
- **2×2 plot grid** — prediction surface, uncertainty (std), variogram, and a cross-validation
  dashboard.
- **Detachable, closable & reopenable panels** — drag any plot panel out to a second monitor
  (great for teaching/presenting); it stays live. Close a panel with its ✕ and bring it back
  from **View ▸ Panels**, or **View ▸ Reset Layout** to restore the 2×2 grid.
- **Optional cross-validation** — a checkbox lets you skip leave-one-out / k-fold CV (which is
  O(N³) and slow on large datasets) when you only need the interpolated surface.
- **Ground-truth validation** — compare predictions against a held-out dataset (scatter,
  spatial error map, metrics, residual histogram).
- **Save / Load config** and **flexible export** of figures and data.

---

## Prerequisites

- [Miniconda or Anaconda](https://docs.conda.io/en/latest/miniconda.html)
- Developed and tested on **Windows 10**. (The geospatial dependencies are pulled from
  `conda-forge`, which is the reliable way to install them on Windows.)

---

## Installation

```bash
git clone https://github.com/david-ncu2019/interp_engine.git
cd interp_engine
conda env create -f environment.yml
conda activate interp-engine
```

This creates a conda environment named **`interp-engine`** with everything the app needs.

---

## Running the interface

```bash
conda activate interp-engine
python -m ui_pyside.main_window
```

---

## Quick start

A couple of small sample datasets ship in `test_data/`:

1. Launch the app (command above).
2. In the **Data** section, load `test_data/S1_Isotropic.csv` and set the columns to
   `X`, `Y`, `Value`.
3. Click **Optimize Parameters** to auto-fit the variogram (or fine-tune the sliders by hand and
   watch the live preview).
4. Click **Run Interpolation** — the prediction surface and uncertainty map appear in the plot
   grid.
5. Try `test_data/S2_Anisotropic_45deg.csv` to see directional (anisotropic) behaviour.

---

## Using the interface

- **Sidebar sections** expand/collapse; work top to bottom: Data → Preprocessing →
  Variogram Controls → Run Options → Validation.
- **Run Options ▸ Compute cross-validation** is **off by default** (faster, interpolation only).
  Turn it on to populate the CV Dashboard with MAE / RMSE / R².
- **Second monitor:** drag a plot panel's title bar onto another screen. With **Live update** on,
  changing a slider redraws the detached panel too.
- **Closing / reopening panels:** click a panel's ✕ to hide it; reopen it from **View ▸ Panels**.
  **View ▸ Reset Layout** restores the default 2×2 arrangement and reopens everything. Your
  layout is remembered between sessions.

---

## Troubleshooting

- **App crashes immediately on Windows (error `0xc06d007f`)** — this is caused by a conflicting
  conda environment (with Intel MKL/OpenMP) being on your `PATH`. The app already hardens its
  `PATH` on startup (`ui_pyside/__init__.py`), so simply run it from an activated
  `interp-engine` environment. If you run individual modules outside the env for debugging, set
  `KMP_DUPLICATE_LIB_OK=TRUE`.
- **Large datasets feel slow** — leave **Compute cross-validation** off; kriging prediction is
  O(N³), so keep the point count reasonable (well under ~5000 points).

---

## Command-line mode (advanced)

The engine can also run headless from a YAML config:

```bash
python main.py config.yaml
```

`config.yaml` defines the input file, engine, variogram model, and output settings.
