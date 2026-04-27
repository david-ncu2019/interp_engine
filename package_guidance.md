# Interpolation Engine — Configuration Guide

This document explains every parameter in `config.yaml` with intuitive descriptions
grounded in the actual source code.

---

## Table of Contents

1. [input](#1-input)
2. [prediction_points](#2-prediction_points-optional)
3. [geometry](#3-geometry-grid-mode-only)
4. [preprocessing](#4-preprocessing)
   - 4.1 [detrend](#41-detrend)
   - 4.2 [nst (Normal-Score Transform)](#42-nst-normal-score-transform)
   - 4.3 [duplicates](#43-duplicates)
5. [engine](#5-engine)
   - 5.1 [Kriging parameters](#51-kriging-parameters)
   - 5.2 [GP parameters](#52-gp-gaussian-process-parameters)
6. [output](#6-output)

---

## 1. `input`

```yaml
input:
  filepath: test_data_2/input_dem.shp
  format: ""
  columns:
    x:
    y:
    value: grid_code
  ground_truth_filepath: null
```

| Parameter | What it does |
|-----------|-------------|
| `filepath` | Path to your training data. Supported: `.csv`, `.xlsx`, `.shp`. |
| `format` | Force the file format (`csv`, `excel`, `shapefile`). Leave blank to auto-detect from file extension. |
| `columns.x` / `columns.y` | Column names for X and Y coordinates. For shapefiles, leave blank — coordinates are extracted from the geometry automatically. |
| `columns.value` | Column name of the variable you want to interpolate (e.g. elevation, concentration). |
| `ground_truth_filepath` | Optional path to a dense reference dataset. If provided, the engine produces a comparison plot (`H_comparison_*.png`) showing predicted vs. truth. Set to `null` to skip. |

---

## 2. `prediction_points` (optional)

```yaml
prediction_points:
  filepath: test_data_2/output_points.shp
  columns:
    x:
    y:
```

When this section is present, the engine runs in **Point Mode**: it predicts only at
the specific locations listed in this file rather than building a regular grid.

**Use Point Mode when** you need predictions at exact survey locations, borehole collars,
monitoring stations, or any pre-defined set of points.

**Omit this entire section** (or comment it out) to use **Grid Mode**, which generates a
regular prediction grid over the convex hull of your sample points.

---

## 3. `geometry` (Grid Mode only)

```yaml
geometry:
  resolution_m: 50.0
  convex_hull_buffer_percent: 10.0
```

| Parameter | What it does |
|-----------|-------------|
| `resolution_m` | Cell size of the prediction grid, in the same units as your coordinates (usually metres). Smaller = finer grid = slower prediction. A good starting value is 1/50 of the study area width. |
| `convex_hull_buffer_percent` | How much to expand the prediction grid beyond the outermost sample points. 10 means the grid extends 10% of the hull's bounding box beyond the data boundary. This avoids edge effects at the data perimeter. |

---

## 4. `preprocessing`

### 4.1 `detrend`

```yaml
preprocessing:
  detrend:
    auto_detect: true
    enabled: true
    order: 1
```

**What is detrending?**

Both Kriging and GP assume that the spatial field has a *stationary* mean — roughly
speaking, that there is no overall slope or gradient across the study area. If your data
has a clear trend (e.g. elevation rising from south to north), that large-scale gradient
will dominate the variogram and the engine will mistake the trend for spatial correlation,
producing poor predictions.

Detrending fits a polynomial surface to your data, subtracts it, models the leftover
*residuals* with the spatial engine, and then adds the trend back after prediction.
This way the engine only has to handle the spatially random part of the field.

| Parameter | What it does |
|-----------|-------------|
| `auto_detect` | When `true`, the engine runs an F-test and Moran's I on your data to decide automatically whether a trend exists. This is the recommended setting. |
| `enabled` | Only consulted when `auto_detect: false`. Forces detrending on (`true`) or off (`false`) regardless of what the statistics say. |
| `order` | Degree of the polynomial surface. `1` = a tilted plane (linear trend). `2` = a curved bowl (quadratic trend). Use `1` unless you have a physically justified reason for a higher-order surface — higher orders can over-fit. |

**Auto-detection rule** (from `preprocessor.py`):
- Detrending is recommended when: F-test p-value < 0.05 **AND** the polynomial R² > 5%.
- Both conditions must hold to avoid stripping out a trend that is statistically detectable
  but practically negligible.

---

### 4.2 `nst` (Normal-Score Transform)

```yaml
preprocessing:
  nst:
    enabled: null
```

**What is the Normal-Score Transform?**

Kriging and GP both assume the values they model are approximately Gaussian (bell-shaped).
Many real-world datasets are skewed — for example, permeability, ore grades, or pollutant
concentrations can have long right tails where a few extreme values dominate.

When you feed skewed data to these engines three problems arise:
1. The variogram sill is inflated by the outliers.
2. The engine assigns too much weight to extreme values when interpolating.
3. Uncertainty estimates are wrong because they assume symmetry.

The Normal-Score Transform (NST) is the geostatistical industry-standard fix. It rank-maps
your data to a standard normal distribution N(0,1) before modelling, then back-transforms
predictions to original units after. The transform is rank-preserving — it does not change
the relative ordering of your data, only the shape of the distribution.

| `enabled` value | Effect |
|-----------------|--------|
| `null` | Auto-detect: applies NST only when **all three** of these hold: Shapiro-Wilk p < 0.05 **AND** \|skewness\| > 0.5 **AND** \|excess kurtosis\| > 1.0. All three must fail simultaneously to avoid triggering NST on data that is only mildly non-Gaussian. |
| `true` | Always apply NST, regardless of the data distribution. |
| `false` | Never apply NST. |

**Practical advice:** Leave as `null` for most datasets. Only force `true` if you know
your data is heavily skewed (e.g. log-normal ore grades) and the auto-detect keeps
missing it.

---

### 4.3 `duplicates`

```yaml
preprocessing:
  duplicates:
    min_separation: null
```

**Why this matters:** If two sample points are at (or very near) the same location with
different values, the covariance matrix becomes nearly singular — Kriging crashes with a
linear algebra error and GP silently produces nonsense.

The engine resolves this automatically:
- **Exact duplicates** (distance = 0): averaged into a single point.
- **Near-duplicates** (distance < `min_separation`): each point receives a tiny random
  jitter of `min_separation / 10` to push them apart.

| Parameter | What it does |
|-----------|-------------|
| `min_separation` | Minimum allowed distance between two distinct sample points. `null` = auto-calculate from the data (0.1 × median nearest-neighbour distance, floored at 0.001). Override with a specific number if you know your data's GPS precision. |

---

## 5. `engine`

```yaml
engine:
  mode: kriging
```

`mode` selects the spatial interpolation algorithm: `kriging` or `gp`.

**Kriging** is the classical geostatistical estimator. It fits a variogram model to
describe how spatial correlation decays with distance, then uses that model to compute
optimal weighted averages at prediction points. It is fast to set up, interpretable, and
well-understood, but its performance depends on how well the chosen variogram model fits
the data.

**GP (Gaussian Process)** is the machine-learning equivalent. It uses a kernel function
(Matérn or RBF) instead of a variogram, and maximises the log marginal likelihood to find
the best kernel parameters. It tends to give better uncertainty estimates (true predictive
distributions) and can represent a wider variety of spatial structures, but is slower and
less intuitive to inspect.

---

### 5.1 Kriging parameters

```yaml
engine:
  kriging:
    max_anisotropy: 10.0
    n_splits: 5
    n_trials: 10
```

#### `n_splits` — the most important Kriging parameter

**In plain language:**

Imagine you have 100 sample points. The engine needs to find the best variogram model
(shape, range, nugget, anisotropy direction) from the data. To judge how good a given
set of parameters is, it uses *spatial cross-validation*: it splits the data into groups,
hides one group at a time, predicts those hidden points using the rest, and measures the
prediction error. The parameter set that gives the lowest average error wins.

`n_splits` controls **how many groups** the data is divided into for this cross-validation.

**How the splitting works** (from `kriging.py:118-120`):

The engine uses **KMeans spatial clustering**, not random sampling. This is important:
random splits can accidentally put nearby points in both training and test sets, making
the model look better than it really is (it just memorises its neighbours). Spatial
clustering forces the test points to be geographically separated from training points,
giving a more honest estimate of performance.

With `n_splits = 5`, the study area is divided into 5 spatial clusters. In each of the
5 cross-validation rounds, one cluster becomes the test set and the other 4 are used for
training.

**How to choose `n_splits`:**

| `n_splits` | Effect |
|-----------|--------|
| 3 | Coarse evaluation. Each test set is large (1/3 of the data). Fast but noisier estimate of model quality. |
| 5 | Good default. Balances bias and variance in the CV estimate. |
| 10 | Fine evaluation. Each test set is small. Slower per Optuna trial but more stable CV estimate. Use when you have many points (> 200). |

**Warning:** If `n_splits` is larger than roughly `n_samples / 5`, some folds will have
fewer than 5 training points, which makes Kriging unstable. The code guards against this
(`kriging.py:150`).

---

#### `n_trials`

**What is Optuna doing?**

The engine does not just try one variogram model — it searches over a large space of
possible parameters:
- 9 candidate variogram model shapes (spherical, exponential, Gaussian, Matérn, stable,
  circular, rational-quadratic, linear, power)
- Partial sill (`psill`), range, nugget, anisotropy angle, and anisotropy ratio

`n_trials` is the **number of candidate parameter sets** Optuna evaluates. Each trial
runs the full spatial cross-validation described above.

**In plain language:** Think of `n_trials` as the number of different variogram recipes
the engine tries before deciding which one is best.

| `n_trials` | Effect |
|-----------|--------|
| < 100 | Too few. Optuna's TPE sampler needs ~25 warm-up (random) trials before it starts making informed suggestions. With 9 model types and 6+ parameters, fewer than 100 trials gives unreliable results. The engine will print a warning. |
| 100–300 | Solid range for most datasets. |
| 300–500 | High quality. Use when accuracy matters more than speed. |
| > 500 | Diminishing returns in most cases. |

The default in `config.yaml` is `n_trials: 10`, which is intentionally low for quick
testing. **Raise it to at least 100 for production runs.**

---

#### `max_anisotropy`

The anisotropy ratio is the ratio of the spatial range in the direction of maximum
continuity to the range in the perpendicular direction. A ratio of 1.0 means the field is
isotropic (same correlation in all directions). A ratio of 5.0 means the field is 5×
more continuous in one direction than another (common in geology: think bedding layers,
river channels, or wind-driven deposition).

`max_anisotropy` caps the upper bound of this ratio during the Optuna search. This
prevents the optimizer from finding physically implausible solutions (e.g. ratio = 50).

**How to choose:** Start with your domain knowledge. If you expect strong directional
continuity (e.g. a fluvial deposit), allow a higher value (10–15). For roughly isotropic
fields, a lower cap (2–3) keeps the search space smaller and converges faster.

---

### 5.2 GP (Gaussian Process) parameters

```yaml
engine:
  gp:
    angle_min: 0.0
    angle_max: 180.0
    max_anisotropy: 15.0
    n_optuna_trials: 300
    random_state: 42
```

#### `n_optuna_trials`

Same concept as `n_trials` in Kriging — the number of parameter combinations Optuna
evaluates. The GP search space is different: instead of variogram model shapes, Optuna
searches over kernel type (Matérn-3/2, Matérn-5/2, RBF), anisotropy angle, major/minor
length scales, signal variance, nugget variance, and numerical jitter (alpha).

The GP optimizer uses a **two-stage strategy** (from `gp.py`):
1. **Coarse angle scan** (free): evaluates 18 angles at 10° intervals with a fixed
   isotropic kernel. Identifies which angular band contains the anisotropy axis.
2. **Optuna TPE** (`n_optuna_trials` trials): refines within a ±20° window around the
   best coarse angle. Much more efficient than searching all 180°.
3. **L-BFGS-B local refinement** (free): gradient-based polish of the Optuna winner.

300 trials is a good default. Reduce to 100 for quick runs, increase to 500+ for
high-stakes predictions.

---

#### `angle_min` / `angle_max`

The search range (in degrees) for the anisotropy rotation angle. The rotation angle
defines the direction of maximum spatial continuity (the "major axis" of the anisotropy
ellipse).

- 0° = East (aligned with the X axis).
- 90° = North (aligned with the Y axis).
- The full range `[0, 180]` means the optimizer is free to find any direction.

Narrow this range if you have prior knowledge of the dominant structural direction
(e.g. bedding strike at ~30°: set `angle_min: 10`, `angle_max: 50`).

---

#### `max_anisotropy`

Same concept as in Kriging. Caps the ratio `length_scale_major / length_scale_minor`.
15.0 is a generous upper bound that accommodates most geological scenarios. Reduce to 3–5
for roughly isotropic phenomena.

---

#### `random_state`

Seeds the random number generators in Optuna and KMeans so that results are reproducible.
Set to `null` for a different result each run. Set to any integer for reproducibility.

---

## 6. `output`

```yaml
output:
  base_directory: output_2
  netcdf_z_dim_name: elev
  save_diagnostics: true
  formats:
    - nc
    - tif
    - csv
```

| Parameter | What it does |
|-----------|-------------|
| `base_directory` | Root folder for all outputs. The engine creates a subdirectory named after the input file stem (e.g. `output_2/input_dem/`). |
| `netcdf_z_dim_name` | Label for the vertical dimension in the NetCDF file, used by Paraview and other GIS tools. Common values: `Depth`, `elev`, `z`. |
| `save_diagnostics` | When `true`, generates all diagnostic plots: variograms, anisotropy ellipse, CV dashboard, trend components. Set to `false` to skip plots and only export data files. |
| `formats` | List of file formats to export. All listed formats are produced in a single run. |

### Export format reference

**Grid Mode** (when `prediction_points` is not set):

| Format | Files produced | Best for |
|--------|---------------|----------|
| `nc` | `predicted_{engine}.nc` | Scientific analysis, Paraview, xarray workflows |
| `tif` | `predicted_{engine}_mean.tif`, `predicted_{engine}_std.tif` | QGIS, ArcGIS, raster analysis |
| `csv` | `predicted_{engine}.csv` | Spreadsheets, pandas, any tabular tool |

**Point Mode** (when `prediction_points` is set):

| Format | Files produced | Best for |
|--------|---------------|----------|
| `csv` | `predicted_points_{engine}.csv` | Spreadsheets, sharing results |
| `xz` | `predicted_points_{engine}.xz` | Fast Python reload (preserves full GeoDataFrame with geometry) |

**Defaults if `formats` key is omitted:**
- Grid mode: `[nc]`
- Point mode: `[csv, xz]`

> **Note:** GeoTIFF export (`tif`) requires `rasterio`. Install it with:
> `conda install -c conda-forge rasterio`
> If `rasterio` is not installed and `tif` is requested, the engine will print a clear
> error message and exit.

---

## Quick-reference: parameter choices by scenario

| Scenario | Recommended settings |
|----------|----------------------|
| Quick exploratory run | `n_trials: 50`, `n_optuna_trials: 100`, `save_diagnostics: false` |
| Production accuracy run | `n_trials: 300`, `n_optuna_trials: 500`, `random_state: 42` |
| Strongly skewed data | `nst.enabled: true` |
| Known strong trend (e.g. DEM) | `detrend.auto_detect: false`, `detrend.enabled: true`, `order: 1` |
| Known anisotropy direction ~45° | `angle_min: 25`, `angle_max: 65` |
| Isotropic field | `max_anisotropy: 2.0` |
| Exporting for QGIS/ArcGIS | `formats: [tif]` |
| Exporting for Paraview | `formats: [nc]` |
| Everything at once | `formats: [nc, tif, csv]` |
