# Spatial Interpolation Engine — Modification Log v1

**Date:** April 24, 2026  
**Author:** Claude (Cowork AI Assistant)  
**Project:** `D:\1000_SCRIPTS\004_Project003\20260423_Interp_Engine`  
**Status:** All changes implemented and verified

---

## Overview

This document records every code change made to the Spatial Interpolation Engine
during the v1 development and improvement session. Changes are grouped into three
phases that reflect the order they were implemented.

---

## Phase 1 — Bug Fixes and Robustness (5 fixes)

These fixes addressed correctness issues identified during initial code review.

### Fix 1 — `src/preprocessor.py`: Trend detection tests the correct polynomial order

**Problem:** `analyze_trend()` always built an order-1 (linear) OLS model for the
F-test, regardless of the `order` parameter in `config.yaml`. This meant the
detection step and the correction step tested different polynomial families, so
the automatic detrend decision was unreliable for quadratic or cubic trends.

**Fix:** Added an `order: int = 1` parameter to `analyze_trend`. The function now
builds `PolynomialFeatures(degree=order)` before fitting OLS, so the F-test
matches the configured detrend order exactly. Also fixed the KNN weights
construction to use `k=min(5, len(Z)-1)` so it does not crash on small datasets.

**Key code change:**
```python
def analyze_trend(X, Y, Z, order: int = 1):
    poly = PolynomialFeatures(degree=order, include_bias=False)
    X_poly = poly.fit_transform(coords)
    X_model = sm.add_constant(X_poly)
    ols_results = sm.OLS(Z, X_model).fit()
    ...
    w = KNN.from_array(coords, k=min(5, len(Z) - 1))
```

---

### Fix 2 — `src/engines/kriging.py`: Increase default `n_trials` and add guard

**Problem:** The default `n_trials=50` was too low for Optuna to reliably find
good variogram hyperparameters. The Tree-structured Parzen Estimator (TPE)
sampler needs roughly 100+ trials to explore the search space adequately.

**Fix:** Changed the default from 50 → 150. Added a class constant
`_MIN_TRIALS_RECOMMENDED = 100` and a `UserWarning` raised at runtime if the
user sets `n_trials < 100` in config.

---

### Fix 3 — `src/engines/kriging.py`: Fix `best_params_` mutation from `.pop()`

**Problem:** The original code did `self.best_params_ = study.best_params` then
`self.best_params_.pop('model')`. This mutated the dict in-place, meaning the
`'model'` key was destroyed on the Optuna study object and could not be inspected
later. It also broke any subsequent code that iterated `best_params_`.

**Fix:** Replaced the pop with a copy-and-filter pattern:
```python
_raw_best = self.study_.best_params          # untouched original
self.best_model_name_ = _raw_best['model']
self.best_params_ = {k: v for k, v in _raw_best.items() if k != 'model'}
```

---

### Fix 4 — `src/engines/gp.py`: Raise alpha floor and add nugget guard

**Problem:** The alpha (jitter) lower bound was `1e-14`, which is effectively
zero. Near-singular covariance matrices caused the GP to produce a constant-mean
predictor (R² ≈ 0) on noisy datasets like S4 and S8. There was also no explicit
nugget term, so micro-scale variability was absorbed incorrectly into the
length-scale parameter.

**Fix:** Alpha lower bound raised to `1e-6`. A `WhiteKernel` was added as an
explicit nugget component (see Phase 2 for the full kernel redesign).

---

### Fix 5 — `main.py`: Wire `trend_order` into `analyze_trend` call

**Problem:** `main.py` called `analyze_trend(X, Y, Z)` without passing the
`order` argument, so Fix 1 in `preprocessor.py` had no effect at runtime.

**Fix:** The call now reads the order from config and passes it:
```python
trend_order = preproc_cfg.get("detrend", {}).get("order", 1)
trend_info = analyze_trend(X, Y, Z_raw, order=trend_order)
```
The subsequent print statement also now shows `tested_order` so users can confirm
the correct polynomial degree was tested.

---

## Phase 2 — GP Engine Architectural Redesign

This phase was motivated by GP systematically underperforming Kriging on SGS
(Sequential Gaussian Simulation) scenarios (S3, S4, S6, S7, S8).

### `src/engines/gp.py` — Full rewrite of `RotatedGPR`

**Root causes addressed:**

1. **Wrong smoothness assumption.** The original engine used a fixed RBF kernel
   (infinite differentiability). SGS fields are rough by construction (Matérn ν ≈
   1.5 or 2.5). Fitting an overly-smooth kernel to a rough field causes the
   optimizer to collapse the length scale to tiny values as a workaround.

2. **Length scale bounds too rigid.** A hard-coded floor of 10 m caused bound
   collapse when the true range was shorter. The optimizer could not escape.

3. **No explicit nugget.** Without a `WhiteKernel`, the GP could not separate
   structured spatial variance from measurement noise, leading to a constant-mean
   predictor on high-nugget datasets.

4. **Log-space optimization not used.** Searching length scale and variance in
   linear space means coarse coverage at small scales and fine coverage at large
   scales — the inverse of what is needed.

**Changes implemented:**

#### Adaptive kernel catalogue
```python
_KERNEL_CATALOGUE = {
    "matern_32": (Matern, {"nu": 1.5}),   # rough fields (SGS)
    "matern_52": (Matern, {"nu": 2.5}),   # moderately smooth
    "rbf":       (RBF,   {}),             # very smooth (deterministic trends)
}
```
`kernel_type` is an Optuna categorical parameter. The engine selects the kernel
family with the highest Log-Marginal Likelihood (LML).

#### Composite kernel structure
```
ConstantKernel(signal_variance) × build_base_kernel(type) + WhiteKernel(nugget)
```
The `WhiteKernel` provides a dedicated parameter for measurement noise and
micro-scale variability, eliminating the "constant mean" failure mode.

#### Log-space hyperparameter search
All parameters (`log_var`, `log_lmaj`, `ratio_log`, `log_nugget`, `log_alpha`)
are sampled in natural log space by Optuna, then exponentiated before use. This
ensures equal coverage across multiple orders of magnitude.

#### Adaptive length scale bounds
Computed from data geometry at runtime in `main.py`:
```python
_median_nn = median(nearest-neighbour distances)   # lower bound proxy
_max_dist  = max pairwise distance                 # upper bound proxy
ls_min = max(_median_nn * 0.5, 1e-3)
ls_max = _max_dist * 0.6
```
These bounds prevent the optimizer from collapsing to degenerate scales.

#### Updated `get_kernel_params()` return keys
| Key | Meaning |
|---|---|
| `kernel_type` | Selected kernel family (`matern_32` / `matern_52` / `rbf`) |
| `rotation_angle_deg` | Best anisotropy direction |
| `constant_value` | Signal variance (from `ConstantKernel`) |
| `length_scale` | [major, minor] anisotropic length scales |
| `nugget_variance` | Noise variance (from `WhiteKernel`) |
| `jitter_alpha` | Numerical jitter added to diagonal |
| `anisotropy_ratio` | major / minor length scale ratio |
| `log_marginal_likelihood` | LML of the best fit |

---

### `config.yaml` — GP section rewritten

Old GP parameters (`angle_search`, `n_restarts`, `noise_level_*`,
`length_scale_min/max`, `mode: anisotropic`) were removed. New parameters:

```yaml
engine:
  gp:
    n_optuna_trials: 300
    random_state: 42
    angle_min: 0.0
    angle_max: 180.0
    max_anisotropy: 15.0
  kriging:
    n_trials: 150
    max_anisotropy: 15.0
    n_splits: 5
```

---

### `utils.py` — Fix stale alpha key in `perform_gpr_kfold_cv`

After the gp.py rewrite, `RotatedGPR` no longer exposes a `.alpha` attribute.
The CV function was updated to read from the new key:
```python
alpha = params.get("jitter_alpha", getattr(rgpr_model, "best_alpha_", 1e-6))
```

---

## Phase 3 — Cross-Validation Framework Improvements (Council Recommendations)

These changes address why both engines showed R² ≈ 0.01 on `S8_SGS_HighNugget`.
A multi-agent council debate identified two structural problems in the CV
framework and one missing diagnostic.

### `utils.py` — New helper: `make_spatial_block_folds()`

**Problem (Council diagnosis):** KMeans spatial CV creates compact, isolated
clusters. Test cluster centres are 400–500 m from the nearest training cluster,
which exceeds the fitted correlation range (~300 m in S8). The CV is therefore
measuring *extrapolation* skill, not interpolation skill. This systematically
underestimates model quality and gives a misleading picture of real-world
performance.

**Fix:** A new helper function replaces KMeans in both CV functions:

```python
def make_spatial_block_folds(X: np.ndarray, n_folds: int) -> np.ndarray:
    # Projects onto 45° diagonal (X + Y), sorts, divides into N strips.
    diagonal_proj = X[:, 0] + X[:, 1]
    sort_idx = np.argsort(diagonal_proj)
    fold_ids = np.empty(len(X), dtype=int)
    for fold, chunk in enumerate(np.array_split(sort_idx, n_folds)):
        fold_ids[chunk] = fold
    return fold_ids
```

Each strip is a contiguous geographic band surrounded by training data on both
sides, ensuring test points stay within the correlation range (true interpolation).

**Scientific reference:** Wadoux et al. (2021), *Environmental Modelling &
Software* — spatial block CV produces more reliable estimates of interpolation
performance than random or cluster-based CV.

---

### `utils.py` — `perform_kriging_kfold_cv()`: Per-fold nugget re-estimation

**Problem:** The global nugget estimate (fitted on all 500 points) is used for
every CV fold. When a geographic band is removed, the local noise level in that
region may differ from the global average. Using the wrong nugget inflates
kriging variance and degrades predictions in that fold.

**Fix:** Before each fold, the function now attempts to fit a lightweight
`skgstat.Variogram` on the training split alone and extracts a fold-local nugget
and sill. If this fails (library unavailable, too few points, numerical error),
it silently falls back to the global parameters:

```python
try:
    from skgstat import Variogram as _SVario
    _sv = _SVario(X_tr, y_tr, model=model_name, n_lags=10,
                  maxlag=global_bp.get("range", None),
                  fit_method="trf", estimator="matheron")
    _params = _sv.parameters  # [range, sill, nugget]
    if _params is not None and len(_params) >= 3:
        fold_bp["nugget"] = float(max(_params[2], 0.0))
        fold_bp["psill"]  = float(max(_params[1], 1e-6))
except Exception:
    pass  # fall back to global_bp
```

---

### `main.py` — Theoretical R² ceiling diagnostic

**Problem:** When both engines report R² ≈ 0.01, it is impossible to tell from
the output alone whether this is a model failure or a physical data limitation.
For high-nugget datasets, most of the variance is pure noise — no spatial model
can predict it.

**Fix:** After computing CV metrics, `main.py` now reads the fitted nugget and
signal variance from the model parameters and prints the theoretical R² ceiling:

```
R²_ceiling = 1 − C₀ / (C₀ + C)
```

where C₀ is the nugget variance and C is the structured sill. Example output:

```
── Cross-Validation Metrics ──
MAE        : 0.7234
RMSE       : 0.9069
R²         : 0.0148
R² ceiling : 0.0871  (nugget accounts for 91.3% of total variance)
↑ Gap to ceiling: 0.0723 — moderate room for improvement.
```

If R² is near the ceiling, the engine is performing as well as the data
physically allows. If the gap is large, the issue is the model or data density.

Parameter key mapping used:
- **GP:** `nugget_variance` (WhiteKernel) and `constant_value` (ConstantKernel)
- **Kriging:** `nugget` and `psill` from `best_params_`

---

## Summary Table

| # | File | Change | Phase |
|---|---|---|---|
| 1 | `src/preprocessor.py` | `analyze_trend` tests correct polynomial order | 1 |
| 2 | `src/engines/kriging.py` | Raise default `n_trials` 50→150, add warning | 1 |
| 3 | `src/engines/kriging.py` | Fix `best_params_` mutation from `.pop()` | 1 |
| 4 | `src/engines/gp.py` | Raise alpha floor to `1e-6` | 1 |
| 5 | `main.py` | Pass `trend_order` to `analyze_trend` | 1 |
| 6 | `src/engines/gp.py` | Full rewrite: adaptive kernel catalogue (Matérn-3/2, Matérn-5/2, RBF) | 2 |
| 7 | `src/engines/gp.py` | Add `WhiteKernel` for explicit nugget modeling | 2 |
| 8 | `src/engines/gp.py` | Log-space hyperparameter search via Optuna | 2 |
| 9 | `main.py` | Compute adaptive length scale bounds from data geometry | 2 |
| 10 | `config.yaml` | Rewrite GP section for new architecture | 2 |
| 11 | `utils.py` | Fix stale `noise_level` → `jitter_alpha` key in GPR CV | 2 |
| 12 | `utils.py` | Add `make_spatial_block_folds()` helper | 3 |
| 13 | `utils.py` | Replace KMeans with spatial block folds in both CV functions | 3 |
| 14 | `utils.py` | Add per-fold nugget re-estimation in kriging CV | 3 |
| 15 | `main.py` | Print theoretical R² ceiling alongside CV metrics | 3 |
| 16 | `src/engines/gp.py` | Add `_coarse_angle_scan()`: 10°-grid NLL scan before Optuna | 4 |
| 17 | `src/engines/gp.py` | Restrict Optuna angle search to ±20° window from coarse best | 4 |
| 18 | `src/preprocessor.py` | Add `NormalScoreTransform` class (forward + inverse + tail extrapolation) | 4 |
| 19 | `src/preprocessor.py` | Add `check_normality()` (Shapiro-Wilk, skewness, excess kurtosis) | 4 |
| 20 | `src/preprocessor.py` | Embed normality report into `analyze_trend()` return dict | 4 |
| 21 | `main.py` | Add `check_and_clean_duplicates()` helper function | 4 |
| 22 | `main.py` | Call duplicate guard (step 1.1) immediately after data loading | 4 |
| 23 | `main.py` | Wire NST into pipeline: apply after detrend, back-transform after predict | 4 |
| 24 | `main.py` | Print Shapiro-Wilk, skewness, kurtosis in preprocessing report | 4 |
| 25 | `config.yaml` | Add `preprocessing.nst` and `preprocessing.duplicates` config sections | 4 |

---

## Phase 4 — Robustness Enhancements (3 changes)

### Enhancement A — `src/engines/gp.py`: Two-stage angle search

**Problem (S5 scenario):** For extreme anisotropy (range ratio ~15×), the
Log-Marginal Likelihood (LML) landscape has a very narrow valley in the angle
dimension — roughly ±5° wide out of a 180° search space. Optuna's TPE sampler
explores angles uniformly at first and has a low probability of landing inside
this valley in the first 100 trials. The result is that Optuna never learns the
correct anisotropy direction, and the engine fits an approximately isotropic
kernel with a poor R² despite the structured signal being clearly present
(R² ceiling ~0.92).

**Fix — two steps added before Optuna:**

**Step 0 (new):** `_coarse_angle_scan()` evaluates the LML at 19 candidate
angles (0°, 10°, 20°, …, 180°) using a fixed isotropic Matérn-5/2 kernel with
mid-range log-space parameters. This costs only 19 LML evaluations (< 1 second)
but reliably identifies which 10°-wide band contains the true anisotropy axis.

**Step 1 (modified):** Optuna now searches angle only in a ±20° window around
the coarse winner instead of the full [0°, 180°] range. This is ~4.5× smaller,
so each of the 300 Optuna trials is more likely to refine the correct angle
rather than exploring irrelevant directions.

```python
coarse_best_angle, _ = self._coarse_angle_scan(step_deg=10.0, ...)
angle_min_fine = max(angle_bounds[0], coarse_best_angle - 20.0)
angle_max_fine = min(angle_bounds[1], coarse_best_angle + 20.0)
# Optuna then uses: trial.suggest_float("angle", angle_min_fine, angle_max_fine)
```

Expected impact: +0.15–0.30 improvement in R² for strongly anisotropic datasets
(S2, S5). Negligible impact on isotropic datasets (the coarse scan takes ~1 s
and the correct angle valley is wide for isotropic fields).

---

### Enhancement B — `src/preprocessor.py` + `main.py`: Normal-Score Transform

**Problem:** Both Kriging and GP assume the spatial field (or its detrended
residuals) follows a Gaussian distribution. Real geoscientific data commonly
violates this: permeability is log-normal, geochemical concentrations have
heavy tails, ore grade is positively skewed. Feeding skewed data inflates the
variogram sill, distorts kriging weights, and makes uncertainty intervals wrong.

**Two new components in `preprocessor.py`:**

`check_normality(Z, alpha=0.05)` — runs the Shapiro-Wilk test (most powerful
for n < 5000) and computes skewness and excess kurtosis. Returns
`recommend_nst=True` when the data is non-normal AND either `|skewness| > 0.5`
or `|excess kurtosis| > 1.0`. The result is embedded into `analyze_trend()`'s
return dict under the key `"normality"`.

`NormalScoreTransform` — a bijective rank-preserving mapping (Journel &
Huijbregts, 1978) that maps any continuous distribution to N(0,1):
- Forward: sort observations → assign Hazen quantiles → map through Φ⁻¹
- Inverse: map through Φ → interpolate the empirical quantile function
- Tail extrapolation: values outside the training range are handled via the
  normal distribution CDF, preventing hard clamps at observed extremes.

**Wiring in `main.py`:**
- NST is applied after detrending and before engine fitting (to the residuals).
- Auto-detection reads `normality.recommend_nst` from `analyze_trend()`.
- Users can override with `preprocessing.nst.enabled: true/false` in config.
- Predictions are back-transformed via `nst.inverse_transform(means)`.
- Prediction uncertainty (std) is propagated through the nonlinear transform
  using a finite-difference derivative of the inverse NST at each grid point.

---

### Enhancement C — `main.py`: Duplicate/Near-Duplicate Coordinate Guard

**Problem:** Two samples at the same (or nearly the same) location with
different values create a contradiction: the spatial covariance between them
should be near the sill (distance ≈ 0), but their values differ. This makes
the kriging matrix singular (crash) or forces GP to absorb the contradiction
via the nugget term (silent degradation).

Common causes in real data: GPS coordinate snapping to a grid, survey rounding
to the nearest metre, digitisation of paper maps at limited precision.

**New function `check_and_clean_duplicates(X, Z, min_separation)`:**

| Case | Distance | Action |
|---|---|---|
| Exact duplicate | = 0 | Replace group with one point; value = arithmetic mean of group |
| Near-duplicate | 0 < d < `min_separation` | Add independent random jitter ≤ `min_separation / 10` |
| Clean point | ≥ `min_separation` | No change |

`min_separation` defaults to `0.1 × median nearest-neighbour distance` if not
set in config, so it automatically adapts to the data's point spacing.

The function runs immediately after data loading (step 1.1) and before all
other preprocessing, so all downstream steps always receive clean coordinates.
A clear warning is printed if any points were modified.

---

## Performance Impact (Synthetic Benchmarks S1–S8)

The table below compares GP performance before and after the Phase 2 redesign.
Kriging results are unchanged (it was already using Optuna with 9 variogram models).

| Scenario | GP R² (before) | GP R² (after) | Change |
|---|---|---|---|
| S1_Isotropic | ~0.35 | 0.6027 | +0.25 |
| S2_Aniso_45deg | ~0.10 | 0.2331 | +0.13 |
| S3_Sparse_120deg | ~0.02 | 0.2430 | +0.22 |
| S4_HighNugget | <0.0 (negative) | 0.3415 | breakthrough |
| S5_ExtremeAniso | ~0.08 | 0.1858 | +0.11 |
| S6_Nested | ~0.0 | 0.0065 | limited (nested structure) |
| S7_Clustered | ~0.05 | 0.1451 | +0.10 |
| S8_HighNugget | ~0.0 | 0.0050 | limited (physical ceiling ~0.09) |

S6 and S8 remain near zero R² because the physical data structure limits the
theoretical maximum (nested variograms in S6; nugget ratio >90% in S8). The R²
ceiling diagnostic added in Phase 3 makes this transparent to the user.

---

*End of modification log v1*
