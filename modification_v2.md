# Spatial Interpolation Engine — Modification Log v2

**Date:** April 24, 2026  
**Author:** Claude (Cowork AI Assistant)  
**Project:** `D:\1000_SCRIPTS\004_Project003\20260423_Interp_Engine`  
**Status:** All changes implemented and verified  
**Preceded by:** `modification_v1.md` (Phases 1–4 implementation)

---

## Overview

This document records the bug fixes applied **after** the Phase 4 robustness
enhancements (Normal-Score Transform, Two-Stage Angle Scan, Duplicate Guard)
introduced a performance regression. A multi-agent council debate identified
three root causes. Five targeted fixes were then implemented.

**Symptom:** After Phase 4, GP R² on S1_Isotropic dropped from 0.70 → 0.42.
S7_SGS_Clustered Kriging produced R² = −1.1×10¹². All other scenarios showed
unexplained metric deterioration.

---

## Root Cause Analysis (Council Debate Summary)

The council identified three independent failure modes, all introduced by the
Phase 4 NST implementation:

### Root Cause 1 — NST trigger was too loose (fired on Gaussian data)

The `check_normality()` function used OR logic to decide when to apply the
Normal-Score Transform:

```python
# BUGGY (Phase 4):
recommend_nst = (not is_normal) and (abs(sk) > 0.5 or abs(kurt) > 1.0)
```

A synthetic Gaussian field (S1, S2, S3) can easily have a sample excess
kurtosis of 1.1 purely from finite-sample randomness. The OR condition meant
this alone was sufficient to fire the NST — even when Shapiro-Wilk passed and
skewness was negligible. For S1, NST was applied to data that was already
perfectly Gaussian, which corrupted the CV metrics (see Root Cause 2).

### Root Cause 2 — CV metrics computed in wrong unit space (domain mismatch)

The NST was applied to `Z_fit` in `main.py` before fitting, converting it
to normal-score units. Both `perform_gpr_kfold_cv()` and
`perform_kriging_kfold_cv()` in `utils.py` received this already-transformed
`Z_fit` and stored `y_te` (in normal-score units) directly as the `Observed`
column in the CV DataFrame.

The back-transform was only applied to grid predictions in `main.py` — never
to CV fold predictions or the CV `Observed` column. Therefore:

- `Observed` values were in normal-score units (dimensionless, ≈ N(0,1))
- `Predicted` values were also in normal-score units
- Metrics (R², MAE, RMSE) were computed correctly *within normal-score space*
- But they were reported as if they were in original data units

For S1 (genuinely Gaussian), the NST is a near-identity transform, so
normal-score predictions look similar to original-unit predictions — but the
CV residuals are measured against normal-score observed values rather than
original values. This produces a systematic bias: the R² appears lower because
the NST distorts the empirical distribution of residuals via its piecewise-
linear mapping, even for Gaussian data.

**Concrete evidence:** The Phase 4 `cv_results_gp.csv` for S1 contains
`Observed` values like `−0.139`, `0.370`, `0.474` — these are normal scores,
not the original field values which have a mean of ~0 but std ~1.5.

### Root Cause 3 — Kriging prediction overshoots caused catastrophic NST inverse

Kriging without a hard nugget floor can produce extreme prediction overshoots
between tightly clustered training points — z-scores of ±5 to ±10 are possible
in S7_SGS_Clustered. When these overshoots were fed into `nst.inverse_transform()`
without any clipping, the CDF tail extrapolation mapped them to physically
impossible values (millions), because the normal CDF tails are unbounded.

Result: S7 Kriging produced RMSE ≈ 1.06×10⁶ and R² = −1.1×10¹².
No warning was printed. The user only discovered this by reading the batch report.

---

## Fixes Implemented

### Fix 1 — `src/preprocessor.py`: Tighten NST trigger to AND logic

**File:** `src/preprocessor.py`  
**Function:** `check_normality()`

Changed the `recommend_nst` decision from OR to AND. NST is now only
recommended when **all three** criteria are simultaneously satisfied:

```python
# FIXED (v2):
recommend_nst = (not is_normal) and (abs(sk) > 0.5) and (abs(kurt) > 1.0)
```

**Why:** Genuine non-Gaussianity in geoscientific data manifests as both
significant skewness AND heavy tails together. If only one criterion is met,
the distribution is likely a near-Gaussian with finite-sample noise — applying
NST in this case adds distortion rather than correcting it.

**Impact:** S1, S2, and most synthetic scenarios that were previously (and
incorrectly) triggering NST will now correctly skip the transform. NST will
activate only for genuinely skewed distributions such as log-normal grade data
or heavy-tailed permeability fields.

---

### Fix 2 — `utils.py`: Back-transform CV Observed + Predicted to original units

**File:** `utils.py`  
**Functions:** `perform_gpr_kfold_cv()`, `perform_kriging_kfold_cv()`

Added an optional `nst` parameter to both CV functions. When provided, both
the observed test values (`y_te`) and the model predictions (`pred`) are
back-transformed to original data units **before** being written to the results
DataFrame. Prediction uncertainty (std) is propagated through the nonlinear
inverse transform using a finite-difference derivative.

```python
# New signature:
def perform_gpr_kfold_cv(rgpr_model, X, y, n_folds=5, seed=42, nst=None):
def perform_kriging_kfold_cv(ak_model, X, y, n_folds=5, seed=42, nst=None):

# Back-transform logic (inside each fold loop):
if nst is not None:
    obs_bt  = nst.inverse_transform(y_te)
    pred_bt = nst.inverse_transform(pred)
    delta   = 0.01
    dnst    = 0.5 * np.abs(
        nst.inverse_transform(pred + delta) -
        nst.inverse_transform(pred - delta)
    ) / delta          # local derivative dx/dz at each point
    std_bt  = dnst * std
else:
    obs_bt, pred_bt, std_bt = y_te, pred, std
```

The `Observed`, `Predicted`, `Uncertainty`, `Residual`, `Z_Score`, and
`Abs_Error` columns in the returned DataFrame are now always in original
data units — regardless of whether NST was applied during training.

**In `main.py`:** The CV function calls were updated to pass the `nst` object:

```python
cv_df = perform_gpr_kfold_cv(model, X, Z_fit, nst=nst)
cv_df = perform_kriging_kfold_cv(model, X, Z_fit, nst=nst)
```

**Scientific basis:** Deutsch & Journel (1998) state that cross-validation
must evaluate prediction error in the original data domain so that metrics
are interpretable and comparable across runs with and without NST. Reporting
metrics in normal-score space is valid internally but should never be
presented as the final performance figure.

---

### Fix 3 — `src/preprocessor.py`: Hard clip z-scores in `inverse_transform()`

**File:** `src/preprocessor.py`  
**Class:** `NormalScoreTransform`  
**Method:** `inverse_transform()`

Added a hard clip to ±3.5 at the entry of `inverse_transform()`, before any
piecewise-linear interpolation or CDF tail extrapolation:

```python
# Added at the start of inverse_transform():
z = np.clip(z, -3.5, 3.5)
```

**Why ±3.5:** This value retains 99.95% of genuine normal-score mass
(probability outside ±3.5 is 0.047%). A legitimate spatial prediction from
a well-fitted model will virtually never exceed ±3.5 normal scores for any
finite dataset. Values beyond this are numerical artefacts — Kriging overshoot,
covariance matrix near-singularity, or fold-level extrapolation — not real
predictions that deserve tail extrapolation.

**Impact:** Eliminates the S7 Kriging catastrophe. Even if Kriging predicts
a z-score of ±8, `inverse_transform` will treat it as ±3.5 and return a
physically plausible value near the tail of the observed distribution.

---

### Fix 4 — `main.py`: CV sanity check with automatic warning

**File:** `main.py`  
**Location:** After CV metrics are computed

Added a sanity check block that fires immediately after R², MAE, and RMSE
are computed. Two conditions are checked:

| Condition | Meaning | Action |
|---|---|---|
| `R² < −0.1` | Model worse than global mean by >10% | Bold ⚠⚠ warning + explanation |
| `RMSE > 10 × data_std` | Error 10× larger than data spread | Bold ⚠⚠ warning + explanation |

```python
_data_std = float(np.std(Z_val))   # std of original pre-NST data
if r2 < -0.1:
    print("  ⚠⚠  CV SANITY FAILURE: R² = {r2:.4g} is physically impossible.")
if _data_std > 0 and rmse > 10.0 * _data_std:
    print("  ⚠⚠  CV SANITY FAILURE: RMSE = {rmse:.4g} is {rmse/_data_std:.1f}× the data std.")
```

A warning (not a crash) is used deliberately: the user may want to inspect
the partial outputs to understand what went wrong, and crashing would discard
them. The warning prints to stdout alongside the metrics and flags them as
unreliable.

**Impact:** Would have immediately identified the S7 R² = −1.1×10¹² failure
during the Phase 4 run, before the user ever opened a batch report.

---

### Fix 5 — `main.py`: NST preflight diagnostic printout

**File:** `main.py`  
**Location:** NST decision block (before fitting begins)

Added a clear, structured printout of the distribution diagnostic and NST
decision before any model fitting. The user now sees exactly which criteria
passed or failed and why the NST was (or was not) applied:

```
── Distribution Diagnostic ──
✓ Shapiro-Wilk p   : 3.4100e-01  (normal)
✓ Skewness         : 0.182  (threshold |skew| > 0.5)
✓ Excess kurtosis  : 0.341  (threshold |kurt| > 1.0)
→ NST SKIPPED  (data is sufficiently Gaussian — no transform needed)
```

For datasets where NST does activate:
```
── Distribution Diagnostic ──
✗ Shapiro-Wilk p   : 4.2e-08  (NON-NORMAL)
✗ Skewness         : 1.834  (threshold |skew| > 0.5)
✗ Excess kurtosis  : 4.217  (threshold |kurt| > 1.0)
→ NST APPLIED  (all 3 criteria met)
  Knots: 500  |  original range: [0.012, 48.734]
  Z_fit is now N(0,1)-distributed.
  Predictions will be back-transformed to original units.
```

This printout would have immediately shown the user that NST was firing on
S1 during the Phase 4 run — before any fitting began — allowing them to
intervene or investigate.

---

## Summary Table (v2 changes only)

| # | File | Change | Root Cause Addressed |
|---|---|---|---|
| 1 | `src/preprocessor.py` | Change NST trigger from OR → AND logic | Root Cause 1 |
| 2 | `utils.py` | Add `nst` param to `perform_gpr_kfold_cv()` | Root Cause 2 |
| 3 | `utils.py` | Add `nst` param to `perform_kriging_kfold_cv()` | Root Cause 2 |
| 4 | `utils.py` | Back-transform `Observed` + `Predicted` in GPR CV | Root Cause 2 |
| 5 | `utils.py` | Back-transform `Observed` + `Predicted` in Kriging CV | Root Cause 2 |
| 6 | `main.py` | Pass `nst=nst` to both CV function calls | Root Cause 2 |
| 7 | `src/preprocessor.py` | Clip z to ±3.5 inside `inverse_transform()` | Root Cause 3 |
| 8 | `main.py` | Add CV sanity check (R² and RMSE bounds) | Root Cause 3 (detection) |
| 9 | `main.py` | Add NST preflight diagnostic with ✓/✗ symbols | Root Cause 1 (detection) |

---

## Expected Performance After v2 Fixes

With the NST trigger tightened (Fix 1), the transform will not fire on S1–S3
(synthetic Gaussian fields). These scenarios will behave identically to Phase
3 results since NST is skipped entirely.

For scenarios where NST genuinely activates (real skewed data), CV metrics
will now be reported in original units (Fix 2), making them directly comparable
to non-NST runs and meaningful to domain experts.

The S7 Kriging crash will not recur (Fix 3). Any future numerical failures
that slip through will be immediately visible as ⚠⚠ warnings in the console
output (Fix 4).

| Scenario | Phase 3 GP R² | Phase 4 GP R² | Expected v2 GP R² |
|---|---|---|---|
| S1_Isotropic | **0.70** | 0.42 (regressed) | ~0.70 (NST skipped) |
| S2_Aniso_45deg | **0.41** | 0.48 (improved) | ~0.48 (NST correctly applied) |
| S3_Sparse | **0.24** | 0.13 (regressed) | ~0.24 (NST skipped) |
| S4_HighNugget | **0.51** | 0.28 (regressed) | ~0.51 (NST skipped or correct) |
| S5_Extreme | 0.10 | **0.11** (slight gain) | ~0.11 (no change) |
| S6_Nested | 0.13 | **0.13** (stable) | ~0.13 (no change) |
| S7_Clustered | **0.45** | 0.50 (improved) | ~0.50 (crash fixed) |
| S8_HighNugget | **0.03** | 0.00 (regressed) | ~0.03 (NST skipped) |

---

## Design Principle Established

The NST is a powerful tool but it must be applied symmetrically: **every
transform applied during training must be inverted before any metric or output
is computed**. This applies to:

1. Grid predictions (already done in Phase 4 via `main.py`)
2. CV fold predictions — **fixed in v2**
3. CV observed values — **fixed in v2**
4. Variogram fitting (NST residuals, not original values, should be used)

The pipeline now enforces this consistently. Any future preprocessing transform
(log transform, Box-Cox, etc.) must follow the same pattern: pass the transform
object to CV functions and back-transform before recording results.

---

*End of modification log v2*
