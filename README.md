# Spatial Interpolation Engine

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

A production-hardened 2D spatial interpolation engine with dual backends — ordinary
kriging and Gaussian process regression — driven by automatic hyperparameter
optimization. Handles messy real data out of the box: non-Gaussian distributions,
spatial trends, duplicate coordinates, and anisotropic correlation structures.

---

## Quickstart

```bash
pip install -r requirements.txt
```

Set your input file and target variable in `config.yaml`:

```yaml
input:
  filepath: my_data.csv
  columns:
    x: easting
    y: northing
    value: elevation
engine:
  mode: kriging   # or "gp"
```

Run:

```bash
python main.py config.yaml           # normal output
python main.py config.yaml --verbose # show Optuna trials and post-fit details
python main.py config.yaml --quiet   # warnings and errors only
python main.py --help                # see all options
```

Results — predictions, cross-validation metrics, and diagnostic plots — land in
`output/<dataset_name>/`.

---

## Features

- **Two interpolation backends** — ordinary kriging (9 variogram models via PyKrige)
  or Gaussian process (Matérn-3/2, Matérn-5/2, RBF kernels via scikit-learn)
- **Automatic hyperparameter tuning** — Optuna TPE sampler with spatial KMeans
  cross-validation, not random splits (prevents information leakage); supports
  parallel trials via `n_jobs` for multi-core speedup
- **Structured logging** — Python `logging` module with `--verbose` / `--quiet` CLI
  flags; progress bars via `tqdm` for Optuna trials, angle scans, and CV folds
- **Normal-Score Transform** — rank-preserving Gaussian anamorphosis with
  auto-detection (Shapiro-Wilk + skewness + kurtosis thresholds)
- **Polynomial trend detection** — F-test and Moran's I decide when to detrend;
  subtracts the trend before modelling and re-adds it after prediction
- **Input validation** — five pre-flight checks catch bad data (NaN, constant
  values, colinear points, too few samples) before Optuna ever fires
- **Duplicate handling** — exact duplicates averaged, near-duplicates jittered
  with adaptive separation thresholds
- **Anisotropy search** — two-pass strategy: coarse angle scan followed by Optuna
  refinement and L-BFGS-B local polish
- **Fast-path API** — `fit_with_known_params()` on both engines skips Optuna when
  you already have good parameters (CI, repeated runs, parameter sharing)
- **Smart prediction grid** — ConvexHull-bounded grid generation with configurable
  buffer; also supports prediction at arbitrary point locations
- **Multi-format export** — NetCDF (CF-compliant), GeoTIFF, CSV, and compressed
  pickle for point-mode predictions

---

## Architecture

```
config.yaml ──► data_loader ──► preprocessor ──► engine ──► exporter ──► diagnostics
                     │                │            │
                   CSV/Excel/    detrend +    kriging or    NetCDF/    variograms +
                   Shapefile     NST + dup.    GP with      GeoTIFF/   anisotropy +
                                 handling      Optuna CV    CSV        CV dashboard
```

| Module | Role |
|--------|------|
| [`main.py`](main.py) | Pipeline orchestrator with 7 decomposed stages: load → preprocess → geometry → engine → fit → predict → export |
| [`src/engines/kriging.py`](src/engines/kriging.py) | Anisotropic ordinary kriging with Optuna variogram selection |
| [`src/engines/gp.py`](src/engines/gp.py) | Anisotropic Gaussian Process with kernel selection |
| [`src/preprocessor.py`](src/preprocessor.py) | TrendProcessor, NormalScoreTransform, normality checks |
| [`src/validation.py`](src/validation.py) | Input validation (finite, min samples, colinearity, constant values) |
| [`src/geometry.py`](src/geometry.py) | ConvexHull grid generation |
| [`src/data_loader.py`](src/data_loader.py) | CSV / Excel / Shapefile loading with CRS-aware column detection |
| [`src/exporter.py`](src/exporter.py) | NetCDF, GeoTIFF, and CSV export |
| [`utils.py`](utils.py) | Diagnostic plots: variograms, anisotropy ellipse, CV dashboard, trend components |

---

## Quick comparison: kriging vs GP

| Scenario | Description | Kriging R² | GP R² |
|----------|-------------|-----------|-------|
| S1 | Isotropic field | 0.82 | 0.39 |
| S2 | Anisotropic, 45° | 0.68 | 0.72 |
| S3 | Sparse, 120° aniso | 0.23 | 0.14 |
| S4 | High nugget (noisy) | 0.42 | 0.49 |
| S7 | Clustered sampling | 0.28 | 0.51 |

Kriging generally recovers isotropic and well-structured fields more accurately;
GP handles noisy and clustered data better. The 14-benchmark suite in
[`test_data/`](test_data/) covers these scenarios plus extreme cases (colinear
points, log-normal distributions, strong trends, few samples, extreme anisotropy).

---

## Configuration reference

Every `config.yaml` parameter is documented with plain-language explanations and
practical guidance in **[`package_guidance.md`](package_guidance.md)** — variogram
models, anisotropy caps, NST auto-detection rules, export formats, and scenario
recommendations.

---

## Tests

```bash
python test_engine.py
```

26 tests, 97 assertions covering both kriging and GP backends. Parameter caching
avoids re-running Optuna on every invocation — ~50 seconds with cache populated,
~15 minutes without. 14 synthetic datasets cover isotropic through
extreme-anisotropy regimes plus edge cases (duplicates, colinearity, log-normal,
strong trend, few points).

---

## Citation

If you use this engine in published research, please cite the repository:

```bibtex
@software{interp_engine,
  author = {david-ncu2019},
  title = {Spatial Interpolation Engine},
  url = {https://github.com/david-ncu2019/interp_engine},
  year = {2025}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
