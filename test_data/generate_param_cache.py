"""One-shot: fit every scenario and cache best parameters to JSON."""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.engines.kriging import AnisotropicKriging

DATA_DIR = Path(__file__).parent
CACHE_DIR = DATA_DIR / "params_cache"

SCENARIOS = [
    # (name, max_anisotropy, n_trials, n_splits)
    ("S1_Isotropic",              3.0,  100, 5),
    ("S2_Anisotropic_45deg",      5.0,  100, 5),
    ("S3_Sparse_Aniso_120deg",    5.0,  100, 3),
    ("S4_HighNugget_Isotropic",   3.0,  100, 5),
    ("S5_SGS_Extreme_Aniso",     15.0,  150, 5),
    ("S6_SGS_Nested",            10.0,  150, 5),
    ("S7_SGS_Clustered",          5.0,  100, 3),
    ("S8_SGS_HighNugget",         3.0,  100, 5),
    ("S9_FewPoints",              3.0,   80, 3),
    ("S10_Duplicates",            3.0,   80, 3),
    ("S11_Colinear",              3.0,  100, 3),
    ("S12_LogNormal",             3.0,  100, 5),
    ("S13_StrongTrend",           3.0,  100, 5),
    ("S14_ExtremeAniso",         20.0,  150, 5),
]

CACHE_DIR.mkdir(exist_ok=True)

for name, max_aniso, n_trials, n_splits in SCENARIOS:
    cache_path = CACHE_DIR / f"{name}.json"
    if cache_path.exists():
        print(f"  SKIP {name} — cache exists")
        continue

    f_samples = DATA_DIR / f"{name}.csv"
    if not f_samples.exists():
        print(f"  SKIP {name} — no CSV")
        continue

    df = pd.read_csv(f_samples)
    X = df[["X", "Y"]].values.astype(np.float64)
    y = df["Value"].values.astype(np.float64)

    print(f"  FITTING {name} ({len(X)} pts, max_aniso={max_aniso}, n_trials={n_trials}) ...")
    model = AnisotropicKriging(
        n_trials=n_trials, n_splits=n_splits, max_anisotropy=max_aniso
    )
    model.fit(X, y)

    data = {
        "model_name": model.best_model_name_,
        "params": {k: v for k, v in model.best_params_.items()},
    }
    cache_path.write_text(json.dumps(data, indent=2, default=str))
    print(f"    -> {model.best_model_name_} (R={data['params'].get('range', '?'):.1f}) saved")

print("\nDone.")
